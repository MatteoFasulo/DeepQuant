import logging
import os
import re
import sys
from typing import Optional

import brevitas.nn as qnn
import h5py
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from brevitas.export import export_onnx_qcdq
from brevitas.fx.brevitas_tracer import symbolic_trace
from brevitas.graph.calibrate import bias_correction_mode, calibration_mode
from brevitas.graph.quantize import preprocess_for_quantize, quantize
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int32Bias,
    Uint8ActPerTensorFloat,
)
from torch.utils.data import Dataset
from tqdm import tqdm

# CONSTANTS
WINDOW_LEN = 1000  # 0.5 sec @ 2kHz
PATCH_SIZE = 20
IN_CHANS = 16
EMBED_DIM = 192
NUM_CLASSES = 7
N_LAYER = 8
N_HEAD = 3
MLP_RATIO = 4
QKV_BIAS = True
ATTN_DROP = 0.1
PROJ_DROP = 0.1
ACT_LAYER = nn.GELU
NORM_LAYER = nn.LayerNorm
N_TOKENS = 400  # num patches

logger = logging.getLogger("quant_logger")
handler = logging.StreamHandler(sys.stdout)  # stdout only
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)

QUANT_PARAMS = {
    "input_quant": Int8ActPerTensorFloat,
    "weight_quant": Int8WeightPerTensorFloat,
    "output_quant": Int8ActPerTensorFloat,
    "bias_quant": Int32Bias,
    "return_quant_tensor": True,
    "output_bit_width": 8,
}


class EMGDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        transform=None,
        squeeze: bool = False,
        finetune: bool = True,
        zero_pad_chans: int = None,
        zero_pad_toks: int = None,
        regression: bool = False,
    ):
        super().__init__()
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

        self.file_path = file_path
        self.transform = transform
        self.squeeze = squeeze
        self.finetune = finetune
        self.zero_pad_chans = zero_pad_chans
        self.zero_pad_toks = zero_pad_toks
        self.regression = regression

        # Read HDF5 file
        with h5py.File(file_path, "r") as h5f:
            # Data shape: (N, C, T)
            self._data = h5f["data"][:]  # => shape (N, C, T)
            # Read labels if they exist
            if "label" in h5f.keys():
                self._labels = h5f["label"][:]  # => shape (N,)
            else:
                self._labels = None

        # Compute the number of samples
        self._num_samples = self._data.shape[0]

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx: int):
        # Extract a single sample: shape (C, T)
        x_np = self._data[idx]
        x = torch.tensor(x_np, dtype=torch.float32)

        # Optionally squeeze the data => (1, C, T) if it was (C, T)
        if self.squeeze:
            x = x.unsqueeze(0)

        # Optional transform (e.g., min-max normalization to [-1, 1])
        if self.transform:
            max_val = x.amax(dim=-1, keepdim=True)
            min_val = x.amin(dim=-1, keepdim=True)
            # Avoid division by zero
            x = (x - min_val) / (max_val - min_val + 1e-10)
            x = (x - 0.5) * 2  # Rescales to [-1, 1]

        # Zero-pad along the channel dimension if requested
        if self.zero_pad_chans is not None:
            # Current shape: (C, T)
            # Transpose to (T, C), pad, then transpose back to (C, T)
            x = x.transpose(0, 1)  # => (T, C)
            x = F.pad(
                x, (0, self.zero_pad_chans), value=0.0
            )  # => (T, C + zero_pad_chans)
            x = x.transpose(0, 1)  # => (C + zero_pad_chans, T)

        # Zero-pad along the time dimension if requested
        if self.zero_pad_toks is not None:
            # Current shape: (C, T)
            # Pad on the right side of time dimension
            x = F.pad(x, (0, self.zero_pad_toks), value=0.0)

        # If we are finetuning and labels exist, return (x, y), otherwise return x
        if self.finetune:
            y = self._labels[idx]
            y = torch.tensor(y, dtype=torch.float32 if self.regression else torch.long)

        return (x, y) if self.finetune else x


def split_qkv_weight(weight: torch.Tensor):
    """
    Split a qkv weight into q,k,v according to shape.
    Returns tuple (q, k, v).
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected 2D Linear weight, got shape {tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    # concatenated along output rows: (3*H, D)
    if out_dim % 3 == 0:
        h = out_dim // 3
        q = weight[0:h, :].clone()
        k = weight[h : 2 * h, :].clone()
        v = weight[2 * h : 3 * h, :].clone()
        return q, k, v
    raise ValueError(
        f"Weight shape {weight.shape} is not a 3-way concatenation along rows or cols."
    )


def split_qkv_bias(bias: torch.Tensor):
    """
    Split qkv bias into q,k,v. bias must be 1D.
    """
    if bias is None:
        return None, None, None
    if bias.ndim != 1:
        raise ValueError(f"Expected 1D bias, got {tuple(bias.shape)}")
    n = bias.shape[0]
    if n % 3 == 0:
        h = n // 3
        return bias[0:h].clone(), bias[h : 2 * h].clone(), bias[2 * h : 3 * h].clone()
    raise ValueError(f"Bias length {n} is not divisible by 3.")


def convert_state_dict_qkv_to_qkv_separate(
    state_dict: dict,
    qkv_key_pattern=re.compile(r"(.*)\.qkv\.(weight|bias)$"),
    separate_template="{prefix}.q.{param}",
    key_replace_prefix=None,
):
    """
    Convert state_dict keys that contain '.qkv.weight' or '.qkv.bias' into
    separate '.q.weight', '.k.weight', '.v.weight' (and same for biases).
    - state_dict: original state dict (dict of tensors)
    - qkv_key_pattern: regex to find qkv keys. by default matches '<prefix>.qkv.weight' and '<prefix>.qkv.bias'
    - separate_template: how to name output keys (not commonly changed)
    - key_replace_prefix: optional function to alter prefix names (not required)
    Returns: new_state_dict
    """
    new_sd = {}
    handled_keys = set()

    for key, val in state_dict.items():
        m = qkv_key_pattern.match(key)
        if not m:
            # copy other params unchanged
            new_sd[key] = val
            continue

        prefix = m.group(1)  # part before '.qkv.weight' or '.qkv.bias'

        if key in handled_keys:
            continue

        # retrieve weight and bias (bias may be missing)
        weight_key = f"{prefix}.qkv.weight"
        bias_key = f"{prefix}.qkv.bias"

        weight = state_dict.get(weight_key, None)
        bias = state_dict.get(bias_key, None)

        if weight is None:
            raise KeyError(
                f"Expected weight at {weight_key} but it is missing in provided state_dict."
            )

        # split
        try:
            q_w, k_w, v_w = split_qkv_weight(weight)
        except Exception as e:
            raise RuntimeError(f"Error splitting {weight_key}: {e}")

        # split bias if present
        if bias is not None:
            try:
                q_b, k_b, v_b = split_qkv_bias(bias)
            except Exception as e:
                raise RuntimeError(f"Error splitting bias {bias_key}: {e}")
        else:
            q_b = k_b = v_b = None

        # build new keys and insert
        for name, w, b in (("q", q_w, q_b), ("k", k_w, k_b), ("v", v_w, v_b)):
            wkey = f"{prefix}.{name}.weight"
            new_sd[wkey] = w
            if b is not None:
                bkey = f"{prefix}.{name}.bias"
                new_sd[bkey] = b

        handled_keys.add(weight_key)
        if bias is not None:
            handled_keys.add(bias_key)

    return new_sd


class PatchEmbed(nn.Module):
    """
    1D Patch Embedding that mimics the behavior of ViT's PatchEmbed.
    Splits the input signal into patches and projects them into an embedding space.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 23,
        embed_dim: int = 1024,
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (self.img_size // self.patch_size) * self.in_chans
        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(1, self.patch_size),
            stride=(1, self.patch_size),
            bias=bias,
        )

    def forward(self, x):
        B, _, C, T = x.shape
        # shared projection layer across channels
        # x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.proj(x)
        x = x.reshape(B, self.embed_dim, C * (T // self.patch_size))
        x = x.permute(0, 2, 1)  # (B, C*(T//P), D)
        return x


# https://docs.pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
class RotaryPositionalEmbeddings(nn.Module):
    """
    FX-compatible version of Rotary Positional Embeddings (RoPE).

    Computes rotary embeddings dynamically without any caching.
    All operations are FX-traceable.

    Args:
        dim (int): Embedding dimension (usually head_dim)
        max_seq_len (int): Maximum sequence length (kept for API compatibility, unused)
        base (int): Base for geometric progression
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base

        # theta shape: [dim // 2]
        # Using linspace to avoid arange (more FX-friendly)
        indices = torch.linspace(0, dim - 2, dim // 2)
        theta = 1.0 / (self.base ** (indices / dim))
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [B, H, N, HD]
        Returns:
            tensor of shape [B, H, N, HD] with RoPE applied
        """
        b, n_h, s, h_d = x.shape
        half_dim = h_d // 2
        s = N_TOKENS  # n.tokens (must be known statically for FX)

        # Reshape to [b, s, n_h, h_d]
        x = x.permute(0, 2, 1, 3)  # [b, s, n_h, h_d]

        # Create position indices dynamically using linspace
        seq_idx = torch.linspace(
            0, s - 1, s, dtype=self.theta.dtype, device=self.theta.device
        )

        # Compute rope values on the fly
        idx_theta = seq_idx.unsqueeze(1) * self.theta.unsqueeze(0)
        cos_vals = torch.cos(idx_theta)  # [s, half_dim]
        sin_vals = torch.sin(idx_theta)  # [s, half_dim]
        rope_cache = torch.stack([cos_vals, sin_vals], dim=-1)  # [s, half_dim, 2]

        # Reshape input x into [b, s, n_h, half_dim, 2]
        x_shaped = x.reshape(b, s, n_h, half_dim, 2)

        # Add broadcast dimensions: [1, s, 1, half_dim, 2]
        rope_cache_unsq = rope_cache.unsqueeze(0).unsqueeze(2)

        # Extract real and imaginary parts
        x_real = x_shaped[..., 0]  # [b, s, n_h, half_dim]
        x_imag = x_shaped[..., 1]  # [b, s, n_h, half_dim]

        cos_vals = rope_cache_unsq[..., 0]  # [1, s, 1, half_dim]
        sin_vals = rope_cache_unsq[..., 1]  # [1, s, 1, half_dim]

        # Apply rotation: complex multiplication (a+bi)(cos+i*sin)
        out_real = x_real * cos_vals - x_imag * sin_vals
        out_imag = x_imag * cos_vals + x_real * sin_vals

        # Stack and reshape back to [b, n_h, s, h_d]
        out = torch.stack([out_real, out_imag], dim=-1)
        out = out.reshape(b, s, n_h, h_d)
        out = out.permute(0, 2, 1, 3)  # [b, n_h, s, h_d]

        return out.type_as(x)


class RoPEAttention(nn.Module):
    """
    Multi Head Attention with Rotary Position Embedding (RoPE) applied to the Q and K tensors.
    """

    def __init__(
        self,
        dim: int,
        num_heads=8,
        qkv_bias=True,
        qk_norm: bool = False,
        attn_drop=0.1,
        proj_drop=0.1,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads, self.dim = num_heads, dim
        self.hd = dim // num_heads

        # Separate projections
        self.q = nn.Linear(
            dim,
            dim,
            bias=qkv_bias,
        )
        self.k = nn.Linear(
            dim,
            dim,
            bias=qkv_bias,
        )
        self.v = nn.Linear(
            dim,
            dim,
            bias=qkv_bias,
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = RotaryPositionalEmbeddings(
            dim=self.hd,
            max_seq_len=1000,
            base=10_000,
        )

    def forward(self, x, attn_mask=None):
        """
        x: [B, N, D]
        attn_mask: [B, N] or [B, N, N] (mask values: 0 for masked/padded positions, 1 for valid)
        """
        B, N, D = x.shape

        scale_factor = self.hd**-0.5

        # Project to Q, K, V
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.reshape(B, N, self.num_heads, self.hd).transpose(1, 2)  # [B, H, N, hd]
        k = k.reshape(B, N, self.num_heads, self.hd).transpose(1, 2)  # [B, H, N, hd]
        v = v.reshape(B, N, self.num_heads, self.hd).transpose(1, 2)  # [B, H, N, hd]

        # Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)

        # Scaled Dot-Product Attention
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(
            attn_weight, p=self.attn_drop.p, train=self.training
        )
        attn_scores = attn_weight @ v
        attn_scores = attn_scores.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        attn_scores = self.proj(attn_scores)
        attn_scores = self.proj_drop(attn_scores)
        return attn_scores


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CustomAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = RoPEAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x


class MlpClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 192,
        num_classes: int = 10,
        reduction: str = "concat",
        in_chans: int = 16,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.reduction = reduction
        self.in_chans = in_chans

        # after reduction, feature_dim → either embed_dim or in_chans*embed_dim
        feat_dim = embed_dim if reduction == "mean" else in_chans * embed_dim

        self.classifier = nn.Linear(
            feat_dim,
            num_classes,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token embeddings, shape (B, num_tokens, embed_dim)
        Returns:
            logits: (B, num_classes)
        """
        B, N, D = x.shape
        C = self.in_chans
        p = N // C

        x = x.reshape(B, C, p, D)
        if self.reduction == "mean":
            x = x.mean(dim=1)  # (B, num_patches, embed_dim)
        elif self.reduction == "concat":
            # Reshape to (B, num_patches, embed_dim * in_chans)
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(B, p, C * D)
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")

        # pool across patches
        x = x.mean(dim=1)

        # apply classifier
        logits = self.classifier(x)
        return logits


class EmgTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 1000,
        patch_size: int = 20,
        in_chans: int = 16,
        embed_dim: int = 192,
        n_layer: int = 8,
        n_head: int = 3,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.1,
        proj_drop: float = 0.1,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        conv_bias: bool = True,
    ):
        super().__init__()

        # MAE encoder
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.n_layer = n_layer
        self.n_head = n_head
        self.embed_dim = embed_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.patch_embedding = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=conv_bias,
        )
        self.num_patches = self.patch_embedding.num_patches

        self.blocks = nn.ModuleList(
            [
                CustomAttentionBlock(
                    dim=embed_dim,
                    num_heads=n_head,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(n_layer)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # ----------------------------------------------
        # sanity checks
        assert (
            img_size % patch_size == 0
        ), f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

    def forward(self, x, attn_mask=None):
        x = self.patch_embedding(x)

        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = self.norm(x)

        return x  # [B, N, D]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone script to test EmgTransformer model"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the pretrained weights",
        default="/home/matteo/Scaricati/new_uci_emg_pretrained_full_finetune-epoch=08-val_loss=0.9158.ckpt",
    )
    parser.add_argument(
        "--calibration_data",
        type=str,
        default="/home/matteo/Scaricati/val.h5",
        help="Path to calibration data file",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="/home/matteo/Scaricati/test.h5",
        help="Path to test data file",
    )
    parser.add_argument(
        "--apply_calib",
        action="store_true",
        help="Apply calibration to the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported ONNX model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cpu"
    sample_input = torch.randn(1, 1, 8, WINDOW_LEN)  # UCI EMG shape

    # Instantiate the model
    encoder = EmgTransformer(
        img_size=WINDOW_LEN,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=EMBED_DIM,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        mlp_ratio=MLP_RATIO,
        qkv_bias=QKV_BIAS,
        attn_drop=ATTN_DROP,
        proj_drop=PROJ_DROP,
        act_layer=ACT_LAYER,
        norm_layer=NORM_LAYER,
    )
    model_head = MlpClassificationHead(
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        reduction="concat",
        in_chans=IN_CHANS,
        bias=True,
    )

    # load weights
    weights = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = weights["state_dict"]
    print("Loaded state dict keys: %d", len(state_dict.keys()))

    pretrained_params = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    pretrained_params = {
        k.replace("patch_embed.", ""): v for k, v in pretrained_params.items()
    }

    # Convert QKV shared weights to separate Q,K,V weights
    pretrained_params = convert_state_dict_qkv_to_qkv_separate(pretrained_params)
    output = encoder.load_state_dict(pretrained_params, strict=False)
    model_head.load_state_dict(
        {
            k.replace("model_head.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model_head.")
        },
        strict=True,
    )
    encoder.eval()
    model_head.eval()

    class FullModel(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, x):
            x = self.encoder(x)
            x = self.head(x)
            return x

    model = FullModel(encoder, model_head)
    model.to(device)
    model.eval()

    model = symbolic_trace(model)
    model = preprocess_for_quantize(
        model,
        trace_model=False,
        equalize_iters=10,
        equalize_scale_computation="range",
        equalize_merge_bias=False,
    )

    quant_identity_map = {
        "signed": (
            qnn.QuantIdentity,
            {
                "act_quant": Int8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
            },
        ),
        "unsigned": (
            qnn.QuantIdentity,
            {
                "act_quant": Uint8ActPerTensorFloat,
                "return_quant_tensor": True,
                "bit_width": 8,
            },
        ),
    }
    quantized_model = quantize(
        graph_model=model,
        quant_identity_map=quant_identity_map,
        compute_layer_map=dict(),
        quant_act_map=dict(),
    )

    print("Model\n", quantized_model)

    # Prepare dataloaders
    calib_dset = EMGDataset(args.calibration_data, finetune=True)
    calib_loader = torch.utils.data.DataLoader(
        calib_dset, batch_size=32, shuffle=False, pin_memory=True
    )
    test_dset = EMGDataset(args.test_data, finetune=True)
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=1, shuffle=False, pin_memory=True
    )

    if args.verify:
        # Load back the ONNX model to verify
        onnx_model = onnx.load("quant_model_8b.onnx")
        onnx.checker.check_model(onnx_model)

        ort_session = ort.InferenceSession(
            "quant_model_8b.onnx", providers=["CPUExecutionProvider"]
        )
        input_name = ort_session.get_inputs()[0].name

        total_correct = 0
        total_samples = 0
        for _, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = data.unsqueeze(1).cpu().numpy()
            target = target.cpu().numpy()
            ort_inputs = {input_name: data}
            ort_outs = ort_session.run(None, ort_inputs)
            output = torch.tensor(ort_outs[0])
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(torch.tensor(target).view_as(pred)).sum().item()
            total_samples += target.shape[0]
        onnx_accuracy = 100.0 * total_correct / total_samples
        print(f"ONNX Model Test Accuracy: {onnx_accuracy:.2f}%")
        sys.exit(0)

    # Calibration
    if args.apply_calib:
        logger.info("Calibrating quantized model (PTQ) ...")
        with torch.no_grad():
            with calibration_mode(quantized_model):
                for x, _ in tqdm(
                    calib_loader, desc="Calibrating", total=len(calib_loader)
                ):
                    x = x.unsqueeze(1).to(device)
                    quantized_model(x)

            # Apply bias correction if available
            with bias_correction_mode(quantized_model):
                for x, _ in tqdm(
                    calib_loader, desc="Bias Correction", total=len(calib_loader)
                ):
                    x = x.unsqueeze(1).to(device)
                    quantized_model(x)

    # Test
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data = data.unsqueeze(1).to(device)
            target = target.to(device)
            output = quantized_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Export to ONNX
    exported_model = export_onnx_qcdq(
        quantized_model,
        args=sample_input,
        export_path="quant_model_8b.onnx",
        opset_version=20,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
