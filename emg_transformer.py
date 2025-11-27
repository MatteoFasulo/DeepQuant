# import math
from typing import Optional

import torch
import torch.nn as nn

# from timm.layers.weight_init import trunc_normal_

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
        # q = self.rope(q)
        # k = self.rope(k)

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
        x = x + self.mlp(self.norm2(x))
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

        self.classifier = nn.Linear(feat_dim, num_classes, bias=bias)

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
        "--ckpt", type=str, required=True, help="Path to the pretrained weights"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        help="Task type",
        choices=["classification", "pretraining"],
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    args = parser.parse_args()

    # Instantiate the model
    model = EmgTransformer(
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

    print(f"Loading pretrained weights from {args.ckpt}")
    # Load pretrained weights
    weights = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Fixing missing keys
    # Load state dict, remove model. prefix if present
    # Then rename the patch embedder
    state_dict = weights["state_dict"]
    pretrained_params = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }  # needed since original Lightning Module had [model, model_head] components
    pretrained_params = {
        k.replace("patch_embed.", ""): v for k, v in pretrained_params.items()
    }

    model.load_state_dict(pretrained_params, strict=True)
    model.to(args.device)
    model.eval()

    dummy_input = torch.randn(1, 16, 1000, device=args.device)  # [B, C, T]
    print("Testing the model with dummy input...")

    B, C, T = dummy_input.shape
    assert T == WINDOW_LEN, f"Input length {T} should match WINDOW_LEN {WINDOW_LEN}"
    assert C == IN_CHANS, f"Input channels {C} should match IN_CHANS {IN_CHANS}"
    print(f"Input shape: {dummy_input.shape}")

    output = model(dummy_input)
    _, N, D = output.shape
    assert (
        N == (WINDOW_LEN // PATCH_SIZE) * IN_CHANS
    ), f"Output tokens {N} should match num_patches {(WINDOW_LEN // PATCH_SIZE) * IN_CHANS}"
    assert (
        D == EMBED_DIM
    ), f"Output embedding dimension {D} should match EMBED_DIM {EMBED_DIM}"
    print(f"Output shape: {output.shape}")

    if args.task == "classification":
        model_head = MlpClassificationHead(
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            reduction="concat",
            in_chans=IN_CHANS,
        )
        model_head.load_state_dict(
            {
                k.replace("model_head.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model_head.")
            },
            strict=True,
        )
        model_head.to(args.device)
        model_head.eval()

        logits = model_head(output)
        assert logits.shape == (
            B,
            NUM_CLASSES,
        ), f"Logits shape {logits.shape} should be {(B, NUM_CLASSES)}"
        print(f"Logits shape: {logits.shape}")
        print("Probabilities:", torch.softmax(logits, dim=-1))

    print("Model test completed successfully.")

    from torch.profiler import profile

    with profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "./log/emg_transformer"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                prof.step()
                latent = model(dummy_input)
                logits = model_head(latent)

    model.eval()
    with torch.inference_mode():  # better than no_grad for memory & perf
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out = model(dummy_input)
        out = model_head(out)
        torch.cuda.synchronize()
        print("peak:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

    import torchinfo

    torchinfo.summary(
        model,
        input_data=dummy_input,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=2,
        depth=4,
    )
    torchinfo.summary(
        model_head,
        input_data=torch.randn(1, 800, 192, device=args.device),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        verbose=2,
        depth=4,
    )
