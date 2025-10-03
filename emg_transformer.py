#import math
from typing import Optional

import torch
import torch.nn as nn

#from timm.layers.weight_init import trunc_normal_

# CONSTANTS
WINDOW_LEN = 1000 # 0.5 sec @ 2kHz
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

# https://docs.pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        seq_idx = torch.arange(seq_len, dtype=self.theta.dtype, device=x.device)

        # Outer product of theta and position index
        # idx_theta shape: [b, s, dim // 2] or [s, dim // 2]
        idx_theta = torch.einsum("...i, j -> ...ij", seq_idx, self.theta).float()

        # rope_cache includes both the cos and sin components
        # rope_cache shape: [b, s, dim // 2, 2] or [s, dim // 2, 2]
        rope_cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

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
            bias=bias
        )

    def forward(self, x):
        B, _, C, T = x.shape
        # shared projection layer across channels
        #x = x.unsqueeze(1)  # (B, 1, C, T)
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
        norm_layer = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads, self.dim = num_heads, dim
        self.hd = dim // num_heads
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = RotaryPositionalEmbeddings(dim=self.hd, max_seq_len=1024, base=10_000)

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape  # [batch_size, total_number_tokens, embedding_dimension]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.hd)
        q, k, v = torch.chunk(qkv, 3, dim=2)   # each: [B,N,1,heads,hd]
        q = q.squeeze(2)                        # [B,N,heads,hd]
        k = k.squeeze(2)
        v = v.squeeze(2)

        # Apply RoPE on [B, N, heads, hd]
        #q = self.rope(q)
        #k = self.rope(k)

        # Permute to attention layout [B, heads, N, hd]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scale_factor = 1 / (self.hd ** 0.5)
        q = q * scale_factor
        attn = q @ k.transpose(-2, -1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, N, N)
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)

        if attn_mask is not None:
            # Check for padded tensors and set them to zero after softmax
            fully_padded_idx = attn_mask.sum(dim=-1, keepdim=True).eq(0)
            attn = attn.masked_fill(fully_padded_idx, 0.0)

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer = nn.GELU,
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
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm,
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
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)

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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.reduction = reduction
        self.in_chans = in_chans

        # after reduction, feature_dim → either embed_dim or in_chans*embed_dim
        feat_dim = embed_dim if reduction == "mean" else in_chans * embed_dim

        self.classifier = nn.Linear(feat_dim, num_classes)

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

        if self.reduction == "mean":
            x = x.reshape(B, C, p, D)
            x = x.mean(dim=1)  # (B, num_patches, embed_dim)
        elif self.reduction == "concat":
            # Reshape to (B, num_patches, embed_dim * in_chans)
            x = x.reshape(B, C, p, D)
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(B, p, C * D)
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction}")

        # pool across patches
        x = x.mean(dim=1)  # (B, feat_dim)

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
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm,
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
            bias=conv_bias
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
        assert img_size % patch_size == 0, f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"

    def forward(self, x, attn_mask=None):
        x = self.patch_embedding(x)

        # HACK: add zeros just to export "Add" op in ONNX
        x += torch.zeros(1, 400, 192)

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
        "--task", type=str, default="classification", help="Task type", choices=["classification", "pretraining"]
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
    pretrained_params = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")} # needed since original Lightning Module had [model, model_head] components
    pretrained_params = {k.replace("patch_embed.", ""): v for k, v in pretrained_params.items()}

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
    assert N == (WINDOW_LEN // PATCH_SIZE) * IN_CHANS, f"Output tokens {N} should match num_patches {(WINDOW_LEN // PATCH_SIZE) * IN_CHANS}"
    assert D == EMBED_DIM, f"Output embedding dimension {D} should match EMBED_DIM {EMBED_DIM}"
    print(f"Output shape: {output.shape}")

    if args.task == "classification":
        model_head = MlpClassificationHead(
            embed_dim=EMBED_DIM,
            num_classes=NUM_CLASSES,
            reduction="concat",
            in_chans=IN_CHANS,
        )
        model_head.load_state_dict({ k.replace("model_head.", ""): v for k, v in state_dict.items() if k.startswith("model_head.") }, strict=True)
        model_head.to(args.device)
        model_head.eval()

        logits = model_head(output)
        assert logits.shape == (B, NUM_CLASSES), f"Logits shape {logits.shape} should be {(B, NUM_CLASSES)}"
        print(f"Logits shape: {logits.shape}")
        print("Probabilities:", torch.softmax(logits, dim=-1))

    print("Model test completed successfully.")

    from torch.profiler import profile

    with profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/emg_transformer'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True) as prof:
        with torch.no_grad():
            for _ in range(10):
                prof.step()
                latent = model(dummy_input)
                logits = model_head(latent)

    model.eval()
    with torch.inference_mode():   # better than no_grad for memory & perf
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        out = model(dummy_input)
        out = model_head(out)
        torch.cuda.synchronize()
        print("peak:", torch.cuda.max_memory_allocated()/1024**2, "MB")

    import torchinfo

    torchinfo.summary(model, input_data=dummy_input, col_names=["input_size", "output_size", "num_params", "trainable"], verbose=2, depth=4)
    torchinfo.summary(model_head, input_data=torch.randn(1, 800, 192, device=args.device), col_names=["input_size", "output_size", "num_params", "trainable"], verbose=2, depth=4)