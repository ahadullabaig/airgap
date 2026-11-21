"""SmolVLM2's vision tower and connector, standalone and layer by layer.

Loads the full model once, keeps only the vision transformer (embeddings, 12
encoder layers, post-layernorm) and the connector (pixel shuffle x4 + linear
projection to the language model's 960-wide input: 64 tokens per frame), and
moves them to the GPU. FP32 by default, with TF32 off everywhere (the patch
embedding is a convolution, and cuDNN allows TF32 for convolutions by default),
so gating errors are not hidden under matmul rounding.

Inputs are the encoder's view of a frame as HWC uint8 (bench/x0core.model_view),
so the dense and gated runs see exactly the same pixels.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForImageTextToText  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

GRID = 32  # 32 x 32 patch tokens per 512 x 512 frame
TOKENS = GRID * GRID


class VisionStack:
    def __init__(self, model_dir: str, device: str = "cuda", dtype: torch.dtype = torch.float32,
                 keep_full: bool = False):
        full = AutoModelForImageTextToText.from_pretrained(model_dir, local_files_only=True, dtype=dtype)
        full.eval()
        ip = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self.device, self.dtype = device, dtype
        self.vm = full.model.vision_model.to(device)
        self.conn = full.model.connector.to(device)
        self.layers = list(self.vm.encoder.layers)
        self.mean = torch.tensor(ip.image_mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        self._ones = torch.ones((1, GRID, GRID), dtype=torch.bool, device=device)
        self.full = full if keep_full else None  # only for the path check; costs host memory
        if not keep_full:
            del full

    @torch.no_grad()
    def pixels(self, u8: np.ndarray) -> torch.Tensor:
        """(B, 512, 512, 3) or (512, 512, 3) uint8 -> normalised (B, 3, 512, 512)."""
        x = torch.from_numpy(np.ascontiguousarray(u8)).to(self.device)
        if x.ndim == 3:
            x = x[None]
        x = x.permute(0, 3, 1, 2).float() / 255.0
        return ((x - self.mean) / self.std).to(self.dtype)

    @torch.no_grad()
    def embed(self, px: torch.Tensor) -> torch.Tensor:
        return self.vm.embeddings(pixel_values=px, patch_attention_mask=self._ones.expand(px.shape[0], -1, -1))

    @torch.no_grad()
    def layer(self, i: int, h: torch.Tensor) -> torch.Tensor:
        return self.layers[i](h, attention_mask=None)

    @torch.no_grad()
    def head(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Last encoder layer output -> (post-layernorm encoder output, connector output)."""
        enc = self.vm.post_layernorm(h)
        return enc, self.conn(enc)

    @torch.no_grad()
    def forward(self, px: torch.Tensor, keep_layers: bool = False):
        """Dense pass. Returns (encoder output, connector output, [embedding, layer 1..12] if keep_layers)."""
        h = self.embed(px)
        hs = [h] if keep_layers else None
        for i in range(len(self.layers)):
            h = self.layer(i, h)
            if keep_layers:
                hs.append(h)
        enc, con = self.head(h)
        return enc, con, hs

    @torch.no_grad()
    def layer_rows(self, i: int, h: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """Layer i computed for the given query rows only (keys and values from all rows), written out
        by hand with eager attention. Used to check that full-layer-then-select equals row-only compute."""
        lay = self.layers[i]
        att = lay.self_attn
        y = lay.layer_norm1(h)
        b, n, d = y.shape
        hd = att.head_dim
        q = att.q_proj(y[:, rows]).view(b, -1, att.num_heads, hd).transpose(1, 2)
        k = att.k_proj(y).view(b, n, att.num_heads, hd).transpose(1, 2)
        v = att.v_proj(y).view(b, n, att.num_heads, hd).transpose(1, 2)
        w = torch.softmax((q @ k.transpose(-1, -2)) * att.scale, dim=-1)
        o = (w @ v).transpose(1, 2).reshape(b, -1, d)
        x = h[:, rows] + att.out_proj(o)
        return x + lay.mlp(lay.layer_norm2(x))
