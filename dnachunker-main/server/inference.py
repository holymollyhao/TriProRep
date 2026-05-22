"""Inference wrapper for the DNAChunker pretrained checkpoint.

Loads `code_release/pretrained_ckpt/last.ckpt` and runs a forward pass on a
user-supplied DNA sequence, exposing the per-stage chunking decisions for
visualization.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CODE_RELEASE = REPO_ROOT / "code_release"
sys.path.insert(0, str(CODE_RELEASE))

from src.core.modeling import ChunkingModelConfig, ChunkingModelForMaskedLM  # noqa: E402
from src.data.tokenization import (  # noqa: E402
    NUCLEOTIDE_VOCAB,
    PAD_ID,
    VOCAB,
    VOCAB_SIZE,
    tokenize,
)

DEFAULT_CONFIG = CODE_RELEASE / "configs" / "pretrain" / "default.yaml"
DEFAULT_CKPT = CODE_RELEASE / "pretrained_ckpt" / "last.ckpt"
MAX_SEQ_LEN = 8192


def _id_to_base(token_id: int) -> str:
    for base, idx in NUCLEOTIDE_VOCAB.items():
        if idx == token_id:
            return base
    return "?"


class ChunkingInference:
    """Loads the pretrained ChunkingModelForMaskedLM and runs an instrumented
    forward pass that returns chunking decisions per stage."""

    def __init__(
        self,
        ckpt_path: Path = DEFAULT_CKPT,
        config_path: Path = DEFAULT_CONFIG,
        device: Optional[str] = None,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ):
        self.ckpt_path = Path(ckpt_path)
        self.config_path = Path(config_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Model weights stay in fp32 (matches the checkpoint); we autocast at
        # forward time. Casting the whole module to bf16 trips on Mamba's
        # in_proj where some tensors stay fp32 inside the kernel path.
        self.autocast_dtype = autocast_dtype

        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)
        cfg["model"]["vocab_size"] = VOCAB_SIZE
        self.config = ChunkingModelConfig(**cfg["model"])

        self.model = ChunkingModelForMaskedLM(self.config)
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        state = {
            k.removeprefix("model."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[ChunkingInference] missing={len(missing)} unexpected={len(unexpected)}"
            )
        self.model.to(self.device).eval()

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._captures: Dict[str, Any] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        backbone = self.model.net.backbone

        def make_routing_hook(stage: str):
            def hook(_module, _inputs, output):
                p_final, b_final, _entropy, p_raw, b_raw, protection_mask = output
                self._captures[f"routing_{stage}"] = {
                    "p_final": p_final.detach().float().cpu(),
                    "b_final": b_final.detach().float().cpu(),
                    "p_raw": p_raw.detach().float().cpu(),
                    "b_raw": b_raw.detach().float().cpu(),
                    "protection_mask": protection_mask.detach().cpu(),
                }
            return hook

        def make_downsampler_hook(stage: str):
            def hook(_module, _inputs, output):
                (
                    pooled,
                    chunk_lengths,
                    mask_loc,
                    pad_loc,
                    orig_idx,
                    token_to_chunk,
                    seg_conf,
                ) = output
                self._captures[f"downsampler_{stage}"] = {
                    "chunk_lengths": chunk_lengths.detach().cpu(),
                    "comp_pad_loc": pad_loc.detach().cpu(),
                    "token_to_chunk": token_to_chunk.detach().float().cpu(),
                    "segment_confidence": seg_conf.detach().float().cpu(),
                }
            return hook

        self._hooks.append(
            backbone.routing_module_stage1.register_forward_hook(
                make_routing_hook("s1")
            )
        )
        self._hooks.append(
            backbone.routing_module_stage2.register_forward_hook(
                make_routing_hook("s2")
            )
        )
        self._hooks.append(
            backbone.downsampler_s1.register_forward_hook(make_downsampler_hook("s1"))
        )
        self._hooks.append(
            backbone.downsampler_s2.register_forward_hook(make_downsampler_hook("s2"))
        )

    @staticmethod
    def sanitize(sequence: str) -> str:
        """Silently coerce input to A/C/G/T/N uppercase. Drops whitespace and
        any character not in the nucleotide vocab (N is the catch-all unknown)."""
        out_chars: List[str] = []
        for ch in sequence.upper():
            if ch in NUCLEOTIDE_VOCAB:
                out_chars.append(ch)
            elif ch.isspace():
                continue
            else:
                out_chars.append("N")
        return "".join(out_chars)[:MAX_SEQ_LEN]

    @torch.no_grad()
    def run(self, sequence: str) -> Dict[str, Any]:
        seq = self.sanitize(sequence)
        if not seq:
            raise ValueError("Empty sequence after sanitization.")

        ids = tokenize(seq, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        self._captures.clear()
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                outputs = self.model(input_ids=input_ids, return_dict=True)
        else:
            outputs = self.model(input_ids=input_ids, return_dict=True)

        logits = outputs.logits[0]  # [L0, vocab]
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        pred_bases = [_id_to_base(i) if i < len(NUCLEOTIDE_VOCAB) else "?" for i in pred_ids]

        s1 = self._captures["routing_s1"]
        s2 = self._captures["routing_s2"]
        ds1 = self._captures["downsampler_s1"]
        ds2 = self._captures["downsampler_s2"]

        L0 = input_ids.shape[1]
        b1 = s1["b_final"][0, :L0].numpy()           # [L0]
        p1 = s1["p_raw"][0, :L0].numpy()             # [L0]

        # Stage-1 chunk index for each L0 position (0-indexed, dense).
        seg1_ids = b1.cumsum().astype(int) - 1
        seg1_ids = seg1_ids.clip(min=0)
        n_chunks_s1 = int(ds1["chunk_lengths"][0].item())

        L1 = int(s2["b_final"].shape[1])
        b2 = s2["b_final"][0, :n_chunks_s1].numpy()  # [n_chunks_s1]
        p2 = s2["p_raw"][0, :n_chunks_s1].numpy()    # [n_chunks_s1]
        seg2_chunk_ids = b2.cumsum().astype(int) - 1
        seg2_chunk_ids = seg2_chunk_ids.clip(min=0)  # for each S1 chunk -> S2 chunk index
        n_chunks_s2 = int(ds2["chunk_lengths"][0].item())

        # Map each L0 position to its Stage-2 chunk via Stage-1.
        seg2_ids_per_base = [int(seg2_chunk_ids[seg1_ids[i]]) for i in range(L0)]
        # And expose the Stage-2 boundary probability at each L0 position
        # (the probability at the S1 chunk it belongs to).
        p2_per_base = [float(p2[seg1_ids[i]]) for i in range(L0)]

        return {
            "input_sequence": seq,
            "length": L0,
            "n_chunks_stage1": n_chunks_s1,
            "n_chunks_stage2": n_chunks_s2,
            "compression_ratio_stage1": float(outputs.compression_ratio_stage1.item()),
            "compression_ratio_stage2": float(outputs.compression_ratio_stage2.item()),
            "bases": list(seq),
            "predicted_bases": pred_bases,
            "p_stage1": p1.tolist(),                 # per L0
            "b_stage1": [int(x) for x in b1.tolist()],
            "stage1_chunk_id": [int(x) for x in seg1_ids.tolist()],
            "p_stage2": p2_per_base,                 # per L0 (smeared from S1 chunks)
            "stage2_chunk_id": seg2_ids_per_base,    # per L0
        }


_singleton: Optional[ChunkingInference] = None


def get_inference() -> ChunkingInference:
    global _singleton
    if _singleton is None:
        ckpt_env = os.environ.get("DNA_CHUNKER_CKPT")
        cfg_env = os.environ.get("DNA_CHUNKER_CONFIG")
        _singleton = ChunkingInference(
            ckpt_path=Path(ckpt_env) if ckpt_env else DEFAULT_CKPT,
            config_path=Path(cfg_env) if cfg_env else DEFAULT_CONFIG,
        )
    return _singleton
