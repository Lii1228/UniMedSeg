"""UniMedSeg-S constructor and compatibility imports for the original module."""

from .backbones import MODEL_CONFIGS, build_primus
from .primus_icl import (
    EvaClean,
    InitWeights_He,
    LayerNormNd,
    PatchDecode,
    PatchDecode2d,
    PatchEmbed,
    PrimusICL,
    SplitEvaAttention,
    SplitEvaBlock,
    attention,
    convert_dim_to_conv_op,
)

primus_icl_s = MODEL_CONFIGS["S"].build

__all__ = [
    "EvaClean",
    "InitWeights_He",
    "LayerNormNd",
    "MODEL_CONFIGS",
    "PatchDecode",
    "PatchDecode2d",
    "PatchEmbed",
    "PrimusICL",
    "SplitEvaAttention",
    "SplitEvaBlock",
    "attention",
    "build_primus",
    "convert_dim_to_conv_op",
    "primus_icl_s",
]
