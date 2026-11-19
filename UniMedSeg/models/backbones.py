"""Select the backbone that matches a released UniMedSeg checkpoint."""

from .primus_icl import PrimusConfig
from .PrimusL_ICL import L_CONFIG

MODEL_CONFIGS = {"S": PrimusConfig(width=432, depth=12, heads=6), "L": L_CONFIG}


def build_primus(variant, *args, **kwargs):
    return MODEL_CONFIGS[variant].build(*args, **kwargs)
