"""UniMedSeg-L backbone configuration and constructor."""

from .primus_icl import PrimusConfig

L_CONFIG = PrimusConfig(width=864, depth=20, heads=12)
primus_icl_l = L_CONFIG.build
