
from .deformable_detr_ecls import build


def build_model(args):
    return build(args)

