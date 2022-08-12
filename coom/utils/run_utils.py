from typing import Type

from coom.methods.agem import AGEM_SAC
from coom.methods.ewc import EWC_SAC
from coom.methods.l2 import L2_SAC
from coom.methods.mas import MAS_SAC
from coom.methods.packnet import PackNet_SAC
from coom.methods.vcl import VCL_SAC
from coom.sac.sac import SAC


def get_sac_class(cl_method: str) -> Type[SAC]:
    if cl_method is None:
        return SAC
    if cl_method == "l2":
        return L2_SAC
    if cl_method == "ewc":
        return EWC_SAC
    if cl_method == "mas":
        return MAS_SAC
    if cl_method == "vcl":
        return VCL_SAC
    if cl_method == "packnet":
        return PackNet_SAC
    if cl_method == "agem":
        return AGEM_SAC
    assert False, "Bad cl_method!"
