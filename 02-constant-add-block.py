import torch
from torch import Tensor
import triton
import triton.language as tl

import jaxtyping
from jaxtyping import Float32, Int32

import triton_viz_test
from triton_viz_test import test

def add_kernel(x_ptr, z_ptr, N0, B0: tl.constexpr):
    pid = tl.program_id(0)
    off_x = pid * B0 + tl.arange(0, B0)
    
    x = tl.load(x_ptr + off_x)

