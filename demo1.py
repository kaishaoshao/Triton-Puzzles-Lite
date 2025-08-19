import triton
import triton.language as tl

"""
Demo1
这是一个关于内存加载操作的示例。它通过 arange在内存上创建序列。默
认情况下，PyTorch 张量的索引遵循列、行、深度或从右到左的顺序。
该操作还以掩码（mask）作为第二个参数，这在 Triton 中至关重要，
因为所有张量形状必须是 2 的幂次方。




"""


@triton.jit
def demo1(x_ptr):
    
