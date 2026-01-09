# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Legacy GPU Support - FP16 Simulation for FP8 Operations
#
# This module provides optimized Triton kernels for GPUs that don't support
# native FP8 operations (SM < 8.9), such as V100 (SM 7.0) and T600 (SM 7.5).
# Uses FP16 for computation while maintaining FP8 storage format.

import torch

import triton
import triton.language as tl
from comfy_kitchen.float_utils import (
    F8_E4M3_MAX,
    F8_E5M2_MAX,
)


# =============================================================================
# FP8 Quantization - Legacy GPU Version (using FP16/FP32 computation)
# =============================================================================


@triton.jit
def quantize_fp8_kernel_legacy(
    x_ptr,
    output_ptr,
    scale_ptr,
    lp_max,
    n_elements,
    block_size: tl.constexpr,
):
    """FP8 quantization kernel for legacy GPUs.
    
    Uses FP32 computation for maximum precision, stores result as FP16
    which will be reinterpreted as FP8 by the caller.
    
    This avoids using tl.float8e4nv which requires SM 8.9+.
    """
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load scale value from device tensor
    scale = tl.load(scale_ptr)

    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Scale and clamp using FP32 for precision
    scaled = x.to(tl.float32) / scale
    clamped = tl.maximum(tl.minimum(scaled, lp_max), -lp_max)
    
    # Store as FP32 (will be converted to FP8 by PyTorch)
    tl.store(output_ptr + offsets, clamped, mask=mask)


def quantize_per_tensor_fp8_legacy(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.float8_e4m3fn
) -> torch.Tensor:
    """Quantize tensor to FP8 format - optimized for legacy GPUs.
    
    Uses a fused Triton kernel for better performance than eager backend.
    The kernel computes in FP32 and outputs to FP32, then converts to FP8.
    
    Args:
        x: Input tensor
        scale: Scale tensor (scalar)
        output_type: FP8 dtype (float8_e4m3fn or float8_e5m2)
    
    Returns:
        Quantized FP8 tensor
    """
    if output_type == torch.float8_e4m3fn:
        lp_max = F8_E4M3_MAX
    elif output_type == torch.float8_e5m2:
        lp_max = F8_E5M2_MAX
    else:
        raise ValueError(
            f"Unsupported output_type: {output_type}. Expected torch.float8_e4m3fn or torch.float8_e5m2"
        )

    if not x.is_contiguous():
        x = x.contiguous()

    orig_shape = x.shape
    x_flat = x.flatten()
    n_elements = x_flat.numel()

    # Allocate output as FP32 first (for Triton kernel compatibility)
    output_fp32 = torch.empty_like(x_flat, dtype=torch.float32)

    # Adaptive block size based on tensor size
    if n_elements < 32768:  # < 32K elements
        block_size = 128
    elif n_elements < 131072:  # < 128K elements
        block_size = 256
    elif n_elements < 524288:  # < 512K elements
        block_size = 512
    else:
        block_size = 1024

    grid = (triton.cdiv(n_elements, block_size),)

    quantize_fp8_kernel_legacy[grid](
        x_flat,
        output_fp32,
        scale,
        lp_max,
        n_elements,
        block_size=block_size,
    )

    # Convert FP32 output to FP8
    output = output_fp32.to(output_type)
    output = output.view(orig_shape)

    return output


# =============================================================================
# FP8 Dequantization - Legacy GPU Version
# =============================================================================


@triton.jit
def dequantize_fp8_kernel_legacy(
    x_ptr,
    output_ptr,
    scale_ptr,
    n_elements,
    block_size: tl.constexpr,
):
    """FP8 dequantization kernel for legacy GPUs.
    
    Loads FP8 values (stored as uint8), converts to FP32, applies scale,
    and stores result.
    """
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements

    # Load scale value from device tensor
    scale = tl.load(scale_ptr)

    # Load input data (FP8 stored as view)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Dequantize: multiply by scale
    dequantized = x.to(tl.float32) * scale

    tl.store(output_ptr + offsets, dequantized, mask=mask)


def dequantize_per_tensor_fp8_legacy(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """Dequantize tensor from FP8 format - optimized for legacy GPUs.
    
    Uses a fused Triton kernel for better performance than eager backend.
    
    Args:
        x: Input FP8 tensor (float8_e4m3fn or float8_e5m2)
        scale: Scale tensor (scalar)
        output_type: Target dtype (float32, float16, or bfloat16)
    
    Returns:
        Dequantized tensor in specified output format
    """
    if not x.is_contiguous():
        x = x.contiguous()

    orig_shape = x.shape
    
    # Convert FP8 to FP32 first for kernel compatibility
    x_fp32 = x.to(torch.float32).flatten()
    n_elements = x_fp32.numel()

    # Allocate output as FP32
    output_fp32 = torch.empty_like(x_fp32, dtype=torch.float32)

    # Adaptive block size
    if n_elements < 32768:
        block_size = 128
    elif n_elements < 131072:
        block_size = 256
    elif n_elements < 524288:
        block_size = 512
    else:
        block_size = 1024

    grid = (triton.cdiv(n_elements, block_size),)

    dequantize_fp8_kernel_legacy[grid](
        x_fp32,
        output_fp32,
        scale,
        n_elements,
        block_size=block_size,
    )

    # Convert to requested output type
    output = output_fp32.to(output_type)
    output = output.view(orig_shape)

    return output


# =============================================================================
# Utility Functions
# =============================================================================


def is_legacy_gpu() -> bool:
    """Check if current GPU is a legacy GPU (SM < 8.9).
    
    Returns:
        True if GPU doesn't support native FP8 operations.
    """
    if not torch.cuda.is_available():
        return False
    
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    major, minor = props.major, props.minor
    
    # FP8 native support starts at SM 8.9 (Ada Lovelace)
    return (major, minor) < (8, 9)


def get_gpu_info() -> dict:
    """Get information about the current GPU.
    
    Returns:
        Dictionary with GPU name, compute capability, and capability flags.
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    major, minor = props.major, props.minor
    
    return {
        "available": True,
        "name": props.name,
        "compute_capability": (major, minor),
        "compute_capability_str": f"SM {major}.{minor}",
        "fp8_native": (major, minor) >= (8, 9),
        "fp16_tensor_core": (major, minor) >= (7, 0),
        "nvfp4_native": (major, minor) >= (10, 0),
        "total_memory_gb": props.total_memory / (1024 ** 3),
    }
