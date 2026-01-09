__all__ = [
    "apply_rope",
    "apply_rope1",
    "dequantize_nvfp4",
    "dequantize_per_tensor_fp8",
    "quantize_nvfp4",
    "quantize_per_tensor_fp8",
]

# Try to import triton and register if available
_TRITON_AVAILABLE = True
_TRITON_ERROR = None
_IS_LEGACY_GPU = False

try:
    import triton  # noqa: F401
    import torch

    # Check if this is a legacy GPU (SM < 8.9, no native FP8 support)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        _IS_LEGACY_GPU = (props.major, props.minor) < (8, 9)

    # Import standard implementations
    from .quantization import (
        dequantize_nvfp4,
        quantize_nvfp4,
    )
    from .rope import apply_rope, apply_rope1
    
    # Import FP8 operations - use legacy or standard based on GPU
    if _IS_LEGACY_GPU:
        # Use optimized legacy kernels for V100/T600/RTX 30 series
        from .legacy import (
            quantize_per_tensor_fp8_legacy as quantize_per_tensor_fp8,
            dequantize_per_tensor_fp8_legacy as dequantize_per_tensor_fp8,
        )
    else:
        # Use standard kernels for modern GPUs
        from .quantization import (
            dequantize_per_tensor_fp8,
            quantize_per_tensor_fp8,
        )

except ImportError as e:
    _TRITON_AVAILABLE = False
    _TRITON_ERROR = f"ImportError: {e!s}"


def _build_constraints() -> dict:
    import torch

    from comfy_kitchen.constraints import (
        ExactDims,
        FunctionConstraints,
        ParamConstraint,
    )

    cuda_devices = frozenset({"cuda"})
    standard_floats = frozenset({torch.float32, torch.float16, torch.bfloat16})

    constraints = {
        # FP8 quantize/dequantize work on all CUDA GPUs (using legacy kernels if needed)
        "quantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
            },
            default_devices=cuda_devices,
            # No min_compute_capability - legacy kernels support all CUDA GPUs
        ),
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn, torch.float8_e5m2}),
                ),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
            # No min_compute_capability - legacy kernels support all CUDA GPUs
        ),
        # NVFP4 quantize uses SM100 PTX instructions
        "quantize_nvfp4": FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=standard_floats,
                    shape_rules=(ExactDims(2),),
                ),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            },
            default_devices=cuda_devices,
            min_compute_capability=(10, 0),  # SM100 required for cvt.rn.satfinite.e2m1x2.f32
        ),
        # Uses inline PTX: cvt.rn.f16x2.e2m1x2 (SM100/Blackwell instruction)
        "dequantize_nvfp4": FunctionConstraints(
            params={
                "qx": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2),),
                ),
                "per_tensor_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "block_scales": ParamConstraint(
                    dtypes=frozenset({torch.float8_e4m3fn}),
                ),
                "output_type": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
            min_compute_capability=(10, 0),  # SM100 required for cvt.rn.f16x2.e2m1x2
        ),
        "apply_rope1": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
        ),
        "apply_rope": FunctionConstraints(
            params={
                "xq": ParamConstraint(dtypes=standard_floats),
                "xk": ParamConstraint(dtypes=standard_floats),
                "freqs_cis": ParamConstraint(dtypes=standard_floats),
            },
            default_devices=cuda_devices,
        ),
    }

    return constraints


def _register():
    import torch

    from comfy_kitchen.registry import registry

    if not _TRITON_AVAILABLE:
        registry.mark_unavailable("triton", _TRITON_ERROR)
        return

    if not torch.cuda.is_available():
        registry.mark_unavailable("triton", "CUDA not available on this system")
        return

    registry.register(
        name="triton",
        module=__import__(__name__, fromlist=__all__),
        capabilities=_build_constraints(),
    )


_register()
