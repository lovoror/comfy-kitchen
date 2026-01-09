# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# GPU Compatibility Test Suite
#
# Tests for V100 (SM 7.0), T600 (SM 7.5), and other GPU architectures.
# Verifies that the library correctly handles different GPU capabilities
# and automatically selects the appropriate backend.

from __future__ import annotations

import pytest
import torch

import comfy_kitchen as ck


def get_compute_capability() -> tuple[int, int] | None:
    """Get the compute capability of the current CUDA device.
    
    Returns:
        Tuple of (major, minor) version, or None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (props.major, props.minor)


def get_gpu_name() -> str | None:
    """Get the name of the current CUDA device."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(torch.cuda.current_device()).name


def is_legacy_gpu() -> bool:
    """Check if current GPU is a legacy GPU (SM < 8.9)."""
    cc = get_compute_capability()
    if cc is None:
        return False
    return cc < (8, 9)


def is_v100() -> bool:
    """Check if current GPU is V100 (SM 7.0)."""
    cc = get_compute_capability()
    return cc == (7, 0)


def is_t600_or_turing() -> bool:
    """Check if current GPU is T600 or other Turing GPU (SM 7.5)."""
    cc = get_compute_capability()
    return cc == (7, 5)


class TestGPUDetection:
    """Tests for GPU capability detection."""
    
    @pytest.mark.cuda
    def test_compute_capability_detection(self):
        """Test that compute capability is correctly detected."""
        cc = get_compute_capability()
        assert cc is not None, "CUDA should be available for this test"
        
        major, minor = cc
        assert major >= 1, "Major version should be at least 1"
        assert minor >= 0, "Minor version should be non-negative"
        
        gpu_name = get_gpu_name()
        print(f"\n  GPU: {gpu_name}")
        print(f"  Compute Capability: SM {major}.{minor}")
        print(f"  Legacy GPU: {is_legacy_gpu()}")
    
    @pytest.mark.cuda
    def test_backend_availability(self):
        """Test that backends are correctly identified as available."""
        backends = ck.list_backends()
        
        # Eager should always be available
        assert backends["eager"]["available"] is True
        
        # Report backend status
        print("\n  Backend Status:")
        for name, info in backends.items():
            status = "✓" if info["available"] else "✗"
            if info.get("disabled"):
                status += " (disabled)"
            if info.get("unavailable_reason"):
                status += f" ({info['unavailable_reason'][:50]}...)"
            print(f"    {name}: {status}")
            if info["available"]:
                print(f"      Capabilities: {info['capabilities']}")


class TestFP8Operations:
    """Tests for FP8 quantization/dequantization on all GPU types."""
    
    @pytest.fixture
    def test_tensor(self):
        """Create a test tensor."""
        return torch.randn(256, 512, device="cuda", dtype=torch.float16)
    
    @pytest.fixture
    def scale_tensor(self):
        """Create a scale tensor."""
        return torch.tensor([1.0], device="cuda", dtype=torch.float32)
    
    @pytest.mark.cuda
    def test_fp8_quantize_basic(self, test_tensor, scale_tensor):
        """Test basic FP8 quantization works on current GPU."""
        result = ck.quantize_per_tensor_fp8(test_tensor, scale_tensor)
        
        assert result.shape == test_tensor.shape
        assert result.dtype == torch.float8_e4m3fn
        assert result.device == test_tensor.device
    
    @pytest.mark.cuda
    def test_fp8_dequantize_basic(self, test_tensor, scale_tensor):
        """Test basic FP8 dequantization works on current GPU."""
        # Quantize first
        quantized = ck.quantize_per_tensor_fp8(test_tensor, scale_tensor)
        
        # Dequantize
        result = ck.dequantize_per_tensor_fp8(quantized, scale_tensor, output_type=torch.float16)
        
        assert result.shape == test_tensor.shape
        assert result.dtype == torch.float16
        assert result.device == test_tensor.device
    
    @pytest.mark.cuda
    def test_fp8_roundtrip_accuracy(self, test_tensor, scale_tensor):
        """Test that FP8 roundtrip maintains reasonable accuracy."""
        # Use a tensor with values in FP8 range
        x = torch.clamp(test_tensor, -448.0, 448.0)
        
        # Quantize and dequantize
        quantized = ck.quantize_per_tensor_fp8(x, scale_tensor)
        dequantized = ck.dequantize_per_tensor_fp8(quantized, scale_tensor, output_type=torch.float16)
        
        # Check relative error (FP8 E4M3 has ~0.5% precision)
        rel_error = torch.abs(dequantized - x) / (torch.abs(x) + 1e-6)
        mean_rel_error = rel_error.mean().item()
        
        # Allow up to 10% mean relative error due to FP8 precision limits
        assert mean_rel_error < 0.1, f"Mean relative error {mean_rel_error:.4f} exceeds threshold"
        print(f"\n  FP8 roundtrip mean relative error: {mean_rel_error:.4%}")
    
    @pytest.mark.cuda
    def test_fp8_e5m2_format(self, test_tensor, scale_tensor):
        """Test FP8 E5M2 format (alternative FP8 type)."""
        result = ck.quantize_per_tensor_fp8(test_tensor, scale_tensor, output_type=torch.float8_e5m2)
        
        assert result.dtype == torch.float8_e5m2


class TestRoPEOperations:
    """Tests for RoPE (Rotary Position Embedding) on all GPU types."""
    
    @pytest.mark.cuda
    def test_rope_basic(self):
        """Test basic RoPE operation works on current GPU."""
        batch, seq, heads, dim = 2, 128, 8, 64
        
        xq = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.float16)
        xk = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.float16)
        
        # Create frequency tensor (6D for CUDA backend compatibility)
        freqs_cis = torch.randn(1, seq, 1, 1, dim // 2, 2, device="cuda", dtype=torch.float16)
        
        xq_out, xk_out = ck.apply_rope(xq, xk, freqs_cis)
        
        assert xq_out.shape == xq.shape
        assert xk_out.shape == xk.shape
        assert xq_out.dtype == xq.dtype
    
    @pytest.mark.cuda
    def test_rope1_basic(self):
        """Test single-tensor RoPE operation."""
        batch, seq, heads, dim = 2, 128, 8, 64
        
        x = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.float16)
        freqs_cis = torch.randn(1, seq, 1, 1, dim // 2, 2, device="cuda", dtype=torch.float16)
        
        x_out = ck.apply_rope1(x, freqs_cis)
        
        assert x_out.shape == x.shape
        assert x_out.dtype == x.dtype


class TestBackendFallback:
    """Tests for backend auto-selection and fallback mechanisms."""
    
    @pytest.mark.cuda
    def test_triton_selected_for_fp8(self):
        """Test that Triton backend is selected for FP8 on CUDA."""
        backends = ck.list_backends()
        
        # Check if triton is available
        if not backends.get("triton", {}).get("available", False):
            pytest.skip("Triton not available")
        
        # FP8 quantize should be in triton capabilities
        triton_caps = backends["triton"]["capabilities"]
        assert "quantize_per_tensor_fp8" in triton_caps
        assert "dequantize_per_tensor_fp8" in triton_caps
    
    @pytest.mark.cuda
    def test_eager_fallback(self):
        """Test that eager backend works as fallback."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        
        # Force eager backend
        with ck.use_backend("eager"):
            result = ck.quantize_per_tensor_fp8(x, scale)
        
        assert result.shape == x.shape
        assert result.dtype == torch.float8_e4m3fn
    
    @pytest.mark.cuda
    def test_nvfp4_fallback_on_legacy(self):
        """Test NVFP4 operations fall back to eager on legacy GPUs."""
        if not is_legacy_gpu():
            pytest.skip("Not a legacy GPU")
        
        x = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        
        # NVFP4 should work via eager backend on legacy GPUs
        qx, block_scales = ck.quantize_nvfp4(x, scale)
        
        assert qx.dtype == torch.uint8
        assert block_scales.dtype == torch.float8_e4m3fn


class TestPerformance:
    """Performance comparison tests (optional, for benchmarking)."""
    
    @pytest.mark.cuda
    @pytest.mark.slow
    def test_fp8_quantize_performance(self):
        """Benchmark FP8 quantization performance."""
        sizes = [(256, 256), (1024, 1024), (2048, 2048)]
        
        print("\n  FP8 Quantization Benchmark:")
        print(f"  GPU: {get_gpu_name()} (SM {get_compute_capability()})")
        
        for rows, cols in sizes:
            x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
            scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
            
            # Warmup
            for _ in range(10):
                _ = ck.quantize_per_tensor_fp8(x, scale)
            
            torch.cuda.synchronize()
            
            # Benchmark
            import time
            n_iters = 100
            start = time.perf_counter()
            for _ in range(n_iters):
                _ = ck.quantize_per_tensor_fp8(x, scale)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            avg_ms = (elapsed / n_iters) * 1000
            throughput = (rows * cols * 2) / (elapsed / n_iters) / 1e9  # GB/s (FP16 input)
            
            print(f"    {rows}x{cols}: {avg_ms:.3f} ms/iter, {throughput:.1f} GB/s")
    
    @pytest.mark.cuda
    @pytest.mark.slow
    def test_eager_vs_triton_comparison(self):
        """Compare eager and triton backend performance."""
        backends = ck.list_backends()
        
        if not backends.get("triton", {}).get("available", False):
            pytest.skip("Triton not available for comparison")
        
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        
        import time
        n_iters = 50
        
        results = {}
        for backend_name in ["eager", "triton"]:
            # Warmup
            with ck.use_backend(backend_name):
                for _ in range(10):
                    _ = ck.quantize_per_tensor_fp8(x, scale)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            with ck.use_backend(backend_name):
                for _ in range(n_iters):
                    _ = ck.quantize_per_tensor_fp8(x, scale)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            results[backend_name] = (elapsed / n_iters) * 1000
        
        print(f"\n  Backend Comparison (1024x1024 FP8 quantize):")
        print(f"    Eager:  {results['eager']:.3f} ms")
        print(f"    Triton: {results['triton']:.3f} ms")
        
        if results['triton'] < results['eager']:
            speedup = results['eager'] / results['triton']
            print(f"    Triton speedup: {speedup:.2f}x")
        else:
            slowdown = results['triton'] / results['eager']
            print(f"    Triton slower by: {slowdown:.2f}x")


class TestV100Specific:
    """V100-specific tests (only run on V100)."""
    
    @pytest.fixture(autouse=True)
    def skip_if_not_v100(self):
        if not is_v100():
            pytest.skip("V100 GPU required")
    
    @pytest.mark.cuda
    def test_v100_detected_as_legacy(self):
        """Test that V100 is correctly detected as legacy GPU."""
        assert is_legacy_gpu(), "V100 should be detected as legacy GPU"
        
        cc = get_compute_capability()
        assert cc == (7, 0), f"V100 should have SM 7.0, got SM {cc}"
    
    @pytest.mark.cuda
    def test_v100_fp8_operations(self):
        """Test FP8 operations work correctly on V100."""
        x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        
        # Quantize
        qx = ck.quantize_per_tensor_fp8(x, scale)
        assert qx.dtype == torch.float8_e4m3fn
        
        # Dequantize
        dx = ck.dequantize_per_tensor_fp8(qx, scale, output_type=torch.float16)
        assert dx.dtype == torch.float16


class TestT600Specific:
    """T600/Turing-specific tests (only run on Turing GPUs)."""
    
    @pytest.fixture(autouse=True)
    def skip_if_not_turing(self):
        if not is_t600_or_turing():
            pytest.skip("Turing GPU (SM 7.5) required")
    
    @pytest.mark.cuda
    def test_turing_detected_as_legacy(self):
        """Test that Turing GPUs are correctly detected as legacy."""
        assert is_legacy_gpu(), "Turing should be detected as legacy GPU"
        
        cc = get_compute_capability()
        assert cc == (7, 5), f"Turing should have SM 7.5, got SM {cc}"
    
    @pytest.mark.cuda
    def test_turing_memory_efficient(self):
        """Test memory-efficient operations for T600 (limited VRAM)."""
        # T600 has 4GB VRAM, test with smaller tensors
        x = torch.randn(128, 128, device="cuda", dtype=torch.float16)
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        
        qx = ck.quantize_per_tensor_fp8(x, scale)
        dx = ck.dequantize_per_tensor_fp8(qx, scale, output_type=torch.float16)
        
        assert qx.shape == x.shape
        assert dx.shape == x.shape


class TestLegacyGPUGeneric:
    """Tests that apply to all legacy GPUs (SM < 8.9)."""
    
    @pytest.fixture(autouse=True)
    def skip_if_modern_gpu(self):
        if not is_legacy_gpu():
            pytest.skip("Legacy GPU required")
    
    @pytest.mark.cuda
    def test_legacy_kernel_used(self):
        """Verify that legacy kernels are selected on legacy GPUs."""
        # Import the triton backend to check which kernels are loaded
        from comfy_kitchen.backends import triton as triton_backend
        
        # Check if legacy mode is active
        assert triton_backend._IS_LEGACY_GPU, "Legacy GPU should be detected"
    
    @pytest.mark.cuda
    def test_all_fp8_dtypes_supported(self):
        """Test all FP8 data types work on legacy GPUs."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        
        for fp8_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            qx = ck.quantize_per_tensor_fp8(x, scale, output_type=fp8_dtype)
            assert qx.dtype == fp8_dtype, f"Expected {fp8_dtype}, got {qx.dtype}"
    
    @pytest.mark.cuda
    def test_various_tensor_sizes(self):
        """Test various tensor sizes work correctly on legacy GPUs."""
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        
        sizes = [
            (32, 32),      # Small
            (128, 128),    # Medium
            (512, 512),    # Large
            (1, 1024),     # Row vector
            (1024, 1),     # Column vector
            (100, 200),    # Non-power-of-2
        ]
        
        for rows, cols in sizes:
            x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
            qx = ck.quantize_per_tensor_fp8(x, scale)
            dx = ck.dequantize_per_tensor_fp8(qx, scale, output_type=torch.float16)
            
            assert qx.shape == (rows, cols)
            assert dx.shape == (rows, cols)
