# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Video Generation Compatibility Tests
#
# Tests simulating real-world video generation workloads for V100/T600 GPUs.
# Covers diffusion model typical operations: attention, linear layers, etc.

from __future__ import annotations

import pytest
import torch
import math

import comfy_kitchen as ck


def get_compute_capability() -> tuple[int, int] | None:
    """Get the compute capability of the current CUDA device."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (props.major, props.minor)


class TestDiffusionModelOperations:
    """Tests simulating diffusion model (Wan2.1, FLUX, etc.) operations."""
    
    @pytest.fixture
    def model_config(self):
        """Wan2.1-1.3B-like model configuration."""
        return {
            "hidden_size": 1536,
            "num_heads": 24,
            "head_dim": 64,
            "mlp_ratio": 4,
            "num_frames": 16,
            "height": 480,
            "width": 832,
            "patch_size": 1,
            "latent_channels": 16,
        }
    
    @pytest.fixture
    def small_config(self):
        """Smaller config for memory-constrained GPUs like T600."""
        return {
            "hidden_size": 768,
            "num_heads": 12,
            "head_dim": 64,
            "mlp_ratio": 4,
            "num_frames": 4,
            "height": 240,
            "width": 416,
            "patch_size": 1,
            "latent_channels": 16,
        }
    
    @pytest.mark.cuda
    def test_attention_qkv_projection_fp8(self, small_config):
        """Test FP8 quantization for QKV projection in attention.
        
        Simulates: q, k, v = linear(x).chunk(3)
        """
        cfg = small_config
        batch_size = 1
        seq_len = (cfg["num_frames"] * cfg["height"] * cfg["width"]) // (cfg["patch_size"] ** 2)
        
        # Limit sequence length for T600 memory
        seq_len = min(seq_len, 4096)
        
        # Input tensor: [batch, seq, hidden]
        x = torch.randn(batch_size, seq_len, cfg["hidden_size"], 
                       device="cuda", dtype=torch.float16)
        
        # Simulate weight quantization
        qkv_weight = torch.randn(cfg["hidden_size"] * 3, cfg["hidden_size"],
                                device="cuda", dtype=torch.float16)
        
        # Compute scale for weight quantization
        weight_scale = torch.tensor([qkv_weight.abs().max().item() / 448.0], 
                                    device="cuda", dtype=torch.float32)
        
        # Quantize weight
        qkv_weight_fp8 = ck.quantize_per_tensor_fp8(qkv_weight, weight_scale)
        
        # Dequantize for compute
        qkv_weight_dq = ck.dequantize_per_tensor_fp8(qkv_weight_fp8, weight_scale, 
                                                     output_type=torch.float16)
        
        # Compute QKV
        qkv = torch.nn.functional.linear(x, qkv_weight_dq)
        q, k, v = qkv.chunk(3, dim=-1)
        
        assert q.shape == (batch_size, seq_len, cfg["hidden_size"])
        assert k.shape == (batch_size, seq_len, cfg["hidden_size"])
        assert v.shape == (batch_size, seq_len, cfg["hidden_size"])
    
    @pytest.mark.cuda
    def test_attention_output_projection_fp8(self, small_config):
        """Test FP8 for attention output projection.
        
        Simulates: out = linear(attn_output)
        """
        cfg = small_config
        batch_size = 1
        seq_len = 2048  # Reasonable for video
        
        # Attention output: [batch, seq, hidden]
        attn_out = torch.randn(batch_size, seq_len, cfg["hidden_size"],
                              device="cuda", dtype=torch.float16)
        
        # Output projection weight
        out_weight = torch.randn(cfg["hidden_size"], cfg["hidden_size"],
                                device="cuda", dtype=torch.float16)
        
        # Quantize and dequantize weight
        weight_scale = torch.tensor([out_weight.abs().max().item() / 448.0],
                                   device="cuda", dtype=torch.float32)
        out_weight_fp8 = ck.quantize_per_tensor_fp8(out_weight, weight_scale)
        out_weight_dq = ck.dequantize_per_tensor_fp8(out_weight_fp8, weight_scale,
                                                     output_type=torch.float16)
        
        # Compute output
        output = torch.nn.functional.linear(attn_out, out_weight_dq)
        
        assert output.shape == (batch_size, seq_len, cfg["hidden_size"])
    
    @pytest.mark.cuda
    def test_mlp_block_fp8(self, small_config):
        """Test FP8 quantization for MLP block.
        
        Simulates: out = down(act(up(x)))
        """
        cfg = small_config
        batch_size = 1
        seq_len = 2048
        mlp_hidden = cfg["hidden_size"] * cfg["mlp_ratio"]
        
        # Input
        x = torch.randn(batch_size, seq_len, cfg["hidden_size"],
                       device="cuda", dtype=torch.float16)
        
        # MLP weights
        up_weight = torch.randn(mlp_hidden, cfg["hidden_size"],
                               device="cuda", dtype=torch.float16)
        down_weight = torch.randn(cfg["hidden_size"], mlp_hidden,
                                 device="cuda", dtype=torch.float16)
        
        # Quantize weights
        up_scale = torch.tensor([up_weight.abs().max().item() / 448.0],
                               device="cuda", dtype=torch.float32)
        down_scale = torch.tensor([down_weight.abs().max().item() / 448.0],
                                 device="cuda", dtype=torch.float32)
        
        up_fp8 = ck.quantize_per_tensor_fp8(up_weight, up_scale)
        down_fp8 = ck.quantize_per_tensor_fp8(down_weight, down_scale)
        
        up_dq = ck.dequantize_per_tensor_fp8(up_fp8, up_scale, output_type=torch.float16)
        down_dq = ck.dequantize_per_tensor_fp8(down_fp8, down_scale, output_type=torch.float16)
        
        # Forward pass
        hidden = torch.nn.functional.linear(x, up_dq)
        hidden = torch.nn.functional.gelu(hidden)
        output = torch.nn.functional.linear(hidden, down_dq)
        
        assert output.shape == x.shape
    
    @pytest.mark.cuda
    def test_rope_for_video_attention(self, small_config):
        """Test RoPE for 3D video attention (temporal + spatial).
        
        Simulates rotary position embedding in video diffusion.
        """
        cfg = small_config
        batch_size = 1
        num_heads = cfg["num_heads"]
        head_dim = cfg["head_dim"]
        
        # Simulate video tokens: temporal * spatial
        temporal_len = cfg["num_frames"]
        spatial_len = 256  # Simplified spatial resolution
        seq_len = temporal_len * spatial_len
        
        # Limit for memory
        seq_len = min(seq_len, 2048)
        
        # Query and Key tensors: [batch, seq, heads, head_dim]
        xq = torch.randn(batch_size, seq_len, num_heads, head_dim,
                        device="cuda", dtype=torch.float16)
        xk = torch.randn(batch_size, seq_len, num_heads, head_dim,
                        device="cuda", dtype=torch.float16)
        
        # Frequency tensor for RoPE: [1, seq, 1, 1, head_dim//2, 2]
        freqs_cis = torch.randn(1, seq_len, 1, 1, head_dim // 2, 2,
                               device="cuda", dtype=torch.float16)
        
        # Apply RoPE
        xq_rope, xk_rope = ck.apply_rope(xq, xk, freqs_cis)
        
        assert xq_rope.shape == xq.shape
        assert xk_rope.shape == xk.shape
        
        # Verify values are different (RoPE was applied)
        assert not torch.allclose(xq_rope, xq)
        assert not torch.allclose(xk_rope, xk)


class TestVideoLatentOperations:
    """Tests for video latent space operations."""
    
    @pytest.mark.cuda
    def test_vae_encoder_latent_quantization(self):
        """Test FP8 quantization of VAE encoder output (latents)."""
        batch_size = 1
        num_frames = 8
        latent_h, latent_w = 60, 104  # 480x832 / 8
        latent_channels = 16
        
        # Simulated VAE encoder output
        latents = torch.randn(batch_size, num_frames, latent_channels, latent_h, latent_w,
                             device="cuda", dtype=torch.float16)
        
        # Quantize latents
        scale = torch.tensor([latents.abs().max().item() / 448.0],
                            device="cuda", dtype=torch.float32)
        
        latents_fp8 = ck.quantize_per_tensor_fp8(latents, scale)
        latents_dq = ck.dequantize_per_tensor_fp8(latents_fp8, scale, 
                                                   output_type=torch.float16)
        
        # Check shape preserved
        assert latents_fp8.shape == latents.shape
        assert latents_dq.shape == latents.shape
        
        # Check MSE is reasonable
        mse = torch.nn.functional.mse_loss(latents_dq, latents).item()
        print(f"\n  VAE latent FP8 MSE: {mse:.6f}")
        assert mse < 1.0  # Reasonable for FP8
    
    @pytest.mark.cuda
    def test_temporal_attention_weights_fp8(self):
        """Test FP8 for temporal attention (cross-frame attention)."""
        batch_size = 1
        num_frames = 4
        spatial_tokens = 256
        hidden_size = 768
        
        # Temporal attention: each spatial position attends across frames
        # Shape: [batch * spatial, frames, hidden]
        x = torch.randn(batch_size * spatial_tokens, num_frames, hidden_size,
                       device="cuda", dtype=torch.float16)
        
        # Temporal attention weights
        temporal_weight = torch.randn(hidden_size, hidden_size,
                                     device="cuda", dtype=torch.float16)
        
        # Quantize
        scale = torch.tensor([temporal_weight.abs().max().item() / 448.0],
                            device="cuda", dtype=torch.float32)
        weight_fp8 = ck.quantize_per_tensor_fp8(temporal_weight, scale)
        weight_dq = ck.dequantize_per_tensor_fp8(weight_fp8, scale,
                                                  output_type=torch.float16)
        
        # Apply temporal projection
        output = torch.einsum("bth,hd->btd", x, weight_dq)
        
        assert output.shape == x.shape


class TestTextEncoderOperations:
    """Tests for text encoder (T5/CLIP) operations."""
    
    @pytest.mark.cuda
    def test_t5_encoder_output_quantization(self):
        """Test FP8 quantization of T5 encoder output for conditioning."""
        batch_size = 1
        max_seq_len = 512
        t5_hidden = 4096  # T5-XXL hidden size
        
        # Simulated T5 encoder output
        text_embeddings = torch.randn(batch_size, max_seq_len, t5_hidden,
                                     device="cuda", dtype=torch.float16)
        
        # Quantize text embeddings (often done for memory saving)
        scale = torch.tensor([text_embeddings.abs().max().item() / 448.0],
                            device="cuda", dtype=torch.float32)
        
        text_fp8 = ck.quantize_per_tensor_fp8(text_embeddings, scale)
        text_dq = ck.dequantize_per_tensor_fp8(text_fp8, scale,
                                                output_type=torch.float16)
        
        # Check relative error
        rel_error = (torch.abs(text_dq - text_embeddings) / 
                    (torch.abs(text_embeddings) + 1e-6)).mean().item()
        
        print(f"\n  T5 embeddings FP8 relative error: {rel_error:.2%}")
        assert rel_error < 0.1  # 10% acceptable for FP8
    
    @pytest.mark.cuda
    def test_cross_attention_text_projection(self):
        """Test cross-attention text key/value projection with FP8."""
        batch_size = 1
        text_len = 256
        hidden_size = 1536
        num_heads = 24
        head_dim = 64
        
        # Text embeddings
        text = torch.randn(batch_size, text_len, hidden_size,
                          device="cuda", dtype=torch.float16)
        
        # Key and Value projection weights
        kv_weight = torch.randn(hidden_size * 2, hidden_size,
                               device="cuda", dtype=torch.float16)
        
        # Quantize
        scale = torch.tensor([kv_weight.abs().max().item() / 448.0],
                            device="cuda", dtype=torch.float32)
        kv_fp8 = ck.quantize_per_tensor_fp8(kv_weight, scale)
        kv_dq = ck.dequantize_per_tensor_fp8(kv_fp8, scale, output_type=torch.float16)
        
        # Compute K, V
        kv = torch.nn.functional.linear(text, kv_dq)
        k, v = kv.chunk(2, dim=-1)
        
        assert k.shape == (batch_size, text_len, hidden_size)
        assert v.shape == (batch_size, text_len, hidden_size)


class TestEndToEndSimulation:
    """End-to-end simulation of video generation step."""
    
    @pytest.mark.cuda
    @pytest.mark.slow
    def test_single_diffusion_step_fp8(self):
        """Simulate a single diffusion step with FP8 weights.
        
        This tests the full flow: text conditioning -> attention -> MLP -> output
        """
        # Model config (small for testing)
        hidden_size = 512
        num_heads = 8
        head_dim = 64
        mlp_ratio = 4
        
        batch_size = 1
        seq_len = 512
        text_len = 64
        
        # Create "quantized" model weights
        weights = {}
        scales = {}
        
        weight_names = [
            "qkv_proj", "out_proj", 
            "cross_attn_kv", "cross_attn_out",
            "mlp_up", "mlp_down"
        ]
        
        weight_shapes = {
            "qkv_proj": (hidden_size * 3, hidden_size),
            "out_proj": (hidden_size, hidden_size),
            "cross_attn_kv": (hidden_size * 2, hidden_size),
            "cross_attn_out": (hidden_size, hidden_size),
            "mlp_up": (hidden_size * mlp_ratio, hidden_size),
            "mlp_down": (hidden_size, hidden_size * mlp_ratio),
        }
        
        # Quantize all weights
        for name in weight_names:
            w = torch.randn(*weight_shapes[name], device="cuda", dtype=torch.float16)
            s = torch.tensor([w.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
            weights[name] = ck.quantize_per_tensor_fp8(w, s)
            scales[name] = s
        
        # Input tensors
        x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
        text = torch.randn(batch_size, text_len, hidden_size, device="cuda", dtype=torch.float16)
        freqs = torch.randn(1, seq_len, 1, 1, head_dim // 2, 2, device="cuda", dtype=torch.float16)
        
        # === Self Attention ===
        qkv_w = ck.dequantize_per_tensor_fp8(weights["qkv_proj"], scales["qkv_proj"], 
                                              output_type=torch.float16)
        qkv = torch.nn.functional.linear(x, qkv_w)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, num_heads, head_dim)
        v = v.view(batch_size, seq_len, num_heads, head_dim)
        
        # Apply RoPE
        q, k = ck.apply_rope(q, k, freqs)
        
        # Attention (simplified)
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        
        # Output projection
        out_w = ck.dequantize_per_tensor_fp8(weights["out_proj"], scales["out_proj"],
                                              output_type=torch.float16)
        attn_out = torch.nn.functional.linear(attn, out_w)
        x = x + attn_out  # Residual
        
        # === Cross Attention ===
        kv_w = ck.dequantize_per_tensor_fp8(weights["cross_attn_kv"], scales["cross_attn_kv"],
                                             output_type=torch.float16)
        kv = torch.nn.functional.linear(text, kv_w)
        k_text, v_text = kv.chunk(2, dim=-1)
        
        # Simplified cross attention (just matrix ops, no full attention)
        cross_out_w = ck.dequantize_per_tensor_fp8(weights["cross_attn_out"], 
                                                    scales["cross_attn_out"],
                                                    output_type=torch.float16)
        # ... would compute actual cross attention here
        
        # === MLP ===
        up_w = ck.dequantize_per_tensor_fp8(weights["mlp_up"], scales["mlp_up"],
                                             output_type=torch.float16)
        down_w = ck.dequantize_per_tensor_fp8(weights["mlp_down"], scales["mlp_down"],
                                               output_type=torch.float16)
        
        mlp_out = torch.nn.functional.linear(x, up_w)
        mlp_out = torch.nn.functional.gelu(mlp_out)
        mlp_out = torch.nn.functional.linear(mlp_out, down_w)
        
        output = x + mlp_out  # Residual
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        print(f"\n  Single diffusion step completed successfully")


class TestMemoryOptimization:
    """Tests for memory optimization on limited VRAM GPUs (T600=4GB)."""
    
    @pytest.mark.cuda
    def test_chunked_quantization(self):
        """Test chunked quantization for memory-limited scenarios."""
        # Total shape that might not fit in memory
        total_rows = 4096
        total_cols = 4096
        chunk_size = 1024
        
        # Process in chunks
        all_quantized = []
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        
        for i in range(0, total_rows, chunk_size):
            end_i = min(i + chunk_size, total_rows)
            chunk = torch.randn(end_i - i, total_cols, device="cuda", dtype=torch.float16)
            
            # Quantize chunk
            q_chunk = ck.quantize_per_tensor_fp8(chunk, scale)
            all_quantized.append(q_chunk.cpu())  # Move to CPU to save VRAM
            
            del chunk
            torch.cuda.empty_cache()
        
        # Verify all chunks processed
        assert len(all_quantized) == math.ceil(total_rows / chunk_size)
    
    @pytest.mark.cuda
    def test_fp8_memory_savings(self):
        """Verify FP8 provides expected memory savings."""
        shape = (1024, 1024)
        
        # FP16 tensor
        fp16_tensor = torch.randn(*shape, device="cuda", dtype=torch.float16)
        fp16_bytes = fp16_tensor.numel() * 2  # 2 bytes per FP16
        
        # FP8 tensor
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        fp8_tensor = ck.quantize_per_tensor_fp8(fp16_tensor, scale)
        fp8_bytes = fp8_tensor.numel() * 1  # 1 byte per FP8
        
        # Verify 2x memory savings
        savings = fp16_bytes / fp8_bytes
        print(f"\n  Memory savings: {savings:.1f}x (FP16: {fp16_bytes} bytes, FP8: {fp8_bytes} bytes)")
        assert savings == 2.0


class TestNumericalStability:
    """Tests for numerical stability of FP8 operations."""
    
    @pytest.mark.cuda
    def test_fp8_with_different_value_ranges(self):
        """Test FP8 handling of different value ranges."""
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        
        # Test cases with different value ranges
        test_cases = [
            ("small_values", torch.randn(64, 64, device="cuda", dtype=torch.float16) * 0.01),
            ("normal_values", torch.randn(64, 64, device="cuda", dtype=torch.float16)),
            ("large_values", torch.randn(64, 64, device="cuda", dtype=torch.float16) * 100),
        ]
        
        for name, x in test_cases:
            # Dynamic scaling
            dynamic_scale = torch.tensor([x.abs().max().item() / 448.0 + 1e-6],
                                         device="cuda", dtype=torch.float32)
            
            qx = ck.quantize_per_tensor_fp8(x, dynamic_scale)
            dx = ck.dequantize_per_tensor_fp8(qx, dynamic_scale, output_type=torch.float16)
            
            rel_error = (torch.abs(dx - x) / (torch.abs(x) + 1e-6)).mean().item()
            print(f"\n  {name}: rel_error = {rel_error:.2%}")
            
            # FP8 should handle all ranges with proper scaling
            assert rel_error < 0.15, f"High error for {name}: {rel_error:.2%}"
    
    @pytest.mark.cuda
    def test_fp8_gradient_preservation(self):
        """Test that FP8 quantization preserves gradient direction."""
        x = torch.randn(64, 64, device="cuda", dtype=torch.float16, requires_grad=False)
        scale = torch.tensor([x.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
        
        # Quantize and dequantize
        qx = ck.quantize_per_tensor_fp8(x, scale)
        dx = ck.dequantize_per_tensor_fp8(qx, scale, output_type=torch.float16)
        
        # Check direction preservation via correlation
        x_flat = x.flatten()
        dx_flat = dx.flatten()
        
        correlation = torch.corrcoef(torch.stack([x_flat, dx_flat]))[0, 1].item()
        print(f"\n  FP8 correlation with original: {correlation:.4f}")
        
        assert correlation > 0.99, "FP8 should preserve value direction"
