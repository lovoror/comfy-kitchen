# Standalone test runner for video generation tests
# (Bypasses pytest version compatibility issues)

import sys
import os
# Add parent directory to path for package import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import comfy_kitchen as ck
import time
import sys

def print_header(name):
    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")

def print_test(name, passed=True, info=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name} {info}")

def main():
    print_header("Video Generation Tests for V100/T600")
    
    # GPU info
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} (SM {props.major}.{props.minor})")
    print(f"Legacy GPU: {(props.major, props.minor) < (8, 9)}")
    
    # Config
    hidden_size = 768
    num_heads = 12
    head_dim = 64
    mlp_ratio = 4
    
    all_passed = True
    
    # ===== Test 1: Attention QKV Projection =====
    print_header("Test 1: Attention QKV Projection")
    try:
        seq_len = 2048
        x = torch.randn(1, seq_len, hidden_size, device="cuda", dtype=torch.float16)
        qkv_w = torch.randn(hidden_size * 3, hidden_size, device="cuda", dtype=torch.float16)
        scale = torch.tensor([qkv_w.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
        qkv_fp8 = ck.quantize_per_tensor_fp8(qkv_w, scale)
        qkv_dq = ck.dequantize_per_tensor_fp8(qkv_fp8, scale, output_type=torch.float16)
        qkv = torch.nn.functional.linear(x, qkv_dq)
        q, k, v = qkv.chunk(3, dim=-1)
        assert q.shape == (1, seq_len, hidden_size)
        print_test("QKV projection with FP8 weights", True, f"Shape: {q.shape}")
    except Exception as e:
        print_test("QKV projection with FP8 weights", False, str(e))
        all_passed = False
    
    # ===== Test 2: RoPE =====
    print_header("Test 2: RoPE for Video Attention")
    try:
        batch, seq, heads, dim = 1, 1024, 12, 64
        xq = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.float16)
        xk = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.float16)
        freqs = torch.randn(1, seq, 1, 1, dim//2, 2, device="cuda", dtype=torch.float16)
        xq_out, xk_out = ck.apply_rope(xq, xk, freqs)
        assert xq_out.shape == xq.shape
        assert not torch.allclose(xq_out, xq)
        print_test("RoPE applied to Q and K", True, f"Shape: {xq_out.shape}")
    except Exception as e:
        print_test("RoPE applied to Q and K", False, str(e))
        all_passed = False
    
    # ===== Test 3: MLP Block =====
    print_header("Test 3: MLP Block with FP8 Weights")
    try:
        seq_len = 2048
        mlp_hidden = hidden_size * mlp_ratio
        x = torch.randn(1, seq_len, hidden_size, device="cuda", dtype=torch.float16)
        up_w = torch.randn(mlp_hidden, hidden_size, device="cuda", dtype=torch.float16)
        down_w = torch.randn(hidden_size, mlp_hidden, device="cuda", dtype=torch.float16)
        up_s = torch.tensor([up_w.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
        down_s = torch.tensor([down_w.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
        up_fp8 = ck.quantize_per_tensor_fp8(up_w, up_s)
        down_fp8 = ck.quantize_per_tensor_fp8(down_w, down_s)
        up_dq = ck.dequantize_per_tensor_fp8(up_fp8, up_s, output_type=torch.float16)
        down_dq = ck.dequantize_per_tensor_fp8(down_fp8, down_s, output_type=torch.float16)
        h = torch.nn.functional.gelu(torch.nn.functional.linear(x, up_dq))
        out = torch.nn.functional.linear(h, down_dq)
        assert out.shape == x.shape
        print_test("MLP forward pass", True, f"Shape: {out.shape}")
    except Exception as e:
        print_test("MLP forward pass", False, str(e))
        all_passed = False
    
    # ===== Test 4: VAE Latent =====
    print_header("Test 4: VAE Latent FP8 Quantization")
    try:
        latents = torch.randn(1, 4, 16, 60, 104, device="cuda", dtype=torch.float16)
        lat_s = torch.tensor([latents.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
        lat_fp8 = ck.quantize_per_tensor_fp8(latents, lat_s)
        lat_dq = ck.dequantize_per_tensor_fp8(lat_fp8, lat_s, output_type=torch.float16)
        mse = torch.nn.functional.mse_loss(lat_dq, latents).item()
        assert mse < 1.0
        print_test("VAE Latent FP8", True, f"MSE: {mse:.6f}")
    except Exception as e:
        print_test("VAE Latent FP8", False, str(e))
        all_passed = False
    
    # ===== Test 5: T5 Encoder Output =====
    print_header("Test 5: T5 Encoder Output Quantization")
    try:
        text_emb = torch.randn(1, 512, 4096, device="cuda", dtype=torch.float16)
        t5_s = torch.tensor([text_emb.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
        t5_fp8 = ck.quantize_per_tensor_fp8(text_emb, t5_s)
        t5_dq = ck.dequantize_per_tensor_fp8(t5_fp8, t5_s, output_type=torch.float16)
        rel_error = (torch.abs(t5_dq - text_emb) / (torch.abs(text_emb) + 1e-6)).mean().item()
        assert rel_error < 0.1
        print_test("T5 embeddings FP8", True, f"Rel error: {rel_error:.2%}")
    except Exception as e:
        print_test("T5 embeddings FP8", False, str(e))
        all_passed = False
    
    # ===== Test 6: Full Diffusion Step =====
    print_header("Test 6: Single Diffusion Step Simulation")
    try:
        hs = 512
        nh = 8
        hd = 64
        seq = 512
        
        # Quantize weights
        qkv_w = torch.randn(hs*3, hs, device="cuda", dtype=torch.float16)
        out_w = torch.randn(hs, hs, device="cuda", dtype=torch.float16)
        up_w = torch.randn(hs*4, hs, device="cuda", dtype=torch.float16)
        down_w = torch.randn(hs, hs*4, device="cuda", dtype=torch.float16)
        
        scales = {}
        for name, w in [("qkv", qkv_w), ("out", out_w), ("up", up_w), ("down", down_w)]:
            scales[name] = torch.tensor([w.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
        
        qkv_fp8 = ck.quantize_per_tensor_fp8(qkv_w, scales["qkv"])
        out_fp8 = ck.quantize_per_tensor_fp8(out_w, scales["out"])
        up_fp8 = ck.quantize_per_tensor_fp8(up_w, scales["up"])
        down_fp8 = ck.quantize_per_tensor_fp8(down_w, scales["down"])
        
        # Forward pass
        x = torch.randn(1, seq, hs, device="cuda", dtype=torch.float16)
        freqs = torch.randn(1, seq, 1, 1, hd//2, 2, device="cuda", dtype=torch.float16)
        
        # Self-attn
        qkv_dq = ck.dequantize_per_tensor_fp8(qkv_fp8, scales["qkv"], output_type=torch.float16)
        qkv = torch.nn.functional.linear(x, qkv_dq)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(1, seq, nh, hd)
        k = k.view(1, seq, nh, hd)
        v = v.view(1, seq, nh, hd)
        q, k = ck.apply_rope(q, k, freqs)
        
        # Attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(1, seq, hs)
        
        out_dq = ck.dequantize_per_tensor_fp8(out_fp8, scales["out"], output_type=torch.float16)
        x = x + torch.nn.functional.linear(attn, out_dq)
        
        # MLP
        up_dq = ck.dequantize_per_tensor_fp8(up_fp8, scales["up"], output_type=torch.float16)
        down_dq = ck.dequantize_per_tensor_fp8(down_fp8, scales["down"], output_type=torch.float16)
        h = torch.nn.functional.gelu(torch.nn.functional.linear(x, up_dq))
        output = x + torch.nn.functional.linear(h, down_dq)
        
        assert output.shape == (1, seq, hs)
        print_test("Full diffusion step", True, f"Output: {output.shape}")
    except Exception as e:
        print_test("Full diffusion step", False, str(e))
        all_passed = False
    
    # ===== Test 7: Performance Benchmark =====
    print_header("Test 7: Performance Benchmark")
    try:
        x = torch.randn(1, 2048, 768, device="cuda", dtype=torch.float16)
        w = torch.randn(768*3, 768, device="cuda", dtype=torch.float16)
        s = torch.tensor([w.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
        w_fp8 = ck.quantize_per_tensor_fp8(w, s)
        
        # Warmup
        for _ in range(10):
            w_dq = ck.dequantize_per_tensor_fp8(w_fp8, s, output_type=torch.float16)
            _ = torch.nn.functional.linear(x, w_dq)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            w_dq = ck.dequantize_per_tensor_fp8(w_fp8, s, output_type=torch.float16)
            _ = torch.nn.functional.linear(x, w_dq)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 50 * 1000
        
        print_test("FP8 weight dequant + linear", True, f"{elapsed:.2f} ms")
    except Exception as e:
        print_test("Performance benchmark", False, str(e))
        all_passed = False
    
    # ===== Summary =====
    print_header("Summary")
    if all_passed:
        print("  ✓ ALL TESTS PASSED!")
    else:
        print("  ✗ SOME TESTS FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
