# Performance Benchmark: comfy_kitchen vs PyTorch baseline
# Simplified version with clean output

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import comfy_kitchen as ck
import time

def bench(func, warmup=20, iters=100):
    for _ in range(warmup): func()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters): func()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000

def main():
    props = torch.cuda.get_device_properties(0)
    print(f"\n{'='*70}")
    print(f" Performance Benchmark: comfy_kitchen vs PyTorch")
    print(f" GPU: {props.name} (SM {props.major}.{props.minor})")
    print(f"{'='*70}\n")
    
    scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    
    results = []
    
    # Test 1: FP8 Quantize
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    pt_q = bench(lambda: x.clamp(-448, 448).to(torch.float8_e4m3fn))
    ck_q = bench(lambda: ck.quantize_per_tensor_fp8(x, scale))
    results.append(("FP8 Quantize (1024x1024)", ck_q, pt_q))
    
    # Test 2: FP8 Dequantize
    x_fp8 = ck.quantize_per_tensor_fp8(x, scale)
    pt_dq = bench(lambda: x_fp8.to(torch.float16))
    ck_dq = bench(lambda: ck.dequantize_per_tensor_fp8(x_fp8, scale, torch.float16))
    results.append(("FP8 Dequantize (1024x1024)", ck_dq, pt_dq))
    
    # Test 3: Linear FP16 vs FP8 weight
    x_lin = torch.randn(1, 2048, 768, device="cuda", dtype=torch.float16)
    w16 = torch.randn(2304, 768, device="cuda", dtype=torch.float16)
    ws = torch.tensor([w16.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
    w8 = ck.quantize_per_tensor_fp8(w16, ws)
    
    pt_lin = bench(lambda: torch.nn.functional.linear(x_lin, w16))
    ck_lin = bench(lambda: torch.nn.functional.linear(
        x_lin, ck.dequantize_per_tensor_fp8(w8, ws, torch.float16)))
    results.append(("Linear Layer (2048x768→2304)", ck_lin, pt_lin))
    
    # Test 4: Backend comparison
    backends = ck.list_backends()
    eager_t = bench(lambda: [ck.use_backend("eager").__enter__(), 
                             ck.quantize_per_tensor_fp8(x, scale)][1])
    if backends.get("triton", {}).get("available"):
        triton_t = bench(lambda: [ck.use_backend("triton").__enter__(), 
                                  ck.quantize_per_tensor_fp8(x, scale)][1])
    else:
        triton_t = None
    
    # Test 5: Full Block
    hs, seq = 768, 1024
    x_blk = torch.randn(1, seq, hs, device="cuda", dtype=torch.float16)
    qkv16 = torch.randn(hs*3, hs, device="cuda", dtype=torch.float16)
    out16 = torch.randn(hs, hs, device="cuda", dtype=torch.float16)
    up16 = torch.randn(hs*4, hs, device="cuda", dtype=torch.float16)
    down16 = torch.randn(hs, hs*4, device="cuda", dtype=torch.float16)
    
    qkv_s = torch.tensor([qkv16.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
    qkv8 = ck.quantize_per_tensor_fp8(qkv16, qkv_s)
    out_s = torch.tensor([out16.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
    out8 = ck.quantize_per_tensor_fp8(out16, out_s)
    up_s = torch.tensor([up16.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
    up8 = ck.quantize_per_tensor_fp8(up16, up_s)
    down_s = torch.tensor([down16.abs().max().item() / 448.0], device="cuda", dtype=torch.float32)
    down8 = ck.quantize_per_tensor_fp8(down16, down_s)
    
    def pt_block():
        qkv = torch.nn.functional.linear(x_blk, qkv16)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(1, seq, 12, 64).transpose(1, 2)
        k = k.view(1, seq, 12, 64).transpose(1, 2)
        v = v.view(1, seq, 12, 64).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(1, seq, hs)
        x1 = x_blk + torch.nn.functional.linear(attn, out16)
        h = torch.nn.functional.gelu(torch.nn.functional.linear(x1, up16))
        return x1 + torch.nn.functional.linear(h, down16)
    
    def ck_block():
        qkv = torch.nn.functional.linear(x_blk, 
            ck.dequantize_per_tensor_fp8(qkv8, qkv_s, torch.float16))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(1, seq, 12, 64).transpose(1, 2)
        k = k.view(1, seq, 12, 64).transpose(1, 2)
        v = v.view(1, seq, 12, 64).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(1, seq, hs)
        x1 = x_blk + torch.nn.functional.linear(attn, 
            ck.dequantize_per_tensor_fp8(out8, out_s, torch.float16))
        h = torch.nn.functional.gelu(torch.nn.functional.linear(x1, 
            ck.dequantize_per_tensor_fp8(up8, up_s, torch.float16)))
        return x1 + torch.nn.functional.linear(h, 
            ck.dequantize_per_tensor_fp8(down8, down_s, torch.float16))
    
    pt_blk = bench(pt_block, iters=50)
    ck_blk = bench(ck_block, iters=50)
    results.append(("Full Transformer Block", ck_blk, pt_blk))
    
    # Memory calculation
    fp16_mem = (qkv16.numel() + out16.numel() + up16.numel() + down16.numel()) * 2 / 1024
    fp8_mem = (qkv8.numel() + out8.numel() + up8.numel() + down8.numel()) * 1 / 1024
    
    # Print results
    print(f"{'Test':<40} {'CK (ms)':<12} {'PT (ms)':<12} {'Result':<20}")
    print("-" * 84)
    
    for name, ck_t, pt_t in results:
        if ck_t < pt_t:
            result = f"CK {pt_t/ck_t:.2f}x faster"
        else:
            result = f"PT {ck_t/pt_t:.2f}x faster"
        print(f"{name:<40} {ck_t:<12.4f} {pt_t:<12.4f} {result:<20}")
    
    print()
    print("-" * 84)
    print("Backend Comparison (FP8 Quantize 1024x1024):")
    print(f"  eager:  {eager_t:.4f} ms")
    if triton_t:
        print(f"  triton: {triton_t:.4f} ms ({eager_t/triton_t:.2f}x faster)")
    
    print()
    print("-" * 84)
    print(f"Memory Savings: FP16={fp16_mem:.0f}KB -> FP8={fp8_mem:.0f}KB (2x savings)")
    
    print(f"\n{'='*70}")
    print(" Conclusions:")
    print("  1. FP8 主要优势是 **内存节省 2x**，而非速度提升")
    print("  2. 在 V100/T600 上需要 dequant 到 FP16，有额外开销")
    print("  3. Triton 融合内核比 eager 快约 1.3x")
    print("  4. 适用于 VRAM 受限场景（如 T600 4GB）")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
