#!/usr/bin/env python3
"""
FP8 Deep Diagnostic
===================

This script digs deeper to understand what's actually happening
with FP8 on the DGX Spark.
"""

import torch
import torch.nn as nn
import time

print("="*60)
print("FP8 DEEP DIAGNOSTIC")
print("="*60)

# =============================================================================
# 1. Check what _scaled_mm is actually doing
# =============================================================================
print("\n--- 1. torch._scaled_mm Implementation Check ---")

# Check if there's a CUDA implementation
print(f"torch._scaled_mm exists: {hasattr(torch, '_scaled_mm')}")

# Try to see what backend is being used
try:
    # This will show if it's dispatching to cuBLAS FP8 or something else
    import torch._C
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuBLAS available: {torch.backends.cuda.is_built()}")
except Exception as e:
    print(f"Backend check error: {e}")

# =============================================================================
# 2. Memory bandwidth vs compute bound analysis
# =============================================================================
print("\n--- 2. Memory vs Compute Analysis ---")

# GB10 specs (approximate):
# - Memory bandwidth: ~273 GB/s (unified memory, shared with CPU)
# - FP8 tensor core TFLOPS: ~209 TFLOPS (inference), lower for training
# - For comparison, H100: ~3.35 TB/s bandwidth, ~1979 TFLOPS FP8

# Test different matrix sizes to see if we're compute or memory bound
sizes = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]

print(f"\nMatrix size scaling test (FP8 vs BF16):")
print(f"{'M,K,N':<20} {'BF16 (ms)':<12} {'FP8 (ms)':<12} {'Speedup':<10}")
print("-"*54)

for M, K, N in sizes:
    try:
        # BF16 baseline
        a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        b_bf16 = torch.randn(K, N, device='cuda', dtype=torch.bfloat16)
        
        # Warmup
        for _ in range(5):
            _ = torch.matmul(a_bf16, b_bf16)
        torch.cuda.synchronize()
        
        # Benchmark BF16
        start = time.time()
        for _ in range(20):
            _ = torch.matmul(a_bf16, b_bf16)
        torch.cuda.synchronize()
        bf16_time = (time.time() - start) / 20 * 1000
        
        # FP8
        a_fp8 = a_bf16.to(torch.float8_e4m3fn)
        b_fp8 = b_bf16.to(torch.float8_e4m3fn)
        scale = torch.tensor(1.0, device='cuda')
        
        # Warmup
        for _ in range(5):
            _ = torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        
        # Benchmark FP8
        start = time.time()
        for _ in range(20):
            _ = torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        fp8_time = (time.time() - start) / 20 * 1000
        
        speedup = bf16_time / fp8_time
        print(f"{M},{K},{N:<14} {bf16_time:<12.2f} {fp8_time:<12.2f} {speedup:<10.2f}x")
        
        del a_bf16, b_bf16, a_fp8, b_fp8
        torch.cuda.empty_cache()
        
    except torch.cuda.OutOfMemoryError:
        print(f"{M},{K},{N:<14} OOM")
        break

# =============================================================================
# 3. Check if FP8 storage is actually smaller
# =============================================================================
print("\n--- 3. Memory Usage Check ---")

M, K = 4096, 4096

# BF16 tensor
t_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
bf16_bytes = t_bf16.element_size() * t_bf16.numel()

# FP8 tensor  
t_fp8 = t_bf16.to(torch.float8_e4m3fn)
fp8_bytes = t_fp8.element_size() * t_fp8.numel()

print(f"Tensor shape: ({M}, {K})")
print(f"BF16 size: {bf16_bytes / 1024 / 1024:.2f} MB ({t_bf16.element_size()} bytes/element)")
print(f"FP8 size:  {fp8_bytes / 1024 / 1024:.2f} MB ({t_fp8.element_size()} bytes/element)")
print(f"Memory reduction: {bf16_bytes / fp8_bytes:.1f}x")

if fp8_bytes == bf16_bytes // 2:
    print("✓ FP8 storage is working correctly (half the size of BF16)")
else:
    print("⚠ FP8 storage size unexpected")

# =============================================================================
# 4. Profile a single FP8 matmul
# =============================================================================
print("\n--- 4. Profiling FP8 Matmul ---")

try:
    from torch.profiler import profile, ProfilerActivity
    
    M, K, N = 4096, 4096, 4096
    a = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
    b = torch.randn(K, N, device='cuda').to(torch.float8_e4m3fn)
    scale = torch.tensor(1.0, device='cuda')
    
    # Warmup
    for _ in range(5):
        _ = torch._scaled_mm(a, b.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    
    # Profile
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(10):
            _ = torch._scaled_mm(a, b.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
    
    print("\nCUDA Kernel Events (top 5):")
    events = prof.key_averages()
    events = sorted(events, key=lambda x: x.cuda_time_total, reverse=True)[:5]
    for e in events:
        print(f"  {e.key[:50]:<50} {e.cuda_time_total/1000:.2f}ms total")
    
    # Look for FP8-specific kernel names
    all_keys = [e.key for e in prof.key_averages()]
    fp8_kernels = [k for k in all_keys if 'fp8' in k.lower() or 'e4m3' in k.lower() or 'scaled' in k.lower()]
    
    print(f"\nFP8-related kernels found: {len(fp8_kernels)}")
    for k in fp8_kernels[:5]:
        print(f"  - {k}")
        
except Exception as e:
    print(f"Profiling error: {e}")

# =============================================================================
# 5. Check theoretical vs actual throughput
# =============================================================================
print("\n--- 5. Throughput Analysis ---")

M, K, N = 4096, 4096, 4096
flops_per_matmul = 2 * M * K * N  # multiply-add = 2 ops

# Run FP8 benchmark
a = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
b = torch.randn(K, N, device='cuda').to(torch.float8_e4m3fn)
scale = torch.tensor(1.0, device='cuda')

# Warmup
for _ in range(10):
    _ = torch._scaled_mm(a, b.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
torch.cuda.synchronize()

# Benchmark
num_iters = 100
start = time.time()
for _ in range(num_iters):
    _ = torch._scaled_mm(a, b.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
torch.cuda.synchronize()
elapsed = time.time() - start

tflops = (flops_per_matmul * num_iters) / elapsed / 1e12
print(f"Matrix: ({M}, {K}) x ({K}, {N})")
print(f"FLOPs per matmul: {flops_per_matmul / 1e9:.1f}G")
print(f"Achieved: {tflops:.1f} TFLOPS")

# GB10 theoretical peaks (approximate)
# These are estimates - actual specs may differ
print(f"\nGB10 theoretical peaks (estimated):")
print(f"  FP8 inference: ~209 TFLOPS")
print(f"  BF16: ~104 TFLOPS")
print(f"  Achieved utilization: ~{tflops/209*100:.1f}% of FP8 peak")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("DIAGNOSIS SUMMARY")
print("="*60)

print("""
Key findings:
1. FP8 dtypes and _scaled_mm are available and execute
2. FP8 storage is 1 byte (vs 2 bytes for BF16) - memory savings work
3. Check the speedup scaling with matrix size above
4. Check if FP8-specific CUDA kernels appear in the profile

If FP8 speedup is ~1x across all sizes:
- GB10 may be memory-bandwidth limited (273 GB/s vs H100's 3.35 TB/s)
- FP8 compute savings are masked by memory bottleneck
- This is still valid FP8 training - you get memory savings even if not speed

If larger matrices show better FP8 speedup:
- FP8 tensor cores ARE working
- Small matrices are overhead-dominated

For your CV signal: The important thing is demonstrating you understand
these tradeoffs, not that GB10 achieves H100-like speedups.
""")