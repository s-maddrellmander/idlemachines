#!/usr/bin/env python3
"""
DGX Spark FP8 Verification Suite
=================================

Run each test in order to verify your DGX Spark setup for FP8 training.

Usage:
    python dgx_spark_tests.py          # Run all tests
    python dgx_spark_tests.py --test 1  # Run specific test
"""

import sys
import time

def test_1_cuda_basic():
    """Test 1: Basic CUDA availability"""
    print("\n" + "="*60)
    print("TEST 1: Basic CUDA Availability")
    print("="*60)
    
    import torch
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available!")
        print("Check: nvidia-smi, docker GPU flags, driver version")
        return False
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    print(f"Compute capability: {cap[0]}.{cap[1]}")
    
    # Check for sm_121 (GB10)
    if cap == (12, 1):
        print("✓ GB10 (sm_121) detected")
    elif cap[0] >= 9:
        print(f"✓ Hopper/Ada/Blackwell detected (sm_{cap[0]}{cap[1]})")
    else:
        print(f"⚠ Older GPU (sm_{cap[0]}{cap[1]}) - FP8 may not be available")
    
    # Basic tensor test
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x)
    print(f"Basic matmul: ✓")
    
    # BF16 test
    x_bf16 = x.to(torch.bfloat16)
    y_bf16 = torch.matmul(x_bf16, x_bf16)
    print(f"BF16 matmul: ✓")
    
    print("\n✓ TEST 1 PASSED")
    return True


def test_2_fp8_dtype():
    """Test 2: FP8 data types availability"""
    print("\n" + "="*60)
    print("TEST 2: FP8 Data Types")
    print("="*60)
    
    import torch
    
    # Check for FP8 dtypes
    dtypes_to_check = [
        ('torch.float8_e4m3fn', 'float8_e4m3fn'),
        ('torch.float8_e5m2', 'float8_e5m2'),
    ]
    
    all_available = True
    for name, attr in dtypes_to_check:
        if hasattr(torch, attr):
            print(f"{name}: ✓ Available")
            try:
                x = torch.randn(100, device='cuda').to(getattr(torch, attr))
                print(f"  Can create tensor: ✓")
            except Exception as e:
                print(f"  Can create tensor: ❌ ({e})")
                all_available = False
        else:
            print(f"{name}: ❌ Not available")
            all_available = False
    
    if all_available:
        print("\n✓ TEST 2 PASSED")
    else:
        print("\n⚠ TEST 2 PARTIAL - Some FP8 dtypes unavailable")
    
    return all_available


def test_3_scaled_mm():
    """Test 3: torch._scaled_mm (FP8 GEMM kernel)"""
    print("\n" + "="*60)
    print("TEST 3: FP8 GEMM (torch._scaled_mm)")
    print("="*60)
    
    import torch
    
    if not hasattr(torch, '_scaled_mm'):
        print("❌ torch._scaled_mm not available")
        print("This is the core FP8 matmul operation.")
        print("Your PyTorch build may not support FP8 tensor cores.")
        return False
    
    print("torch._scaled_mm: ✓ Available")
    
    # Test actual execution
    try:
        M, K, N = 1024, 1024, 1024
        
        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        a_fp8 = a.to(torch.float8_e4m3fn)
        b_fp8 = b.to(torch.float8_e4m3fn)
        
        scale_a = torch.tensor(1.0, device='cuda')
        scale_b = torch.tensor(1.0, device='cuda')
        
        result = torch._scaled_mm(
            a_fp8,
            b_fp8.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16
        )
        
        print(f"Execution test: ✓")
        print(f"  Input: ({M}, {K}) x ({K}, {N})")
        print(f"  Output: {result.shape}, dtype={result.dtype}")
        
        print("\n✓ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return False


def test_4_torchao():
    """Test 4: torchao Float8 training"""
    print("\n" + "="*60)
    print("TEST 4: torchao Float8 Training")
    print("="*60)
    
    import torch
    import torch.nn as nn
    
    # Check imports
    try:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        print("torchao.float8 imports: ✓")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Install with: pip install torchao")
        return False
    
    try:
        from torchao.float8.float8_linear import Float8Linear
        print("Float8Linear import: ✓")
    except ImportError as e:
        print(f"❌ Float8Linear import failed: {e}")
        return False
    
    # Create and convert model
    model = nn.Sequential(
        nn.Linear(1024, 2048, bias=False),
        nn.ReLU(),
        nn.Linear(2048, 1024, bias=False),
    ).cuda().bfloat16()
    
    config = Float8LinearConfig.from_recipe_name("rowwise")
    convert_to_float8_training(model, config=config)
    
    fp8_count = sum(1 for m in model.modules() if isinstance(m, Float8Linear))
    print(f"Layers converted to Float8Linear: {fp8_count}")
    
    if fp8_count == 0:
        print("❌ No layers converted!")
        return False
    
    # Test forward/backward
    try:
        model_compiled = torch.compile(model)
        x = torch.randn(32, 1024, device='cuda', dtype=torch.bfloat16)
        y = model_compiled(x)
        y.sum().backward()
        print("Forward/backward pass: ✓")
    except Exception as e:
        print(f"❌ Forward/backward failed: {e}")
        return False
    
    print("\n✓ TEST 4 PASSED")
    return True


def test_5_benchmark():
    """Test 5: Performance benchmark"""
    print("\n" + "="*60)
    print("TEST 5: Performance Benchmark")
    print("="*60)
    
    import torch
    import torch.nn as nn
    import copy
    
    try:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    except ImportError:
        print("❌ torchao not available, skipping benchmark")
        return False
    
    M, K, N = 4096, 4096, 4096
    num_warmup = 10
    num_iters = 50
    
    print(f"Matrix: ({M}, {K}) x ({K}, {N})")
    print(f"Iterations: {num_iters}")
    print("")
    
    # Create models
    bf16_model = nn.Linear(K, N, bias=False).cuda().bfloat16()
    fp8_model = copy.deepcopy(bf16_model)
    
    config = Float8LinearConfig.from_recipe_name("rowwise")
    convert_to_float8_training(fp8_model, config=config)
    
    bf16_compiled = torch.compile(bf16_model)
    fp8_compiled = torch.compile(fp8_model)
    
    x = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    
    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        bf16_compiled(x).sum().backward()
        bf16_compiled.zero_grad()
        fp8_compiled(x).sum().backward()
        fp8_compiled.zero_grad()
    torch.cuda.synchronize()
    
    # Benchmark BF16
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        bf16_compiled(x).sum().backward()
        bf16_compiled.zero_grad()
    torch.cuda.synchronize()
    bf16_time = (time.time() - start) / num_iters * 1000
    
    # Benchmark FP8
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        fp8_compiled(x).sum().backward()
        fp8_compiled.zero_grad()
    torch.cuda.synchronize()
    fp8_time = (time.time() - start) / num_iters * 1000
    
    print(f"\nResults:")
    print(f"  BF16: {bf16_time:.2f} ms/iter")
    print(f"  FP8:  {fp8_time:.2f} ms/iter")
    
    speedup = bf16_time / fp8_time
    if speedup > 1.05:
        print(f"\n✓ FP8 is {speedup:.2f}x FASTER!")
        print("  FP8 tensor cores appear to be working.")
        return True
    elif speedup > 0.95:
        print(f"\n⚠ FP8 is about the same speed ({speedup:.2f}x)")
        print("  May be using emulation or overhead is high.")
        return True
    else:
        print(f"\n⚠ FP8 is {speedup:.2f}x (slower than BF16)")
        print("  FP8 may be falling back to emulation.")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DGX Spark FP8 Tests')
    parser.add_argument('--test', type=int, help='Run specific test (1-5)')
    args = parser.parse_args()
    
    tests = [
        (1, test_1_cuda_basic),
        (2, test_2_fp8_dtype),
        (3, test_3_scaled_mm),
        (4, test_4_torchao),
        (5, test_5_benchmark),
    ]
    
    print("="*60)
    print("DGX SPARK FP8 VERIFICATION SUITE")
    print("="*60)
    
    if args.test:
        # Run specific test
        for num, func in tests:
            if num == args.test:
                func()
                break
    else:
        # Run all tests
        results = []
        for num, func in tests:
            try:
                passed = func()
                results.append((num, passed))
            except Exception as e:
                print(f"\n❌ TEST {num} CRASHED: {e}")
                results.append((num, False))
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for num, passed in results:
            status = "✓ PASSED" if passed else "❌ FAILED"
            print(f"  Test {num}: {status}")
        
        all_passed = all(p for _, p in results)
        print("")
        if all_passed:
            print("✓ All tests passed! Ready for FP8 training.")
        else:
            print("⚠ Some tests failed. Review output above.")


if __name__ == "__main__":
    main()
