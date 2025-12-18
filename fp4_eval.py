import time, torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def bench_nvfp4(M=1024, K=4096, N=4096, iters=30, warmup=10):
    lin = te.Linear(K, N, bias=False).cuda().bfloat16()
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, fused=True)

    recipe = NVFP4BlockScaling()
    # warmup (eager)
    with te.autocast(enabled=True, recipe=recipe):
        for _ in range(warmup):
            y = lin(x).sum()
            y.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # timed (eager)
    t0 = time.time()
    with te.autocast(enabled=True, recipe=recipe):
        for _ in range(iters):
            y = lin(x).sum()
            y.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"NVFP4: {(time.time()-t0)*1000/iters:.2f} ms/iter")

if __name__ == "__main__":
    print("GPU:", torch.cuda.get_device_name(), "SM:", torch.cuda.get_device_capability())
    print("NVFP4 available:", te.is_nvfp4_available())
    bench_nvfp4()
