from time_utils import measure_runtime, time_fn
import matplotlib.pyplot as plt
import mlx.core as mx
import json
import numpy as np
# import pandas as pd
import datetime


def benchmark_complex_mlx_matmul(B, D, T, use_gpu=False):
    mx.set_default_device(mx.gpu if use_gpu else mx.cpu)

    def cplx(shape):
        return mx.array(
            mx.random.uniform(shape=shape) + 1j * mx.random.uniform(shape=shape),
            dtype=mx.complex64
        )

    # Keep original layout and square RHS
    A     = cplx((B, T, D))   # (batch,  T, D)
    Bmat  = cplx((D, D))      # (D, D)   — square as before

    # print("A dtype :", A.dtype)        # should say complex64
    # print("Bmat dtype :", Bmat.dtype)  # should also say complex64
    # print("A shape:", A.shape)         # sanity‑check layout
    # print("Bmat shape:", Bmat.shape)

    mx.eval(A, Bmat)          # warm‑up with correct tensors

    # time_fn must return the elapsed ms
    ms = time_fn(lambda: mx.matmul(A, Bmat),
                 msg=f"MatMul {B}×{T}×{D} @ {D}×{D}")
    return ms


if __name__ == "__main__":
    save = True
    spreadsheet = True
    plot = True

    BATCH_SIZE = 8
    D = 512
    T = 512
    sizes = list(range(500, 6000, 500))
    times = []
    
    for D in sizes:
        # print(f"Benchmarking complex A[{BATCH_SIZE}×{D}×{T}] and B[{T}×{D}] matrices...")
        print(f"Benchmarking complex A[{BATCH_SIZE}×{D}×{T}] and B[{T}×{D}] matrices...")

        # t, gf = benchmark_complex_mlx_matmul(D)
        t = benchmark_complex_mlx_matmul(BATCH_SIZE, D, T, use_gpu=True)
        print(f"  Time: {t:.2f} ms")

        times.append(t)
        # gflops.append(gf)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save == True:
        with open(f"mlx_pip_complex_matmul_benchmark_cpu+{datetime.datetime.now().microsecond}.json", "w") as f:
            json.dump({"sizes": sizes, "times": times}, f)
    
    # if spreadsheet == True:
    #     # Create a DataFrame with the benchmark results
    #     results_df = pd.DataFrame({
    #         "Sizes": sizes,
    #         "Times (ms)": times,
    #         "GFLOPs/s": gflops
    #     })
    #     # Save to Excel file
    #     results_df.to_excel("mlx_complex_matmul_benchmark.xlsx", index=False)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Performance plot
    fig, ax1 = plt.subplots()

    # Create a second y-axis for the time plot
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, "o-")
    plt.xlabel("Matrix dimension D")
    plt.ylabel("Execution time (ms)")
    plt.title(f"Complex matmul • A[{BATCH_SIZE}×D×{T}] · B[{T}×D]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"mlx_complex_matmul_{stamp}.png", dpi=150)
    plt.show()