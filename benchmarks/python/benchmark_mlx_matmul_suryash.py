from time_utils import measure_runtime
import matplotlib.pyplot as plt
import mlx.core as mx
import json
import numpy as np
import pandas as pd
import datetime


def benchmark_complex_mlx_matmul(D, use_gpu=False):
    if use_gpu:
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)
    # randomly generate complex matrices
    def create_complex_matrix(shape):
        return mx.random.uniform(shape=shape) + 1j * mx.random.uniform(shape=shape)
    
    a = create_complex_matrix((batch_size, T, D))
    b = create_complex_matrix((D, D))
    
    mx.eval(a, b)
    
    # operation to benchmark
    def matmul_op():
        return mx.eval(mx.matmul(a, b))
    
    # Measure runtime
    avg_time_ms = measure_runtime(matmul_op)  
    avg_time_sec = avg_time_ms / 1000.0

    total_flops = {'std': 8 * D * D * D,
                   'strassen' : 6 * D * D * D}
    
    # gflops = total_flops["strassen"] / (avg_time_sec * 1e9)
    
    return avg_time_ms

if __name__ == "__main__":
    save = True
    spreadsheet = True
    sizes = list(range(500, 6000, 500))
    times, gflops = [], []
    
    for D in sizes:
        print(f"Benchmarking complex {D}x{D} matrices...")
        # t, gf = benchmark_complex_mlx_matmul(D)
        t = benchmark_complex_mlx_matmul(D, use_gpu=True)

        # print(f"  Time: {t:.2f} ms, Performance: {gf:.2f} GFLOPs/s")
        print(f"  Time: {t:.2f} ms")

        times.append(t)
        # gflops.append(gf)
    
    if save == True:
        with open(f"mlx_pip_complex_matmul_benchmark_cpu+{datetime.datetime.now().microsecond}.json", "w") as f:
            json.dump({"sizes": sizes, "times": times, "gflops": gflops}, f)
    
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

    # Plot for GFLOPs
    # ax1.plot(sizes, gflops, 'o-', linewidth=2, color='blue')
    # ax1.set_xlabel("Matrix Size D (DxD)")
    # ax1.set_ylabel("GFLOPs/s", color='blue')
    # ax1.tick_params(axis='y', labelcolor='blue')
    # ax1.grid(True)

    # Create a second y-axis for the time plot
    ax2 = ax1.twinx()
    ax2.plot(sizes, times, 'o-', linewidth=2, color='red')
    ax2.set_ylabel("Execution Time (ms)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Set title
    plt.title("MLX Complex MatMul Performance and Execution Time")

        
    plt.tight_layout()
    plt.savefig(f"{datetime.datetime.now().microsecond}_mlx_complex_matmul_benchmark.png")
    plt.show()
