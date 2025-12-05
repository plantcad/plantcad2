import time
import numpy as np
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import importlib.metadata

def load_model(model_path: str, device: str = "cuda"):
    """Load HuggingFace model in bfloat16"""
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model.to(torch.bfloat16)
    model.eval()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        tokenizer = None
        print(f"Warning: Could not load tokenizer for {model_path}")
    
    return model, tokenizer

def benchmark_model(model_path: str, model_name: str, 
                   seq_length: int = 8192, n_warmup: int = 5, 
                   n_benchmark: int = 10, batch_size: int = 1, device: str = "cuda") -> dict:
    """
    Benchmark DNA language model inference speed + peak GPU memory
    """
    print(f"Benchmarking {model_name} (batch_size={batch_size})...")
    
    # Load model
    model, tokenizer = load_model(model_path, device)
    
    # Generate random DNA sequences
    bases = ['A', 'T', 'G', 'C']
    sequences = [''.join(np.random.choice(bases, seq_length)) for _ in range(batch_size)]
    
    # Tokenize
    inputs = tokenizer(sequences, return_tensors="pt", truncation=True, 
                        max_length=seq_length, padding=True)['input_ids'].to(device)

    # Warmup
    print(f"  Warming up with {n_warmup} batches...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(inputs)

    # Clear cache + reset memory stat
    if "cuda" in device:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    print(f"  Running {n_benchmark} benchmark iterations...")
    times = []
    peak_mems_allocated = []
    peak_mems_reserved = []

    for i in range(n_benchmark):

        # Reset peak memory each iteration
        if "cuda" in device:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():
            output = model(inputs)

        # <<< THIS LINE IS CRUCIAL >>>
        if "cuda" in device:
            torch.cuda.synchronize()  # force kernels to complete

        end_time = time.perf_counter()
        inference_time = end_time - start_time
        times.append(inference_time)

        # Record peak memory correctly
        if "cuda" in device:
            peak_mem_alloc_bytes = torch.cuda.max_memory_allocated()
            peak_mem_reserved_bytes = torch.cuda.max_memory_reserved()
            peak_mem_alloc_gb = peak_mem_alloc_bytes / (1024**3)
            peak_mem_reserved_gb = peak_mem_reserved_bytes / (1024**3)
        else:
            peak_mem_alloc_gb = 0
            peak_mem_reserved_gb = 0

        peak_mems_allocated.append(peak_mem_alloc_gb)
        peak_mems_reserved.append(peak_mem_reserved_gb)

        print(f"Run {i+1}: peak allocated: {peak_mem_alloc_gb:.2f} GB | peak reserved: {peak_mem_reserved_gb:.2f} GB")

    # Stats
    times = np.array(times)
    tokens_per_second_list = [(seq_length * batch_size) / t for t in times]

    results = {
        'model_name': model_name,
        'batch_size': batch_size,
        'median_time': float(np.median(times)),
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'median_tokens_per_sec': float(np.median(tokens_per_second_list)),
        'mean_tokens_per_sec': float(np.mean(tokens_per_second_list)),
        'median_tokens_per_sec_per_sample': float(np.median(tokens_per_second_list) / batch_size),
        'median_peak_memory_allocated_gb': float(np.median(peak_mems_allocated)),
        'median_peak_memory_reserved_gb': float(np.median(peak_mems_reserved)),
        'all_peak_memory_allocated_gb': peak_mems_allocated,
        'all_peak_memory_reserved_gb': peak_mems_reserved,
        'all_times': times.tolist(),
        'all_tokens_per_sec': tokens_per_second_list,
        'status': 'OK'
    }

    # Cleanup
    del model
    if "cuda" in device:
        torch.cuda.empty_cache()

    return results


def compare_models(model_paths: List[tuple], seq_length: int = 8192, 
                  batch_sizes: List[int] = [1, 4, 8, 16], device: str = "cuda"):
    """
    Compare multiple models for speed + peak memory
    """
    table_data = []

    for model_path, model_name in model_paths:
        print(f"\n{'='*50}")
        print(f"Testing {model_name}")
        print(f"{'='*50}")
        
        for batch_size in batch_sizes:
            try:
                result = benchmark_model(model_path, model_name, seq_length,
                                         batch_size=batch_size, device=device)
                
                # Calculate seq/s
                seq_per_sec = batch_size / result['median_time']
                
                table_data.append({
                    "Model": model_name,
                    "Seq Len": seq_length,
                    "Batch": batch_size,
                    "Peak memory (GB)": round(result['median_peak_memory_reserved_gb'], 2),
                    "seq/s": round(seq_per_sec, 2),
                    "tokens/s": f"{int(result['median_tokens_per_sec']):,}",
                    "Status": "OK"
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM with batch_size {batch_size}")
                    table_data.append({
                        "Model": model_name,
                        "Seq Len": seq_length,
                        "Batch": batch_size,
                        "Peak memory (GB)": "-",
                        "seq/s": "-",
                        "tokens/s": "-",
                        "Status": "OOM"
                    })
                    # Clear cache just in case
                    if "cuda" in device:
                        torch.cuda.empty_cache()
                else:
                    print(f"  Error with batch_size {batch_size}: {e}")
                    table_data.append({
                        "Model": model_name,
                        "Seq Len": seq_length,
                        "Batch": batch_size,
                        "Peak memory (GB)": "-",
                        "seq/s": "-",
                        "tokens/s": "-",
                        "Status": "Error"
                    })
            except Exception as e:
                print(f"  Error with batch_size {batch_size}: {e}")
                table_data.append({
                    "Model": model_name,
                    "Seq Len": seq_length,
                    "Batch": batch_size,
                    "Peak memory (GB)": "-",
                    "seq/s": "-",
                    "tokens/s": "-",
                    "Status": "Error"
                })

    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)

    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    
    # Detect versions for filename
    def get_ver(pkg):
        try:
            return importlib.metadata.version(pkg)
        except:
            return "NA"

    torch_ver = get_ver("torch")
    trans_ver = get_ver("transformers")
    mamba_ver = get_ver("mamba_ssm")
    if mamba_ver == "NA": mamba_ver = get_ver("mamba-ssm")
    causal_ver = get_ver("causal_conv1d")
    if causal_ver == "NA": causal_ver = get_ver("causal-conv1d")
    
    version_str = f"torch{torch_ver}_trans{trans_ver}_mamba{mamba_ver}_causal{causal_ver}"
    output_filename = f"benchmark_results_{version_str}.tsv"

    # Save to CSV
    df.to_csv(output_filename, index=False, sep="\t")
    print(f"\nResults saved to {output_filename}")

    return df

if __name__ == "__main__":
    # Define your three DNA LM model paths
    model_paths = [
        ("kuleshov-group/PlantCAD2-Small-l24-d0768", "pcv2-small"),
        ("kuleshov-group/PlantCAD2-Medium-l48-d1024", "pcv2-medium"),
        ("kuleshov-group/PlantCAD2-Large-l48-d1536", "pcv2-large")
    ]
    
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run benchmark with multiple batch sizes
    batch_sizes = [1, 4, 8, 16, 32, 64]  # Adjust based on your GPU memory
    results = compare_models(model_paths, seq_length=8192, batch_sizes=batch_sizes, device=device)
