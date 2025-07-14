import time
import numpy as np
from typing import List, Callable
import torch
from transformers import AutoModel, AutoTokenizer

def load_model(model_path: str, device: str = "cuda"):
    """Load HuggingFace model in bfloat16"""
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except:
        tokenizer = None
        print(f"Warning: Could not load tokenizer for {model_path}")
    
    return model, tokenizer

def benchmark_model(model_path: str, model_name: str, 
                   seq_length: int = 8192, n_warmup: int = 20, 
                   n_benchmark: int = 10, batch_size: int = 1, device: str = "cuda") -> dict:
    """
    Benchmark DNA language model inference speed
    
    Args:
        model_path: HuggingFace model path
        model_name: Name of the model for logging
        seq_length: Length of DNA sequence
        n_warmup: Number of warmup runs
        n_benchmark: Number of benchmark runs
        batch_size: Batch size for inference
        device: Device to run on
    
    Returns:
        dict with benchmark results
    """
    print(f"Benchmarking {model_name} (batch_size={batch_size})...")
    
    # Load model
    model, tokenizer = load_model(model_path, device)
    
    # Generate random DNA sequences for batch
    bases = ['A', 'T', 'G', 'C']
    sequences = []
    for _ in range(batch_size):
        sequence = ''.join(np.random.choice(bases, seq_length))
        sequences.append(sequence)
    
    # Tokenize if tokenizer available
    inputs = tokenizer(sequences, return_tensors="pt", truncation=True, 
                        max_length=seq_length, padding=True)['input_ids'].to(device)

    
    # Warmup
    print(f"  Warming up with {n_warmup} batches...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(inputs)
    
    # Clear cache
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # Benchmark
    print(f"  Running {n_benchmark} benchmark iterations...")
    times = []
    
    for i in range(n_benchmark):
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            output = model(inputs)
        
        if device == "cuda":
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        
        # Get output tokens count
        if hasattr(output, 'last_hidden_state'):
            tokens_per_batch = output.last_hidden_state.shape[1] * batch_size
        elif hasattr(output, 'logits'):
            tokens_per_batch = output.logits.shape[1] * batch_size
        else:
            tokens_per_batch = seq_length * batch_size  # fallback
        
        tokens_per_second = tokens_per_batch / inference_time
        print(f"    Run {i+1}: {tokens_per_second:.2f} tokens/sec (batch throughput)")
    
    # Calculate statistics
    times = np.array(times)
    tokens_per_second_list = [(seq_length * batch_size) / t for t in times]
    
    results = {
        'model_name': model_name,
        'batch_size': batch_size,
        'median_time': np.median(times),
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'median_tokens_per_sec': np.median(tokens_per_second_list),
        'mean_tokens_per_sec': np.mean(tokens_per_second_list),
        'median_tokens_per_sec_per_sample': np.median(tokens_per_second_list) / batch_size,
        'all_times': times.tolist(),
        'all_tokens_per_sec': tokens_per_second_list
    }
    
    # Clean up
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return results

def compare_models(model_paths: List[tuple], seq_length: int = 8192, 
                  batch_sizes: List[int] = [1, 4, 8, 16], device: str = "cuda"):
    """
    Compare multiple DNA LM models with different batch sizes
    
    Args:
        model_paths: List of (model_path, model_name) tuples
        seq_length: DNA sequence length
        batch_sizes: List of batch sizes to test
        device: Device to run on
    """
    all_results = []
    
    for model_path, model_name in model_paths:
        print(f"\n{'='*50}")
        print(f"Testing {model_name}")
        print(f"{'='*50}")
        
        model_results = []
        for batch_size in batch_sizes:
            try:
                result = benchmark_model(model_path, model_name, seq_length, 
                                       batch_size=batch_size, device=device)
                model_results.append(result)
            except Exception as e:
                print(f"  Error with batch_size {batch_size}: {e}")
                continue
        
        all_results.extend(model_results)
    
    # Print comparison
    print("\n" + "="*80)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*80)
    
    for result in all_results:
        print(f"\n{result['model_name']} (batch_size={result['batch_size']}):")
        print(f"  Median inference time: {result['median_time']:.4f}s")
        print(f"  Total throughput: {result['median_tokens_per_sec']:.2f} tokens/sec")
        print(f"  Per-sample throughput: {result['median_tokens_per_sec_per_sample']:.2f} tokens/sec")
    
    # Find best configurations
    best_total = max(all_results, key=lambda x: x['median_tokens_per_sec'])
    best_per_sample = max(all_results, key=lambda x: x['median_tokens_per_sec_per_sample'])
    
    print(f"\nBest total throughput: {best_total['model_name']} "
          f"(batch_size={best_total['batch_size']}, {best_total['median_tokens_per_sec']:.2f} tokens/sec)")
    print(f"Best per-sample throughput: {best_per_sample['model_name']} "
          f"(batch_size={best_per_sample['batch_size']}, {best_per_sample['median_tokens_per_sec_per_sample']:.2f} tokens/sec)")
    
    return all_results

# Example usage:
if __name__ == "__main__":
    # Define your three DNA LM model paths
    model_paths = [
        ("kuleshov-group/PlantCaduceus_l32", "pcv1"),
        ("maize-genetics/pcv2-l24-d0768", "pcv2-small"),
        ("maize-genetics/pcv2-l48-d1024", "pcv2-medium"),
        ("maize-genetics/pcv2-l48-d1536", "pcv2-large")
    ]
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run benchmark with multiple batch sizes
    batch_sizes = [1, 4, 8, 16]  # Adjust based on your GPU memory
    results = compare_models(model_paths, seq_length=8192, batch_sizes=batch_sizes, device=device)