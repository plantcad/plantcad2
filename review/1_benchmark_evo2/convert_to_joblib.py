
import sys
import os
import pickle
import joblib
import psutil

def get_memory_usage_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024

def convert_to_joblib(input_pkl, output_pkl=None):
    if output_pkl is None:
        output_pkl = input_pkl
        
    print(f"Converting {input_pkl} to joblib format...")
    print(f"Current memory usage: {get_memory_usage_gb():.2f} GB")
    
    try:
        # Load with standard pickle (this requires high RAM)
        print("Loading original pickle file (this may take a while and requires High Memory)...")
        with open(input_pkl, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Loaded! Memory usage: {get_memory_usage_gb():.2f} GB")
        
        # Save with joblib
        print(f"Dumping to {output_pkl} using joblib...")
        joblib.dump(data, output_pkl)
        print("Conversion complete!")
        
    except MemoryError:
        print("\nERROR: Out of memory while loading the pickle file.")
        print("Please run this script on a high-memory node (e.g. >256GB RAM).")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_joblib.py <input.pkl> [output.pkl]")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file
    
    convert_to_joblib(input_file, output_file)
