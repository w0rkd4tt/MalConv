import os
from pathlib import Path
import logging
import magic
from secml.array import CArray
import argparse
import numpy as np


import numpy
if not hasattr(numpy, 'infty'):
    numpy.infty = numpy.inf

if not hasattr(numpy, 'unicode_'):
    numpy.unicode_ = numpy.str_

def patch_array_creation():
    original_array = np.array
    def patched_array(*args, **kwargs):
        if 'copy' in kwargs and not kwargs['copy']:
            return np.asarray(*args)
        return original_array(*args, **kwargs)
    numpy.array = patched_array

patch_array_creation()

from secml_malware.models.malconv2mb import MalConv2MB
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware
from secml_malware.models import End2EndModel
from secml_malware.attack.whitebox.c_header_evasion import CHeaderEvasion
from secml_malware.attack.blackbox.c_black_box_padding_evasion import CBlackBoxPaddingEvasionProblem
from secml_malware.attack.blackbox.c_wrapper_phi import CEnd2EndWrapperPhi
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
import gc
import psutil

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        filename='malconv2_analysis.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def print_step(message, is_error=False):
    """Print step with formatting"""
    if is_error:
        print(f"\033[91m[ERROR] {message}\033[0m")
    else:
        print(f"\033[92m[+] {message}\033[0m")

def load_malconv2_model():
    """Initialize and load MalConv2MB model"""
    print_step("Loading MalConv2MB model...")
    model = MalConv2MB()
    model = CClassifierEnd2EndMalware(model, input_shape=(1, 2000000))
    model.load_pretrained_model()
    return model

def process_malware_samples(folder_path, model):
    """Process malware samples from folder"""
    print_step(f"Processing samples from {folder_path}")
    X, y, file_names = [], [], []
    
    for f in os.listdir(folder_path):
        path = os.path.join(folder_path, f)
        
        # Filter files
        if 'petya' not in path or "PE32" not in magic.from_file(path):
            continue
            
        try:
            # Read and process file
            with open(path, "rb") as file_handle:
                code = file_handle.read()
                
            x = End2EndModel.bytes_to_numpy(
                code, 
                model.get_input_max_length(), 
                model.get_embedding_value(),
                model.get_is_shifting_values()
            )
            
            # Predict and filter by confidence
            pred, confidence = model.predict(CArray(x), True)
            mal_score = confidence[0, 1].item()
            
            if mal_score < 0.5:
                continue
                
            print_step(f"Added {f} with confidence {mal_score:.4f}")
            X.append(x)
            conf = confidence[1][0].item()
            y.append([1 - conf, conf])
            file_names.append(path)
            
        except Exception as e:
            print_step(f"Error processing {f}: {str(e)}", is_error=True)
            logging.error(f"Error processing {f}: {str(e)}")
            
    return X, y, file_names

def monitor_memory():
    """Monitor current memory usage"""
    process = psutil.Process(os.getpid())
    print_step(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def process_single_file(file_path, model):
    """Process a single malware file with memory management and proper padding"""
    print_step(f"Processing file: {file_path}")
    monitor_memory()
    
    try:
        # Check file size before processing
        file_size = os.path.getsize(file_path)
        print_step(f"File size: {file_size/1024:.2f} KB")
        
        # Read file
        with open(file_path, "rb") as file_handle:
            code = file_handle.read()
        
        # Clear memory after reading
        gc.collect()
        monitor_memory()
            
        # Pad or truncate the input to match required size
        input_size = model.get_input_max_length()
        if len(code) < input_size:
            # Pad with zeros if file is smaller
            padded_code = code + b'\x00' * (input_size - len(code))
            print_step(f"Padded file from {len(code)} to {len(padded_code)} bytes")
            x = End2EndModel.bytes_to_numpy(
                padded_code,
                input_size,
                model.get_embedding_value(),
                model.get_is_shifting_values()
            )
        else:
            # Truncate if file is larger
            print_step(f"Truncating file from {len(code)} to {input_size} bytes")
            x = End2EndModel.bytes_to_numpy(
                code[:input_size],
                input_size,
                model.get_embedding_value(),
                model.get_is_shifting_values()
            )
        
        # Reshape to match model input shape
        x = x.reshape(1, input_size)
        
        # Clear original code from memory
        del code
        gc.collect()
        monitor_memory()
        
        # Predict
        pred, confidence = model.predict(CArray(x), True)
        mal_score = confidence[0, 1].item()
        
        print_step(f"Original malware score: {mal_score:.4f}")
        
        return x, mal_score, file_path
        
    except Exception as e:
        print_step(f"Error processing file: {str(e)}", is_error=True)
        logging.error(f"Error processing file: {str(e)}")
        # Ensure cleanup on error
        gc.collect()
        return None, None, None

def run_header_evasion(model, sample, chunk_size=100):
    """Run Header Evasion attack with improved memory management"""
    print_step("Starting Header Evasion attack...")
    monitor_memory()
    
    try:
        # Free memory before attack
        gc.collect()
        monitor_memory()
        
        # Create attack with minimal parameters
        attack = CHeaderEvasion(
            model, 
            random_init=False, 
            iterations=3,  # Further reduced iterations
            optimize_all_dos=False,  # Disable full DOS optimization
            threshold=0.1  # Increased threshold to converge faster
        )
        
        print_step("Running Header Evasion with minimal parameters...")
        
        # Process in chunks to reduce memory usage
        sample_chunks = np.array_split(sample, chunk_size)
        results = []
        
        for i, chunk in enumerate(sample_chunks):
            if i % 10 == 0:  # Print progress every 10 chunks
                print_step(f"Processing chunk {i+1}/{len(sample_chunks)}")
                monitor_memory()
            
            # Run attack on chunk without return_solution parameter
            y_pred, adv_score, adv_ds, _ = attack.run(
                CArray(chunk), 
                CArray([1]),
                ds_init=None  # Don't store initialization
            )
            
            results.append(adv_score)
            
            # Force cleanup after each chunk
            gc.collect()
        
        # Aggregate results
        confidences = attack.confidences_
        
        # Cleanup
        del attack
        del results
        gc.collect()
        
        monitor_memory()
        print_step("Header Evasion completed")
        
        return confidences if confidences is not None else [1.0]
        
    except Exception as e:
        print_step(f"Error in Header Evasion: {str(e)}", is_error=True)
        logging.error(f"Header Evasion error: {str(e)}")
        gc.collect()
        return [1.0]

def run_blackbox_padding(model, sample):
    """Run Black Box Padding attack with memory management"""
    print_step("Starting Black Box Padding attack...")
    monitor_memory()
    
    try:
        wrapper = CEnd2EndWrapperPhi(model)
        ga = CGeneticAlgorithm(
            CBlackBoxPaddingEvasionProblem(
                wrapper, 
                iterations=1,
                how_many_padding_bytes=50,  # Reduce padding size
                population_size=2  # Reduce population size
            )
        )
        
        # Run attack and immediately get results
        result = ga.run(CArray(sample), CArray([1]))
        confidences = ga.confidences_
        
        # Clear objects
        del ga
        del wrapper
        gc.collect()
        monitor_memory()
        
        return confidences
        
    except Exception as e:
        print_step(f"Error in Black Box Padding: {str(e)}", is_error=True)
        gc.collect()
        return None

def setup_memory_management():
    """Configure aggressive memory management"""
    # Set garbage collection thresholds
    gc.set_threshold(100, 5, 5)  # More aggressive collection
    
    # Set memory monitoring
    process = psutil.Process(os.getpid())
    if hasattr(process, "set_memory_limits"):
        # Limit memory to 75% of system RAM
        total_mem = psutil.virtual_memory().total
        process.set_memory_limits(int(total_mem * 0.75))

def main():
    """Main execution function"""
    setup_memory_management()
    parser = argparse.ArgumentParser(description='MalConv2 Black Box Analysis')
    parser.add_argument('--target-file', type=str, help='Path to specific malware file to analyze')
    parser.add_argument('--samples-folder', type=str, default="malware_samples",
                      help='Folder containing malware samples')
    parser.add_argument('--chunk-size', type=int, default=100,
                      help='Chunk size for Header Evasion attack')
    args = parser.parse_args()
    
    setup_logging()
    model = load_malconv2_model()
    
    if args.target_file:
        # Analyze single file
        print_step(f"\nAnalyzing target file: {args.target_file}")
        X, score, file_path = process_single_file(args.target_file, model)
        
        if X is not None:
            # Run attacks
            header_confidences = run_header_evasion(model, X, args.chunk_size)
            ga_confidences = run_blackbox_padding(model, X)
            
            # Print results with safety checks
            print("\n=== Analysis Results ===")
            print(f"File: {os.path.basename(args.target_file)}")
            print(f"Original malware score: {score:.4f}")
            
            # Safe printing of confidences
            header_conf = header_confidences[-1] if header_confidences else 1.0
            ga_conf = ga_confidences[-1] if ga_confidences else 1.0
            
            print(f"Header Evasion final confidence: {header_conf:.4f}")
            print(f"GA Padding final confidence: {ga_conf:.4f}")
            print("=====================")
    else:
        # Process folder
        folder = str(Path(__file__).parent / args.samples_folder)
        X, y, file_names = process_malware_samples(folder, model)
        
        if not X:
            print_step("No valid samples found!", is_error=True)
            return
            
        # Run attacks on first sample
        print_step(f"\nAnalyzing sample: {file_names[0]}")
        header_confidences = run_header_evasion(model, X[0])
        ga_confidences = run_blackbox_padding(model, X[0])
        
        # Print results
        print("\n=== Analysis Results ===")
        print(f"Sample: {os.path.basename(file_names[0])}")
        print(f"Header Evasion final confidence: {header_confidences[-1]}")
        print(f"GA Padding final confidence: {ga_confidences[-1]}")
        print("=====================")

if __name__ == "__main__":
    main()