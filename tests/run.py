import os
import magic
import numpy as np
from secml.array import CArray
import gc
import logging
from tqdm import tqdm
import argparse

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

from secml_malware.models.malconv import MalConv
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel
from secml_malware.attack.whitebox.c_header_evasion import CHeaderEvasion
from secml_malware.attack.whitebox import CKreukEvasion

def print_step(message, is_error=False):
    """Print step with formatting"""
    if is_error:
        print(f"\033[91m[ERROR] {message}\033[0m")  # Red color for errors
    else:
        print(f"\033[92m[+] {message}\033[0m")  # Green color for normal steps

def analyze_malware_file(file_path, net, header_attack, fgsm_attack, output_folder):
    """Analyze a single malware file with both Header Evasion and FGSM attacks"""
    try:
        logging.info(f"Starting analysis of {file_path}")
        
        # Load và chuẩn bị file
        with open(file_path, "rb") as file_handle:
            code = file_handle.read()
        logging.info(f"File size: {len(code)} bytes")
            
        x = End2EndModel.bytes_to_numpy(
            code, 
            net.get_input_max_length(), 
            256, 
            False
        )
        logging.info(f"Converted to numpy array shape: {x.shape}")
        
        # Dự đoán ban đầu
        _, confidence = net.predict(CArray(x), True)
        mal_score = confidence[0,1].item()
        logging.info(f"Original malware score: {mal_score:.4f}")
        
        result = {
            "filename": os.path.basename(file_path),
            "original_score": mal_score,
            "header_evasion": None,
            "fgsm_evasion": None
        }

        if mal_score > 0.5:
            # 1. Header Evasion Attack
            try:
                logging.info("Starting Header Evasion attack...")
                y_pred_header, adv_score_header, adv_ds_header, _ = header_attack.run(
                    CArray(x), 
                    CArray(mal_score)
                )
                logging.info(f"Header Evasion prediction: {y_pred_header}")
                logging.info(f"Header Evasion adversarial score: {adv_score_header}")
                
                # Tạo và lưu adversarial sample từ Header Evasion
                header_sample = header_attack.create_real_sample_from_adv(
                    file_path,
                    adv_ds_header.X[0,:],
                    os.path.join(output_folder, f"header_evasion_{result['filename']}")
                )
                result["header_evasion"] = header_attack.confidences_[-1]
                logging.info(f"Header evasion final score: {result['header_evasion']:.4f}")
                
            except Exception as e:
                logging.error(f"Error during Header Evasion attack: {str(e)}")
                logging.error(f"Header attack parameters: random_init={header_attack.random_init}, "
                            f"iterations={header_attack.iterations}")

            # 2. FGSM Attack
            try:
                print_step("Starting FGSM attack...")
                logging.info("Starting FGSM attack...")
                
                print_step("Running FGSM optimization...")
                y_pred_fgsm, adv_score_fgsm, adv_ds_fgsm, _ = fgsm_attack.run(
                    CArray(x), 
                    y=CArray(mal_score),
                    ds_init=None
                )
                
                print_step(f"FGSM prediction: {y_pred_fgsm}")
                print_step(f"FGSM adversarial score: {adv_score_fgsm}")
                
                if adv_ds_fgsm is not None:
                    print_step("Creating FGSM adversarial sample...")
                    try:
                        fgsm_sample = fgsm_attack.create_real_sample_from_adv(
                            file_path,
                            adv_ds_fgsm.X[0,:],
                            os.path.join(output_folder, f"fgsm_{result['filename']}")
                        )
                        result["fgsm_evasion"] = fgsm_attack.confidences_[-1]
                        print_step(f"FGSM evasion final score: {result['fgsm_evasion']:.4f}")
                        logging.info(f"FGSM evasion final score: {result['fgsm_evasion']:.4f}")
                    except Exception as e:
                        print_step(f"Error creating FGSM sample: {str(e)}", is_error=True)
                        logging.error(f"Error creating FGSM sample: {str(e)}")
                else:
                    print_step("FGSM attack did not produce valid adversarial sample", is_error=True)
                    
            except Exception as e:
                print_step(f"Error during FGSM attack: {str(e)}", is_error=True)
                logging.error(f"Error during FGSM attack: {str(e)}")
                logging.error(f"FGSM parameters: padding={fgsm_attack.how_many_padding_bytes}, "
                            f"epsilon={fgsm_attack.epsilon}, iterations={fgsm_attack.iterations}")

        # Thu hồi bộ nhớ
        del x, code
        gc.collect()
        
        return result
    
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None

def analyze_malware(malware_folder, output_folder, batch_size=5):
    """Analyze malware files in batches"""
    
    # Setup logging
    logging.basicConfig(
        filename='malware_analysis.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize model and attacks
    net = MalConv()
    net = CClassifierEnd2EndMalware(net)
    net.load_pretrained_model()
    
    header_attack = CHeaderEvasion(
        net, 
        random_init=False,
        iterations=50,
        optimize_all_dos=False,
        threshold=0.0
    )
    
    fgsm_attack = CKreukEvasion(
        net,
        how_many_padding_bytes=1024,
        epsilon=4.0,
        iterations=5
    )

    results = []
    
    # Get list of PE files
    pe_files = []
    for f in os.listdir(malware_folder):
        path = os.path.join(malware_folder, f)
        if "PE32" in magic.from_file(path):
            pe_files.append(path)
    
    # Process files in batches with progress bar
    for i in tqdm(range(0, len(pe_files), batch_size)):
        batch_files = pe_files[i:i + batch_size]
        
        for file_path in batch_files:
            result = analyze_malware_file(
                file_path, 
                net, 
                header_attack, 
                fgsm_attack, 
                output_folder
            )
            if result:
                results.append(result)
                
            # Print immediate results
            if result:
                print(f"\nFile: {result['filename']}")
                print(f"Original malware score: {result['original_score']:.4f}")
                if result['header_evasion']:
                    print(f"Header evasion score: {result['header_evasion']:.4f}")
                if result['fgsm_evasion']:
                    print(f"FGSM evasion score: {result['fgsm_evasion']:.4f}")
        
        # Clear memory after each batch
        gc.collect()
    
    return results

def analyze_single_target(target_file, output_folder):
    """Analyze a specific target file"""
    print_step(f"Starting analysis of target file: {target_file}")
    
    # Setup logging
    logging.basicConfig(
        filename='malware_analysis.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Initialize model
        print_step("Loading MalConv model...")
        net = MalConv()
        net = CClassifierEnd2EndMalware(net)
        net.load_pretrained_model()
        
        # Initialize attacks
        print_step("Initializing attack methods...")
        header_attack = CHeaderEvasion(
            net, 
            random_init=False,
            iterations=50,
            optimize_all_dos=False,
            threshold=0.0
        )
        
        fgsm_attack = CKreukEvasion(
            net,
            how_many_padding_bytes=1024,
            epsilon=4.0,
            iterations=5
        )

        # Analyze file
        print_step("Starting file analysis...")
        result = analyze_malware_file(
            target_file, 
            net, 
            header_attack, 
            fgsm_attack, 
            output_folder
        )

        if result:
            print("\n=== Analysis Results ===")
            print(f"File: {result['filename']}")
            print(f"Original malware score: {result['original_score']:.4f}")
            
            if result['header_evasion']:
                print(f"Header evasion score: {result['header_evasion']:.4f}")
                print(f"Header evasion sample saved as: header_evasion_{result['filename']}")
                
            if result['fgsm_evasion']:
                print(f"FGSM evasion score: {result['fgsm_evasion']:.4f}")
                print(f"FGSM evasion sample saved as: fgsm_{result['filename']}")
                
            print("======================")
            
        return result

    except Exception as e:
        print_step(f"Error analyzing target file: {str(e)}", is_error=True)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Malware Analysis with MalConv')
    parser.add_argument('--target-file', type=str, help='Path to specific file to analyze')
    parser.add_argument('--malware-folder', type=str, default="/home/datnlq/Malconv/MalConv/samples",
                        help='Folder containing malware samples')
    parser.add_argument('--output-folder', type=str, default="/home/datnlq/Malconv/MalConv/results",
                        help='Folder to save results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    try:
        if args.target_file:
            if not os.path.exists(args.target_file):
                print_step(f"Target file not found: {args.target_file}", is_error=True)
            else:
                analyze_single_target(args.target_file, args.output_folder)
        else:
            results = analyze_malware(args.malware_folder, args.output_folder)
            
    except Exception as e:
        print_step(f"Error during analysis: {str(e)}", is_error=True)
