#!/usr/bin/env python3
"""
MalConv Analyzer Tool
Author: DatNLQ
Description: Advanced malware analysis tool using MalConv with evasion techniques
Version: 1.0
"""

import os
import magic
import numpy as np
from secml.array import CArray
import gc
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
import sys
from typing import Dict, Any, Optional, Tuple
import vt
import hashlib
import time
import resource
import psutil

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


# Banner art
BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                     MalConv Analyzer Tool                      ║
║                Advanced Malware Analysis & Evasion             ║
╚═══════════════════════════════════════════════════════════════╝
"""

# Configure logging
def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration with timestamp in filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'malconv_analysis_{timestamp}.log'
    
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Starting MalConv Analyzer Tool")
    logging.debug(f"Debug mode: {debug}")

def print_banner() -> None:
    """Print tool banner with color"""
    print("\033[94m" + BANNER + "\033[0m")

def print_step(message: str, is_error: bool = False, is_warning: bool = False) -> None:
    """Enhanced step printing with multiple color options"""
    if is_error:
        color = "\033[91m"  # Red
        prefix = "[ERROR]"
        logging.error(message)
    elif is_warning:
        color = "\033[93m"  # Yellow
        prefix = "[WARNING]"
        logging.warning(message)
    else:
        color = "\033[92m"  # Green
        prefix = "[+]"
        logging.info(message)
    
    print(f"{color}{prefix} {message}\033[0m")

def monitor_memory():
    """Monitor memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    print_step(f"Memory Usage: {mem_info.rss / 1024 / 1024:.1f} MB", is_warning=True)

class MalwareAnalyzer:
    """Main class for malware analysis operations"""
    
    def __init__(self, output_folder: str, vt_api_key: str = None, debug: bool = False):
        self.output_folder = output_folder
        self.debug = debug
        self.virustotal = None
        
        if vt_api_key:
            self.initialize_virustotal(vt_api_key)
        
        self._initialize_models()
    
    def initialize_virustotal(self, api_key: str) -> None:
        """Initialize VirusTotal client"""
        try:
            print_step("Initializing VirusTotal client...")
            self.virustotal = vt.Client(api_key)
            print_step("VirusTotal client initialized")
        except Exception as e:
            print_step(f"Error initializing VirusTotal: {str(e)}", is_error=True)
            raise

    def _initialize_models(self) -> None:
        """Initialize MalConv model with memory optimization"""
        try:
            print_step("Initializing MalConv model...")
            # Set memory limits
            resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, -1))  # 2GB limit
            
            self.net = MalConv()
            self.net = CClassifierEnd2EndMalware(self.net)
            self.net.load_pretrained_model()
            
            # Force garbage collection
            gc.collect()
            
            print_step("Initializing attack methods...")
            self.header_attack = CHeaderEvasion(
                self.net,
                random_init=False,
                iterations=25,  # Reduced iterations
                optimize_all_dos=False,
                threshold=0.1  # Increased threshold
            )
            
            self.fgsm_attack = CKreukEvasion(
                self.net,
                how_many_padding_bytes=512,  # Reduced padding size
                epsilon=4.0,
                iterations=3  # Reduced iterations
            )
            
        except Exception as e:
            print_step(f"Error initializing models: {str(e)}", is_error=True)
            raise

    def analyze_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze file with memory monitoring"""
        try:
            monitor_memory()
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Validate PE format
            if "PE32" not in magic.from_file(file_path):
                raise ValueError("Not a valid PE32 file")
                
            print_step(f"Analyzing file: {file_path}")
            logging.debug(f"Starting analysis of {file_path}")
            
            # Load and prepare file
            with open(file_path, "rb") as f:
                code = f.read()
            
            file_size = len(code)
            print_step(f"File size: {file_size/1024:.2f} KB")
            
            # Convert to numpy array
            x = End2EndModel.bytes_to_numpy(
                code,
                self.net.get_input_max_length(),
                256,
                False
            )
            
            # Initial prediction
            _, confidence = self.net.predict(CArray(x), True)
            mal_score = confidence[0,1].item()
            
            result = {
                "filename": os.path.basename(file_path),
                "file_size": file_size,
                "original_score": mal_score,
                "classification": "Malware" if mal_score > 0.5 else "Benign",
                "vt_original": None,  # Add VirusTotal results
                "header_evasion": None,
                "fgsm_evasion": None
            }

            # Check original file with VirusTotal
            if self.virustotal:
                vt_results = self._check_virustotal(file_path)
                if vt_results:
                    result["vt_original"] = vt_results

            # Perform evasion attacks if malware
            if mal_score > 0.5:
                # Header Evasion
                try:
                    header_result = self._perform_header_evasion(x, file_path)
                    if header_result.get("header_evasion_path"):
                        if self.virustotal:
                            vt_results = self._check_virustotal(header_result["header_evasion_path"])
                            if vt_results:
                                header_result["vt_scan"] = vt_results
                    result.update(header_result)
                except Exception as e:
                    logging.error(f"Header evasion failed: {str(e)}")

                # FGSM Attack
                try:
                    fgsm_result = self._perform_fgsm_attack(x, file_path)
                    if fgsm_result.get("fgsm_evasion_path"):
                        if self.virustotal:
                            vt_results = self._check_virustotal(fgsm_result["fgsm_evasion_path"])
                            if vt_results:
                                fgsm_result["vt_scan"] = vt_results
                except Exception as e:
                    logging.error(f"FGSM attack failed: {str(e)}")
            
            # Print analysis report
            self.print_analysis_report(result)
            
            return result
        except Exception as e:
            print_step(f"Error analyzing file {file_path}: {str(e)}", is_error=True)
            return None
        finally:
            gc.collect()

    def _perform_header_evasion(self, x: np.ndarray, file_path: str) -> Dict[str, Any]:
        """Perform header evasion attack and return results"""
        try:
            print_step("Performing Header Evasion Attack...")
            # Run header evasion with proper parameters
            y_pred, adv_score, adv_ds, _ = self.header_attack.run(
                CArray(x), 
                CArray([1])
            )
            print_step("Header Evasion Attack completed")
            
            if adv_ds is None:
                raise ValueError("Header evasion failed to generate adversarial example")
            
            # Save evasion sample
            evasion_path = os.path.join(self.output_folder, f"header_evasion_{os.path.basename(file_path)}")
            success = self.header_attack.create_real_sample_from_adv(
                file_path,
                adv_ds.X[0,:],
                evasion_path
            )
            
            if not os.path.exists(evasion_path):
                raise FileNotFoundError("Failed to save header evasion sample")
                
            print_step(f"Evasion sample saved: {evasion_path}")
            
            # Get final confidence score
            _, confidence = self.net.predict(CArray(adv_ds.X), True)
            evasion_score = confidence[0,1].item()
            
            print_step(f"Header Evasion score: {evasion_score:.4f}")
            
            return {
                "header_evasion": evasion_score,
                "header_evasion_path": evasion_path
            }
        except Exception as e:
            print_step(f"Error during header evasion: {str(e)}", is_error=True)
            logging.exception("Header evasion failed:")
            return {}

    def _perform_fgsm_attack(self, x: np.ndarray, file_path: str) -> Dict[str, Any]:
        """Perform FGSM attack and return results"""
        try:
            print_step("Performing FGSM Attack...")
            # Run FGSM attack with proper parameters
            y_pred, adv_score, adv_ds, _ = self.fgsm_attack.run(
                CArray(x), 
                CArray([1])
            )
            print_step("FGSM Attack completed")
            
            if adv_ds is None:
                raise ValueError("FGSM attack failed to generate adversarial example")
            
            # Save FGSM sample
            fgsm_path = os.path.join(self.output_folder, f"fgsm_{os.path.basename(file_path)}")
            success = self.fgsm_attack.create_real_sample_from_adv(
                file_path,
                adv_ds.X[0,:],
                fgsm_path
            )
            
            if not os.path.exists(fgsm_path):
                raise FileNotFoundError("Failed to save FGSM sample")
                
            print_step(f"FGSM sample saved: {fgsm_path}")
            
            # Get final confidence score
            _, confidence = self.net.predict(CArray(adv_ds.X), True)
            fgsm_score = confidence[0,1].item()
            
            print_step(f"FGSM score: {fgsm_score:.4f}")
            
            return {
                "fgsm_evasion": fgsm_score,
                "fgsm_evasion_path": fgsm_path
            }
        except Exception as e:
            print_step(f"Error during FGSM attack: {str(e)}", is_error=True)
            logging.exception("FGSM attack failed:")
            return {}

    def _check_virustotal(self, file_path: str) -> Dict:
        """Check file with VirusTotal API"""
        try:
            if not self.virustotal:
                return {}

            print_step("Scanning with VirusTotal...")
            file_hash = self._get_file_hash(file_path)

            try:
                # First try to get existing report
                file_report = self.virustotal.get_object(f"/files/{file_hash}")
                return self._parse_vt_report(file_report)
            except vt.APIError as e:
                # If file not found or other API error, upload and scan
                if e.code == "NotFoundError":
                    print_step("File not found in VT database, uploading...", is_warning=True)
                    with open(file_path, "rb") as f:
                        upload_result = self.virustotal.scan_file(f)
                    
                    # Wait for analysis to complete (max 60 seconds)
                    for _ in range(60):
                        analysis = self.virustotal.get_object(f"/analyses/{upload_result.id}")
                        if analysis.status == "completed":
                            file_report = self.virustotal.get_object(f"/files/{file_hash}")
                            return self._parse_vt_report(file_report)
                        time.sleep(1)
                    
                    print_step("VirusTotal analysis timeout", is_warning=True)
                    return {}
                else:
                    raise  # Re-raise other API errors

        except Exception as e:
            print_step(f"VirusTotal scan error: {str(e)}", is_error=True)
            logging.exception("VirusTotal scan failed:")
            return {}

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _parse_vt_report(self, report) -> Dict:
        """Parse VirusTotal report into summary format"""
        return {
            "positives": report.last_analysis_stats.get("malicious", 0),
            "total": sum(report.last_analysis_stats.values()),
            "scan_date": report.last_analysis_date,
            "permalink": f"https://www.virustotal.com/gui/file/{report.id}"
        }

    def print_analysis_report(self, result: Dict[str, Any]) -> None:
        """Print detailed analysis report"""
        if not result:
            return
            
        print("\n" + "="*50)
        print("           MALWARE ANALYSIS REPORT           ")
        print("="*50)
        print(f"File: {result['filename']}")
        print(f"Size: {result['file_size']/1024:.2f} KB")
        print(f"Classification: {result['classification']}")
        print(f"Confidence Score: {result['original_score']:.4f}")
        
        # Print VirusTotal results for original file
        if result.get("vt_original"):
            print("\nVirusTotal Results (Original):")
            print(f"- Detections: {result['vt_original']['positives']}/{result['vt_original']['total']}")
            print(f"- Scan Date: {result['vt_original']['scan_date']}")
            print(f"- Report URL: {result['vt_original']['permalink']}")
        
        # Print Header Evasion results
        if result.get("header_evasion"):
            print("\nHeader Evasion Attack:")
            print(f"- Final Score: {result['header_evasion']:.4f}")
            print(f"- Sample saved as: header_evasion_{result['filename']}")
            if "vt_scan" in result:
                print("\nVirusTotal Results (Header Evasion):")
                print(f"- Detections: {result['vt_scan']['positives']}/{result['vt_scan']['total']}")
                print(f"- Scan Date: {result['vt_scan']['scan_date']}")
                print(f"- Report URL: {result['vt_scan']['permalink']}")
        
        # Print FGSM results
        if result.get("fgsm_evasion"):
            print("\nFGSM Attack:")
            print(f"- Final Score: {result['fgsm_evasion']:.4f}")
            print(f"- Sample saved as: fgsm_{result['filename']}")
            if "vt_scan" in result:
                print("\nVirusTotal Results (FGSM):")
                print(f"- Detections: {result['vt_scan']['positives']}/{result['vt_scan']['total']}")
                print(f"- Scan Date: {result['vt_scan']['scan_date']}")
                print(f"- Report URL: {result['vt_scan']['permalink']}")
        
        print("="*50 + "\n")

def main():
    """Main execution function"""
    usage_description = """
Usage Examples:
--------------
1. Basic analysis:
   python malconv_analyzer.py --target-file <file_path>
2. With VirusTotal:
   python malconv_analyzer.py --target-file <file_path> --vt-api-key <your_api_key>
3. Custom output folder:
   python malconv_analyzer.py --target-file <file_path> --output-folder results/
4. Debug mode:
   python malconv_analyzer.py --target-file <file_path> --debug
"""

    parser = argparse.ArgumentParser(
        description='Advanced Malware Analysis Tool using MalConv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=usage_description
    )
    
    parser.add_argument(
        '--target-file', 
        type=str, 
        required=True,
        help='Path to file for analysis'
    )
    
    parser.add_argument(
        '--output-folder', 
        type=str, 
        default="analysis_results",
        help='Folder to save results and modified samples'
    )
    
    parser.add_argument(
        '--vt-api-key', 
        type=str,
        help='VirusTotal API key for additional analysis'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    
    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        print("\nError: No parameters provided!")
        print("Run with --help for detailed usage information")
        sys.exit(1)

    args = parser.parse_args()
    
    # Print banner and setup logging
    print_banner()
    setup_logging(args.debug)
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = MalwareAnalyzer(
            output_folder=args.output_folder,
            vt_api_key=args.vt_api_key,
            debug=args.debug
        )
        
        # Analyze target file
        result = analyzer.analyze_file(args.target_file)
        
        if not result:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print_step("\nAnalysis interrupted by user", is_warning=True)
        sys.exit(1)
    except Exception as e:
        print_step(f"Critical error: {str(e)}", is_error=True)
        logging.exception("Critical error occurred")
        sys.exit(1)
    finally:
        # Cleanup
        gc.collect()

if __name__ == "__main__":
    main()

