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
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import aiohttp

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
def setup_logging(target_file: str, debug: bool = False) -> None:
    """Setup logging configuration with enhanced debug information"""
    if debug:
        # Create logs directory inside output folder
        logs_dir = os.path.join("logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Generate log filename with timestamp and target file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(target_file)
        log_file = os.path.join(logs_dir, f"{filename}_{timestamp}.log")
        
        # Create a logger instance
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Create formatters and add it to handlers
        debug_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(debug_format)
        console_handler.setFormatter(debug_format)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Initial debug messages
        logger.debug("=== System Information ===")
        logger.debug(f"Analysis started at: {timestamp}")
        logger.debug(f"Target file: {filename}")
        logger.debug(f"Log file: {log_file}")
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Operating system: {os.uname().sysname}")
        logger.debug(f"Machine: {os.uname().machine}")
        logger.debug(f"Process ID: {os.getpid()}")
        
        # Log memory information
        mem_info = psutil.virtual_memory()
        logger.debug(f"Total memory: {mem_info.total / (1024**3):.2f} GB")
        logger.debug(f"Available memory: {mem_info.available / (1024**3):.2f} GB")
        logger.debug("="*50)
    else:
        # Basic console logging for non-debug mode
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

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
        self.vt_api_key = vt_api_key  # Store API key
        
        if vt_api_key:
            self.initialize_virustotal(vt_api_key)
        
        self._initialize_models()
    
    def initialize_virustotal(self, api_key: str) -> None:
        """Initialize VirusTotal client"""
        try:
            print_step("Initializing VirusTotal client...")
            self.virustotal = vt.Client(api_key)
            self.vt_api_key = api_key  # Ensure API key is set
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

    @asynccontextmanager
    async def timeout(self, seconds: int) -> AsyncGenerator[None, None]:
        """Async timeout context manager"""
        try:
            task = asyncio.current_task()
            if task is None:
                raise RuntimeError("No active task")
            yield
        except asyncio.TimeoutError:
            print_step(f"Operation timed out after {seconds} seconds", is_warning=True)
            raise
        finally:
            task.cancel()

    async def _check_virustotal_async(self, file_path: str) -> Dict:
        """Asynchronous VirusTotal check"""
        if not self.virustotal:
            return {}
            
        try:
            print_step("Scanning with VirusTotal...")
            file_hash = self._get_file_hash(file_path)
            
            async with aiohttp.ClientSession() as session:
                # Try to get existing report
                try:
                    headers = {"x-apikey": self.vt_api_key}
                    async with session.get(
                        f"https://www.virustotal.com/api/v3/files/{file_hash}",
                        headers=headers,
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            report = await response.json()
                            return self._parse_vt_report(report)
                            
                        elif response.status == 404:
                            print_step("File not found in VT database, uploading...", is_warning=True)
                            # Upload file
                            data = aiohttp.FormData()
                            data.add_field('file', 
                                         open(file_path, 'rb'),
                                         filename=os.path.basename(file_path))
                                         
                            async with session.post(
                                "https://www.virustotal.com/api/v3/files",
                                data=data,
                                headers=headers,
                                timeout=60
                            ) as upload_response:
                                if upload_response.status == 200:
                                    upload_json = await upload_response.json()
                                    analysis_id = upload_json["data"]["id"]
                                    
                                    # Wait for analysis to complete
                                    for _ in range(60):
                                        async with session.get(
                                            f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
                                            headers=headers
                                        ) as analysis_response:
                                            analysis = await analysis_response.json()
                                            if analysis["data"]["attributes"]["status"] == "completed":
                                                return self._parse_vt_report(analysis)
                                        await asyncio.sleep(1)
                                        
                except asyncio.TimeoutError:
                    print_step("VirusTotal request timed out", is_warning=True)
                    return {}
                    
        except Exception as e:
            print_step(f"VirusTotal scan error: {str(e)}", is_error=True)
            logging.exception("VirusTotal scan failed:")
            return {}

    def _check_virustotal(self, file_path: str) -> Dict:
        """Synchronous wrapper for VirusTotal check"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._check_virustotal_async(file_path))
        finally:
            loop.close()

    def _get_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _parse_vt_report(self, report) -> Dict:
        """Parse VirusTotal report into summary format"""
        try:
            # Handle API v3 response format
            data = report.get("data", {})
            attributes = data.get("attributes", {})
            stats = attributes.get("last_analysis_stats", {})
            
            return {
                "positives": stats.get("malicious", 0),
                "total": sum(stats.values()) if stats else 0,
                "scan_date": attributes.get("last_analysis_date", "N/A"),
                "permalink": f"https://www.virustotal.com/gui/file/{data.get('id', 'unknown')}"
            }
        except Exception as e:
            logging.error(f"Error parsing VirusTotal report: {str(e)}")
            logging.debug(f"Raw report: {report}")
            return {
                "positives": 0,
                "total": 0,
                "scan_date": "Error parsing report",
                "permalink": "N/A"
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
    setup_logging(
        target_file=args.target_file,
        debug=args.debug
    )
    
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

