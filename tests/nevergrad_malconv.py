import argparse
import logging
from pathlib import Path
import gc
import psutil
from secml.array import CArray
from secml_malware.models import MalConv, CClassifierEnd2EndMalware, CClassifierEmber
from secml_malware.attack.blackbox.c_black_box_padding_evasion import CBlackBoxPaddingEvasionProblem
from secml_malware.attack.blackbox.c_gamma_sections_evasion import CGammaSectionsEvasionProblem
from secml_malware.attack.blackbox.c_wrapper_phi import CEnd2EndWrapperPhi, CEmberWrapperPhi
from secml_malware.attack.blackbox.ga.c_nevergrad_ga import CNevergradGeneticAlgorithm
from secml_malware.models import End2EndModel

def print_step(message, is_error=False):
    """Print formatted step message"""
    if is_error:
        print(f"\033[91m[ERROR] {message}\033[0m")
    else:
        print(f"\033[92m[+] {message}\033[0m")

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        filename='nevergrad_analysis.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def monitor_memory():
    """Monitor current memory usage"""
    process = psutil.Process()
    print_step(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def load_models():
    """Load and prepare both MalConv and EMBER models"""
    print_step("Loading MalConv model...")
    malconv = MalConv()
    malconv_model = CClassifierEnd2EndMalware(malconv)
    malconv_model.load_pretrained_model()
    wrapped_malconv = CEnd2EndWrapperPhi(malconv_model)

    print_step("Loading EMBER model...")
    tree_path = Path(__file__).parent / "models" / "ember_model.txt"
    tree = CClassifierEmber(str(tree_path))
    wrapped_tree = CEmberWrapperPhi(tree)

    return wrapped_malconv, wrapped_tree, malconv_model

def process_file(file_path, malconv_model):
    """Process single file for analysis"""
    try:
        print_step(f"Processing file: {file_path}")
        with open(file_path, "rb") as f:
            code = f.read()
            
        x = End2EndModel.bytes_to_numpy(
            code,
            malconv_model.get_input_max_length(),
            malconv_model.get_embedding_value(),
            malconv_model.get_is_shifting_values()
        )
        
        return CArray(x)
    except Exception as e:
        print_step(f"Error processing file: {str(e)}", is_error=True)
        return None

def run_padding_attack(model_wrapper, sample, name="Unknown"):
    """Run padding-based evasion attack"""
    print_step(f"Running padding attack on {name}...")
    
    try:
        problem = CBlackBoxPaddingEvasionProblem(
            model_wrapper, 
            population_size=3,
            how_many_padding_bytes=1024,
            iterations=3
        )
        engine = CNevergradGeneticAlgorithm(problem)
        results = engine.run(sample, CArray([1]))
        
        print_step(f"{name} Padding Attack Results:")
        print_step(f"Final confidence: {engine.confidences_[-1]:.4f}")
        
        return engine.confidences_[-1]
        
    except Exception as e:
        print_step(f"Error in padding attack: {str(e)}", is_error=True)
        return None

def run_gamma_sections_attack(model_wrapper, sample, goodware_folder):
    """Run gamma sections evasion attack"""
    print_step("Running gamma sections attack...")
    
    try:
        sections, _ = CGammaSectionsEvasionProblem.create_section_population_from_folder(
            goodware_folder,
            how_many=3
        )
        
        problem = CGammaSectionsEvasionProblem(
            model_wrapper=model_wrapper,
            section_population=sections,
            population_size=3,
            penalty_regularizer=1e-2,
            iterations=3
        )
        
        engine = CNevergradGeneticAlgorithm(problem, random_state=12)
        results = engine.run(sample, CArray([1]))
        
        print_step("Gamma Sections Attack Results:")
        print_step(f"Final confidence: {engine.confidences_[-1]:.4f}")
        
        return engine.confidences_[-1]
        
    except Exception as e:
        print_step(f"Error in gamma sections attack: {str(e)}", is_error=True)
        return None

def main():
    parser = argparse.ArgumentParser(description='Nevergrad-based Malware Analysis')
    parser.add_argument('--target-file', type=str, required=True,
                      help='Path to malware file to analyze')
    parser.add_argument('--goodware-folder', type=str,
                      default="goodware_samples",
                      help='Folder containing goodware samples for gamma sections attack')
    args = parser.parse_args()

    setup_logging()
    monitor_memory()

    # Load models
    wrapped_malconv, wrapped_tree, malconv_model = load_models()
    monitor_memory()

    # Process target file
    sample = process_file(args.target_file, malconv_model)
    if sample is None:
        return

    # Run attacks
    print("\n=== Starting Attacks ===")
    
    # MalConv Padding Attack
    malconv_padding = run_padding_attack(wrapped_malconv, sample, "MalConv")
    monitor_memory()
    
    # EMBER Padding Attack
    ember_padding = run_padding_attack(wrapped_tree, sample, "EMBER")
    monitor_memory()
    
    # EMBER Gamma Sections Attack
    ember_gamma = run_gamma_sections_attack(wrapped_tree, sample, args.goodware_folder)
    monitor_memory()

    # Print final results
    print("\n=== Analysis Results ===")
    print(f"File: {Path(args.target_file).name}")
    print(f"MalConv Padding Attack Score: {malconv_padding:.4f}")
    print(f"EMBER Padding Attack Score: {ember_padding:.4f}")
    print(f"EMBER Gamma Sections Score: {ember_gamma:.4f}")
    print("========================\n")

if __name__ == "__main__":
    main()