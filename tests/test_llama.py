#!/usr/bin/env python3

"""
This script runs Spectrum's Page+SVHT SNR profiling and optional Page-SVHT denoising
for every linear layer of a specified LLaMA model, without requiring CLI arguments.
"""
import os
import sys
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectrum.spectrum import ModelModifier

MODEL_NAME   = "meta-llama/Meta-Llama-3-8B"
BATCH_SIZE   = 1
TOP_PERCENT  = 50
DO_DENOISE   = False  
OUTPUT_DIR   = "llama_denoised" 

def main():
    print(f"Loading model '{MODEL_NAME}'...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loaded. dtype={next(model.parameters()).dtype}")

    mod = ModelModifier(
        model_name=MODEL_NAME,
        top_percent=TOP_PERCENT,
        batch_size=BATCH_SIZE
    )
    mod.model = model
    mod.use_page_svht = True

    weight_types = mod.get_weight_types()
    print(f" Profiling {len(weight_types)} weight-type groups for SNR...")
    t0 = time.time()
    mod.assess_layers_snr(weight_types)
    print(f" Profiling completed in {time.time() - t0:.1f}s")

    print(" Saving SNR results to JSON...")
    mod.save_snr_to_json()
    slug = MODEL_NAME.replace('/', '-').replace('_', '-')
    json_fn = os.path.join('model_snr_results', f"snr_results_{slug}.json")
    print(f" Generating YAML of top {TOP_PERCENT}% unfrozen layers...")
    mod.generate_unfrozen_params_yaml(json_fn, TOP_PERCENT)

    if DO_DENOISE:
        print("Running Page+SVHT denoising on all linear layers...")
        mod.denoise_all_layers()
        print(f"    Saving denoised model to '{OUTPUT_DIR}'...")
        mod.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("    Denoised checkpoint saved.")
    else:
        print("Skipping denoising. Done profiling only.")

if __name__ == "__main__":
    main()
