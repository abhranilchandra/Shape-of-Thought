import os
import json
import torch
import time
import re
import gc
import argparse
import glob
import tempfile
import shutil
import sys
import subprocess
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams

# ==========================================
# CONFIGURATION
# ==========================================

DEFAULT_BASE_MODEL = "google/gemma-2-2b" 
DEFAULT_OUTPUT_DIR = "./eval_results_code"
DEFAULT_TEST_DATASET_PATH = "./test_data.jsonl"
TIMEOUT_SECONDS = 5  # Timeout for code execution

# ==========================================
# PROMPT LOGIC (UPDATED FOR ASSERTION INJECTION)
# ==========================================

def build_mbpp_prompt(text: str, test_list: list) -> str:
    """
    Constructs the prompt dynamically.
    Appends the FIRST assertion from test_list to the problem description.
    """
    prefix = (
        "You are an expert Python programmer.\n"
        "I will give you a programming task description. Your job is to write a correct, "
        "efficient, and clean Python solution. Start directly with the coding solution.\n\n"
        "Requirements:\n"
        "- Use only the Python standard library.\n"
        "- Your code must strictly satisfy the provided assertion.\n"
        "- Respond with ONLY Python code (no backticks, no comments, no explanations).\n"
        "\nProblem:\n"
    )
    
    prompt = prefix + text.strip()
    
    # CRITICAL: Append the first assertion to guide the model
    if test_list and len(test_list) > 0:
        assertion = test_list[0].strip()
        prompt += f"\n\nYour code should satisfy the following assertion:\n{assertion}"
    
    prompt += "\n\nSolution:"
    return prompt

# ------------ GENERATION PARAMETERS ------------
MAX_TOKENS = 1024
TEMPERATURE = 0.0  # Greedy decoding for Pass@1

# ==========================================
# UTILITY FUNCTIONS: MODEL & DATA
# ==========================================

def load_jsonl_dataset(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def cleanup_llm(llm):
    """Clean up LLM resources to prevent memory leaks"""
    if hasattr(llm, "shutdown") and callable(llm.shutdown):
        try:
            llm.shutdown()
        except Exception as e:
            print(f"Error during llm.shutdown(): {e}")
    del llm
    torch.cuda.empty_cache()
    gc.collect()

def merge_and_save_peft_checkpoint(checkpoint_path, base_model_id, tokenizer):
    """Merge PEFT adapter with the base model and save to a temporary directory."""
    print(f"Loading base model from {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print(f"Loading PEFT adapter from {checkpoint_path}...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch.float16
    )
    
    print("Merging adapter weights with base model...")
    merged_model = peft_model.merge_and_unload()
    
    temp_dir = tempfile.mkdtemp(prefix="merged_model_")
    print(f"Saving merged model to temporary directory: {temp_dir}")
    
    merged_model.save_pretrained(temp_dir, safe_serialization=True)
    tokenizer.save_pretrained(temp_dir)
    
    del base_model
    del peft_model
    del merged_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return temp_dir

def generate_solutions_vllm(llm, dataset):
    """
    Generate solutions using vLLM.
    UPDATED: Accepts full dataset to access 'test_list' for prompt building.
    """
    prompts = []
    for item in dataset:
        text = item.get("text", "")
        # Robustly handle test_list being missing or empty
        test_list = item.get("test_list", [])
        
        # Build prompt using the new logic
        prompts.append(build_mbpp_prompt(text, test_list))
    
    # Stop tokens to prevent the model from hallucinating new problems
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS, 
        temperature=TEMPERATURE, 
        stop=["Problem:", "<end_of_turn>", "User:", "Observation:", "\nclass", "\ndef test", "\nif __name__"]
    )
    
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for output in outputs:
        prompt = output.prompt
        solution = output.outputs[0].text.strip()
        results.append({"prompt": prompt, "solution": solution})
    
    return results

# ==========================================
# UTILITY FUNCTIONS: CODE EVALUATION
# ==========================================

def sanitize_code(code_str):
    """Cleans Markdown formatting and language identifiers."""
    if not code_str:
        return ""
    # Remove markdown code blocks
    code_str = re.sub(r'```python', '', code_str, flags=re.IGNORECASE)
    code_str = re.sub(r'```', '', code_str)
    # Remove "python" text if it appears at the very start
    if code_str.strip().lower().startswith("python"):
        code_str = code_str.strip()[6:]
    return code_str.strip()

def check_compilability(code):
    """Checks if the code is valid Python syntax."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}"
    except Exception as e:
        return False, str(e)

def create_test_runner_script(candidate_code, test_setup, test_list):
    """Creates a standalone Python script to run the tests."""
    tests_json = json.dumps(test_list)
    
    # Indent candidate code for the try-block
    indented_candidate = '\n'.join(['    ' + line for line in candidate_code.splitlines()])
    setup_code = test_setup if test_setup else '    pass'

    # The runner script that captures granular test results
    runner_code = f"""
import json
import sys
import re

# 1. Test Setup Code
try:
{setup_code}
except Exception as e:
    print(json.dumps({{"status": "setup_error", "error": str(e)}}))
    sys.exit(0)

# 2. Candidate Code
try:
{indented_candidate}
except Exception as e:
    print(json.dumps({{"status": "runtime_error", "error": str(e)}}))
    sys.exit(0)

# 3. Test Execution Logic
tests = {tests_json}
results = {{}}
all_passed = True
first_error = None

for i, test_case in enumerate(tests):
    try:
        exec(test_case)
        results[test_case] = "Passed"
    except AssertionError:
        results[test_case] = "Failed"
        all_passed = False
        if not first_error: first_error = "AssertionError"
    except Exception as e:
        results[test_case] = f"Error: {{str(e)}}"
        all_passed = False
        if not first_error: first_error = type(e).__name__

# 4. Output Results
print(json.dumps({{
    "status": "completed",
    "all_passed": all_passed,
    "test_details": results,
    "primary_error": first_error
}}))
"""
    return runner_code

def evaluate_single_code_sample(code, test_setup, test_list):
    """
    Runs the full evaluation pipeline for a single sample:
    Sanitize -> Compile -> Run Tests (Subprocess)
    """
    clean_code = sanitize_code(code)
    is_compilable, compile_err = check_compilability(clean_code)

    result_data = {
        "is_compilable": is_compilable,
        "compile_error": compile_err,
        "is_correct": False,
        "error_type": None,
        "test_results": {},
        "sanitized_code": clean_code
    }

    if not is_compilable:
        result_data["error_type"] = "SyntaxError"
        return result_data

    # Create temporary runner script
    runner_script = create_test_runner_script(clean_code, test_setup, test_list)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
        tmp_file.write(runner_script)

    try:
        process = subprocess.run(
            [sys.executable, tmp_file_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS
        )
        stdout = process.stdout.strip()
        
        if stdout and stdout.startswith("{") and stdout.endswith("}"):
            try:
                exec_data = json.loads(stdout)
                status = exec_data.get("status")
                
                if status == "completed":
                    result_data["is_correct"] = exec_data["all_passed"]
                    result_data["test_results"] = exec_data["test_details"]
                    result_data["error_type"] = exec_data["primary_error"]
                elif status == "setup_error":
                    result_data["error_type"] = "TestSetupError"
                    result_data["test_results"] = {"setup": exec_data["error"]}
                elif status == "runtime_error":
                    result_data["error_type"] = "RuntimeError"
                    result_data["test_results"] = {"runtime": exec_data["error"]}
            except json.JSONDecodeError:
                result_data["error_type"] = "JSONDecodeError"
    except subprocess.TimeoutExpired:
        result_data["error_type"] = "Timeout"
    except Exception as e:
        result_data["error_type"] = "HarnessError"
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

    return result_data

# ==========================================
# MAIN PROCESS FUNCTION
# ==========================================

def generate_and_evaluate_solutions(
    base_model=DEFAULT_BASE_MODEL, 
    dataset_path=DEFAULT_TEST_DATASET_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    output_suffix="baseline",
    tensor_parallel_size=1,
    eval_mode="baseline",
    checkpoint_dir=None,
    checkpoint=None,
    force=False,
    skip_generation=False,
    skip_evaluation=False
):
    os.makedirs(output_dir, exist_ok=True)
    
    generation_file = os.path.join(output_dir, f"code_generated_{output_suffix}.jsonl")
    output_file = os.path.join(output_dir, f"code_evaluated_{output_suffix}.jsonl")
    summary_file = os.path.join(output_dir, f"summary_{output_suffix}.json")
    
    if os.path.exists(output_file) and not force and not skip_evaluation:
        print(f"Combined output already exists at {output_file}. Skipping... (Use --force to override)")
        return True
    
    print(f"Loading dataset from {dataset_path}...")
    test_dataset = load_jsonl_dataset(dataset_path)
    print(f"Loaded dataset with {len(test_dataset)} examples")
    
    # ---------------- PHASE 1: GENERATION ----------------
    if not skip_generation:
        if not os.path.exists(generation_file) or force:
            print(f"Generating solutions using base model: {base_model}")
            start_time = time.time()
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
                
                # Model Loading Logic (PEFT vs Standard)
                if eval_mode == "peft" and checkpoint_dir and checkpoint:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                    print(f"Processing PEFT checkpoint: {checkpoint_path}")
                    merged_model_path = merge_and_save_peft_checkpoint(checkpoint_path, base_model, tokenizer)
                    print(f"Loading merged model with vLLM from {merged_model_path}...")
                    llm = LLM(model=merged_model_path, tensor_parallel_size=tensor_parallel_size)
                else:
                    model_path = os.path.join(checkpoint_dir, checkpoint) if eval_mode == "checkpoints" and checkpoint_dir and checkpoint else base_model
                    print(f"Loading model {model_path} with vLLM...")
                    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
                
                # UPDATED: We no longer extract just texts.
                # We pass the full dataset to generate_solutions_vllm so it can build dynamic prompts.
                print(f"Generating solutions...")
                generation_results = generate_solutions_vllm(llm, test_dataset)
                
                generation_time = time.time() - start_time
                print(f"Solutions generated. Time: {generation_time:.2f}s")
                
                # Combine original data with generations
                results = []
                for i, example in enumerate(test_dataset):
                    result = example.copy()
                    result["prompt_used"] = generation_results[i]["prompt"]
                    result["generated_code"] = generation_results[i]["solution"]
                    results.append(result)
                
                with open(generation_file, "w") as f:
                    for result in results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                cleanup_llm(llm)
                if eval_mode == "peft" and 'merged_model_path' in locals() and os.path.exists(merged_model_path):
                    shutil.rmtree(merged_model_path)
            
            except Exception as e:
                print(f"Error in generation process: {str(e)}")
                import traceback
                traceback.print_exc()
                if skip_evaluation: return False
                if not os.path.exists(generation_file): return False
        else:
            print(f"Generated solutions file already exists at {generation_file}. Loading...")

    if skip_evaluation:
        print("Skipping evaluation phase.")
        return True

    # ---------------- PHASE 2: EVALUATION ----------------
    print("\nStarting evaluation of generated code...")
    eval_start_time = time.time()
    
    if 'results' not in locals():
        results = load_jsonl_dataset(generation_file)
    
    correct_count = 0
    compilable_count = 0
    total = len(results)
    
    with open(output_file, "w") as outfile:
        for idx, data in enumerate(results):
            if idx % 50 == 0:
                print(f"Evaluating example {idx}/{total}...")
            
            gen_code = data.get("generated_code", "")
            test_setup = data.get("test_setup_code", "")
            test_list = data.get("test_list", [])

            # Run the robust evaluator
            eval_result = evaluate_single_code_sample(gen_code, test_setup, test_list)
            
            # Merge results back into data row
            data.update(eval_result)
            
            if data["is_compilable"]:
                compilable_count += 1
            if data["is_correct"]:
                correct_count += 1
                
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            
    eval_time = time.time() - eval_start_time
    accuracy = (correct_count / total) * 100 if total > 0 else 0.0
    compilability = (compilable_count / total) * 100 if total > 0 else 0.0
    
    # Summary
    summary = {
        "model": base_model,
        "dataset_path": dataset_path,
        "output_suffix": output_suffix,
        "total_examples": total,
        "correct": correct_count,
        "compilable": compilable_count,
        "pass_at_1": accuracy,
        "compilability_rate": compilability,
        "evaluation_time": eval_time
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation Results for {output_suffix}")
    print(f"Pass@1: {accuracy:.2f}%")
    print(f"Compilability: {compilability:.2f}%")
    print(f"Time: {eval_time:.2f}s")
    print(f"Saved to {output_file}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and evaluate code solutions.")
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_TEST_DATASET_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_suffix", type=str, default="baseline")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    
    parser.add_argument("--eval_mode", type=str, choices=["base", "checkpoints", "peft"], default="base")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, nargs="+", default=[])
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    
    args = parser.parse_args()
    
    # Checkpoint iteration logic (same as your math script)
    if args.eval_mode in ["checkpoints", "peft"] and args.checkpoint_dir:
        if not args.checkpoints:
            patterns = ["checkpoint-*", "adapter_model.bin", "adapter_model.safetensors"]
            found = []
            for pat in patterns:
                found.extend(glob.glob(os.path.join(args.checkpoint_dir, pat)))
            # If finding files directly, get parent dir name if it's adapter file, else basename
            args.checkpoints = list(set([os.path.basename(os.path.dirname(p)) if 'adapter' in p else os.path.basename(p) for p in found]))
            # Fallback for standard checkpoint directories
            if not args.checkpoints:
                 args.checkpoints = [d for d in os.listdir(args.checkpoint_dir) if d.startswith('checkpoint')]

        for checkpoint in args.checkpoints:
            suffix = f"{args.output_suffix}_{checkpoint}"
            generate_and_evaluate_solutions(
                base_model=args.base_model if args.eval_mode == "peft" else args.model,
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                output_suffix=suffix,
                tensor_parallel_size=args.tensor_parallel_size,
                eval_mode=args.eval_mode,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint=checkpoint,
                force=args.force,
                skip_generation=args.skip_generation,
                skip_evaluation=args.skip_evaluation
            )
    else:
        generate_and_evaluate_solutions(
            base_model=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            output_suffix=args.output_suffix,
            tensor_parallel_size=args.tensor_parallel_size,
            force=args.force,
            skip_generation=args.skip_generation,
            skip_evaluation=args.skip_evaluation
        )













# python /u1/a23chand/synthetic_questions/00_data_train_final/code_mbpp/mbpp_eval_checkpoints/gen_plus_eval_mbpp_correct.py --model google/gemma-2-2b  --base_model google/gemma-2-2b  --eval_mode checkpoints --dataset /u1/a23chand/synthetic_questions/00_data_train_final/code_mbpp/mbpp_test_200.jsonl --checkpoint_dir /u1/a23chand/synthetic_questions/model/MBPP_Correct/G2B_on_G27B_IT_W    --output_dir /u1/a23chand/synthetic_questions/00_data_train_final/code_mbpp/mbpp_eval_FINAL/G2B_W_Balanced --output_suffix W



#python /u1/a23chand/synthetic_questions/00_data_train_final/code_mbpp/mbpp_eval/gen_plus_eval_mbpp.py   --model google/gemma-2-2b --eval_mode checkpoints  --dataset /u1/a23chand/synthetic_questions/00_data_train_final/code_mbpp/mbpp_test_200.jsonl   --output_dir /u1/a23chand/synthetic_questions/00_data_train_final/code_mbpp/mbpp_eval/G2B_H --checkpoint_dir /u1/a23chand/synthetic_questions/model/MBPP/G2B_on_Human --output_suffix H
