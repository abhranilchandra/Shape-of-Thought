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
import multiprocessing
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams

#import os
#os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

#import multiprocessing
#multiprocessing.set_start_method('spawn', force=True)
# Default path settings pass arguments to change
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DEFAULT_OUTPUT_DIR = ""
DEFAULT_TEST_DATASET_PATH = ""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# ------------ PROMPT TEMPLATES ------------
# Notice the doubled curly braces {{ }} for escaping them in the template

# Zero-shot prompt template
ZERO_SHOT_TEMPLATE = """You are a math expert. I am going to give you a Problem that you need to solve. When you respond, respond with the Solution, thinking step by step. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\nProblem:\nUsing the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Do not use any other operations or numbers.\n\nSolution:"""



FEW_SHOT_TEMPLATE = """You are a math expert. I am going to give you a Problem that you need to solve. When you respond, respond with the Solution, thinking step by step. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.

Problem:
Using the numbers [38, 98, 56, 14], create an equation that equals 91. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Do not use any other operations or numbers.

Solution:
I am looking for a combination of numbers and operations that results in 91.
I can try to combine numbers using addition first. Let's try adding 38 and 14.
38 + 14 = 52.
Now I have the numbers 52, 98, and 56 left to use.
I need to get to 91. Let's see if multiplication or division can help.
Let's try multiplying 52 by 98.
52 * 98 is a large number. Let's try dividing by 56.
So, (52 * 98) / 56.
I can simplify this calculation. 98 and 56 are both divisible by 14.
98 / 14 = 7.
56 / 14 = 4.
So, the expression becomes 52 * (7 / 4).
I can rewrite this as (52 / 4) * 7.
52 / 4 = 13.
Now I just need to multiply 13 by 7.
13 * 7 = 91.
This gives the target number. The full equation is ((38 + 14) * 98) / 56.
<answer> ((38 + 14) * 98) / 56 </answer>


Problem:
Using the numbers [23, 63, 79, 51], create an equation that equals 68. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Do not use any other operations or numbers.

Solution:
I am looking for a combination of numbers and operations that results in 68.
I'll start with the largest number, 79.
To get to 68, I need to subtract 11.
Can I make 11 from 23, 63, and 51?
63 - 51 = 12. This is close to 11.
Let's try 79 - (63 - 51) = 79 - 12 = 67. This is very close to 68, but not exactly.
Let's try another combination.
79 + 51 = 130.
63 + 23 = 86.
130 - 86 = 44.
Let's try another path.
63 - 23 = 40.
79 - 51 = 28.
40 + 28 = 68.
This works! I have found a solution.
The steps are: subtract 23 from 63 to get 40. Subtract 51 from 79 to get 28. Add the results together.
<answer> (63 - 23) + (79 - 51) </answer>


Problem:
Using the numbers [16, 17, 58], create an equation that equals 91. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Do not use any other operations or numbers.

Solution:
I need to reach the target of 91 using the numbers 16, 17, and 58.
Since there are only three numbers, I'll try adding them up first.
58 + 17 = 75.
Now, I need to incorporate the last number, 16.
75 + 16 = 91.
This is the target number.
So, the solution is to add all the numbers together.
<answer> 58 + 17 + 16 </answer>


Problem:
Using the numbers [2, 28, 78], create an equation that equals 11. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Do not use any other operations or numbers.
Solution:
The target is 11. The numbers are 2, 28, and 78.
The numbers are quite spread out, so simple addition or subtraction of all of them at once is unlikely to work.
Let's see if there is a division or multiplication that simplifies the problem.
78 is an even number, so it's divisible by 2.
78 / 2 = 39.
Now I have the number 39, and the remaining number is 28.
I need to reach the target of 11.
Let's see the difference between 39 and 28.
39 - 28 = 11.
This is the target number.
So the equation is (78 / 2) - 28.
<answer> (78 / 2) - 28 </answer>


Problem:
Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Do not use any other operations or numbers.

Solution:"""

# For countdown task, we only use zero-shot template

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    #if "Assistant:" in solution_str:
    #    solution_str = solution_str.split("Assistant:", 1)[1]
    #elif "<|im_start|>assistant" in solution_str:
    #    solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    #else:
    #    return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, rule_score=0.5, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        rule_score: the score for following rules but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    equation = extract_solution(solution_str=solution_str)
    
    if equation is None:
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            return score
        else:
            return rule_score
    except:
        return format_score

# ------------ GENERATION PARAMETERS ------------
MAX_TOKENS = 4096
TEMPERATURE = 0.0  # Greedy decoding

# ------------ UTILITY FUNCTIONS ------------
def load_jsonl_dataset(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def cleanup_llm(llm):
    """Clean up LLM resources to prevent memory leaks"""
    if hasattr(llm, "shutdown") and callable(llm.shutdown):
        try:
            llm.shutdown()
        except Exception as e:
            print(f"Error during llm.shutdown(): {e}")
    elif hasattr(llm, "close") and callable(llm.close):
        try:
            llm.close()
        except Exception as e:
            print(f"Error during llm.close(): {e}")
    del llm
    torch.cuda.empty_cache()
    gc.collect()

def merge_and_save_peft_checkpoint(checkpoint_path, base_model_id, tokenizer):
    """
    Merge PEFT adapter with the base model and save to a temporary directory.
    Returns the path to the merged model.
    """
    print(f"Loading base model from {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,  # Use half precision to save memory
        trust_remote_code=True,
        device_map="auto"  # Let HF decide the optimal device mapping
    )
    
    print(f"Loading PEFT adapter from {checkpoint_path}...")
    peft_model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch.float16
    )
    
    print("Merging adapter weights with base model...")
    merged_model = peft_model.merge_and_unload()
    
    # Create a temporary directory to save the merged model
    temp_dir = tempfile.mkdtemp(prefix="merged_model_")
    print(f"Saving merged model to temporary directory: {temp_dir}")
    
    # Save the merged model
    merged_model.save_pretrained(temp_dir, safe_serialization=True)
    tokenizer.save_pretrained(temp_dir)
    
    # Clean up GPU memory
    del base_model
    del peft_model
    del merged_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return temp_dir

def generate_solutions_vllm(llm, test_examples, shot_mode="zero_shot"):
    """Generate solutions using vLLM for countdown problems"""
    if shot_mode == "zero_shot":
        template = ZERO_SHOT_TEMPLATE
    elif shot_mode == "few_shot":
        template = FEW_SHOT_TEMPLATE
    else:
        raise ValueError(f"Invalid shot mode: {shot_mode}")
    
    prompts = []
    for example in test_examples:
        # Format the prompt with numbers and target from the example
        prompt = template.format(nums=example['nums'], target=example['target'])
        prompts.append(prompt)
    
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=42, stop=["Problem:"])
    outputs = llm.generate(prompts, sampling_params)
    solutions = [output.outputs[0].text.strip() for output in outputs]
    return solutions


# ------------ PARALLEL VERIFICATION FUNCTIONS ------------
def verify_single_problem(args):
    """Worker function for parallel verification using compute_score"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Verification timed out")
    
    idx, ground_truth, pred_text, timeout_seconds = args
    try:
        # Set up timeout using signal (simpler and faster)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            # Use compute_score function with the ground truth and prediction
            score = compute_score(pred_text, ground_truth)
            signal.alarm(0)  # Cancel alarm
            
            # Convert score to boolean (only score=1 is considered correct)
            is_correct = (score >= 1.0)
            return (idx, 'success', is_correct, score)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
    except TimeoutError as e:
        return (idx, 'timeout', False, str(e))
    except Exception as e:
        return (idx, 'error', False, str(e))

def parallel_verification(results, num_processes=None, timeout_seconds=60):
    """Perform verification in parallel using multiprocessing with compute_score"""
    if num_processes is None:
        num_processes = min(multiprocessing.cpu_count(), 32)  # Use up to 32 CPUs
    
    print(f"Starting parallel verification with {num_processes} processes...")
    
    # Prepare arguments for parallel processing
    verification_args = []
    for idx, data in enumerate(results):
        pred_text = data.get("generated_solution", "")
        
        # Create ground truth dictionary with target and numbers
        ground_truth = {
            'target': data.get('target'),
            'numbers': data.get('nums')
        }
        
        # Only process if we have the required data
        if pred_text and ground_truth['target'] is not None and ground_truth['numbers']:
            verification_args.append((idx, ground_truth, pred_text, timeout_seconds))
    
    print(f"Will verify {len(verification_args)} problems (skipping {len(results) - len(verification_args)} with missing data)")
    
    # Run verification in parallel with progress tracking
    verification_results = []
    start_time = time.time()
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        print("Starting verification with progress tracking...")
        
        # Use imap for progress tracking instead of map
        for i, result in enumerate(pool.imap(verify_single_problem, verification_args)):
            verification_results.append(result)
            
            # Progress update every 50 problems
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(verification_args) - i - 1) / rate if rate > 0 else 0
                print(f"Progress: {i+1}/{len(verification_args)} ({100*(i+1)/len(verification_args):.1f}%) "
                      f"- Rate: {rate:.2f} problems/sec - ETA: {eta/60:.1f} minutes")
    
    total_time = time.time() - start_time
    print(f"Verification completed in {total_time:.2f} seconds")
    return verification_results

def save_generation_results(results, output_file):
    """Save generation results to file"""
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Generation results saved to {output_file}")

def save_verification_results(verification_results, verification_file):
    """Save verification results to separate file"""
    with open(verification_file, "w") as f:
        for result in verification_results:
            f.write(json.dumps({
                'idx': result[0],
                'status': result[1],
                'correct': result[2],
                'score': result[3] if len(result) > 3 and result[1] == 'success' else None,
                'error_msg': result[3] if len(result) > 3 and result[1] != 'success' else None
            }, ensure_ascii=False) + "\n")
    print(f"Verification results saved to {verification_file}")

# ------------ MAIN PROCESS FUNCTION ------------
def generate_and_evaluate_solutions(
    base_model=DEFAULT_BASE_MODEL, 
    dataset_path=DEFAULT_TEST_DATASET_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    output_suffix="baseline",
    tensor_parallel_size=1,
    eval_mode="baseline",
    checkpoint_dir=None,
    checkpoint=None,
    shot_mode="few_shot",
    force=False
):
    """Generate solutions and evaluate them in one function"""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Multi-stage file structure
    generation_file = os.path.join(output_dir, f"generation_{output_suffix}.jsonl")
    verification_file = os.path.join(output_dir, f"verification_{output_suffix}.jsonl")
    output_file = os.path.join(output_dir, f"math128_combined_{output_suffix}.jsonl")
    log_file = os.path.join(output_dir, f"errorindices_{output_suffix}.log")
    summary_file = os.path.join(output_dir, f"summary_{output_suffix}.json")
    
    # Check if summary JSON file already exists and force flag is not set
    # This is a better indicator that the full evaluation is complete
    if os.path.exists(summary_file) and not force:
        print(f"Evaluation already completed. Summary file exists at {summary_file}. Skipping...")
        return True
    
    print(f"Loading dataset with test problems from {dataset_path}...")
    test_dataset = load_jsonl_dataset(dataset_path)
    print(f"Loaded dataset with {len(test_dataset)} examples")
    
    # Check if generation is already complete
    generation_complete = False
    if os.path.exists(generation_file) and not force:
        try:
            # Load existing results to check if generation is complete
            existing_results = load_jsonl_dataset(generation_file)
            if len(existing_results) == len(test_dataset):
                # Check if all have generated solutions
                all_have_solutions = all('generated_solution' in result for result in existing_results)
                if all_have_solutions:
                    print(f"Generation already complete with {len(existing_results)} solutions. Proceeding to verification only...")
                    generation_complete = True
                    results = existing_results
                else:
                    print(f"Found {len(existing_results)} results but some missing generated solutions. Will regenerate...")
            else:
                print(f"Found {len(existing_results)} results but expected {len(test_dataset)}. Will regenerate...")
        except Exception as e:
            print(f"Error reading existing results: {e}. Will regenerate...")
    
    if not generation_complete:
        # Step 1: Generate solutions
        print(f"Generating solutions using base model: {base_model}")
        start_time = time.time()
    
        try:
            # Load tokenizer
            print("Loading tokenizer from base model...")
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
            # Load model based on evaluation mode
            if eval_mode == "peft" and checkpoint_dir and checkpoint:
                # For PEFT mode, merge the checkpoint with base model and save to temp dir
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                print(f"Processing PEFT checkpoint: {checkpoint_path}")
            
                # Merge PEFT adapter with base model and save to temporary directory
                merged_model_path = merge_and_save_peft_checkpoint(checkpoint_path, base_model, tokenizer)
                
                try:
                    # Load the merged model with vLLM
                    print(f"Loading merged model with vLLM from {merged_model_path}...")
                    llm = LLM(
                        model=merged_model_path,
                        tensor_parallel_size=tensor_parallel_size,
                        enforce_eager=True,
                        disable_custom_all_reduce=True
                    )
                except Exception as e:
                    print(f"Error loading merged model with vLLM: {e}")
                    # Fallback to loading with PEFT adapter directly
                    print("Falling back to using PEFT adapter with vLLM...")
                    llm = LLM(
                        model=base_model,
                        tensor_parallel_size=tensor_parallel_size,
                        peft_model_path=checkpoint_path,
                        enforce_eager=True,
                        disable_custom_all_reduce=True
                    )
            else:
                # For baseline mode, load the base model
                print(f"Loading base model {base_model} with vLLM...")
                llm = LLM(
                    model=base_model,
                    tensor_parallel_size=tensor_parallel_size,
                    enforce_eager=True,
                    disable_custom_all_reduce=True
                )
        
            # Generate solutions
            print(f"Generating solutions using {shot_mode} prompting...")
            solutions = generate_solutions_vllm(llm, test_dataset, shot_mode=shot_mode)
            
            # Save results to generation file
            results = []
            for i, example in enumerate(test_dataset):
                result = example.copy()
                result["generated_solution"] = solutions[i]
                results.append(result)
            
            # Save generation results immediately
            save_generation_results(results, generation_file)
            
            generation_time = time.time() - start_time
            print(f"Solutions generated. Generation time: {generation_time:.2f} seconds")
            
            # Clean up model to free memory
            cleanup_llm(llm)
            time.sleep(5)  # Ensure cleanup is complete before proceeding
            
            # Clean up temporary merged model directory if it exists
            if eval_mode == "peft" and 'merged_model_path' in locals() and os.path.exists(merged_model_path):
                print(f"Cleaning up temporary directory: {merged_model_path}")
                try:
                    shutil.rmtree(merged_model_path)
                except Exception as e:
                    print(f"Error cleaning up temporary directory: {e}")
        
        except Exception as e:
            print(f"Error in generation process: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    else:
        # Generation already complete, set default values
        generation_time = 0.0
        start_time = time.time()
        
    # Step 2: Load generation results for verification
    if generation_complete:
        # Load existing generation results
        results = load_jsonl_dataset(generation_file)
        print(f"Loaded {len(results)} generation results from {generation_file}")
    else:
        # Results already available from generation stage above
        print(f"Using freshly generated {len(results)} results")
    
    # Step 3: Check if verification is already complete
    verification_complete = False
    if os.path.exists(verification_file) and not force:
        print(f"Verification file exists. Checking if verification is complete...")
        try:
            verification_results = []
            with open(verification_file, "r") as f:
                for line in f:
                    verification_results.append(json.loads(line))
            
            # Check if we have verification results for most problems
            # Use a simple heuristic: if we have verification for >90% of total problems, assume complete
            expected_min = int(0.9 * len(results))  # Expect at least 90% (some may fail parsing)
            
            if len(verification_results) >= expected_min:
                print(f"Verification already complete with {len(verification_results)} results (>90% of {len(results)} total). Proceeding to combine results...")
                verification_complete = True
            else:
                print(f"Found {len(verification_results)} verification results but expected at least {expected_min}. Re-running verification...")
        except Exception as e:
            print(f"Error reading verification results: {e}. Re-running verification...")
    
    # Step 4: Run parallel verification if needed
    eval_start_time = time.time()
    if not verification_complete:
        print("\nStarting parallel verification of generated solutions...")
        # Run parallel verification with 60-second timeout
        verification_results = parallel_verification(results, num_processes=None, timeout_seconds=60)
        
        # Save verification results
        save_verification_results(verification_results, verification_file)
        print(f"Parallel verification completed. Results saved to {verification_file}")
    
    # Step 5: Combine results and generate summary
    eval_time = time.time() - eval_start_time
    total_time = time.time() - start_time
    
    print("\nCombining results and generating summary...")
    
    # Load generation results
    print("Loading generation results...")
    generation_results = load_jsonl_dataset(generation_file)
    print(f"Loaded {len(generation_results)} generation results")
    
    # Load verification results
    print("Loading verification results...")
    verification_dict = {}
    if os.path.exists(verification_file):
        with open(verification_file, "r") as f:
            for line_num, line in enumerate(f):
                if line_num % 1000 == 0:
                    print(f"Loading verification result {line_num}...")
                ver_result = json.loads(line)
                verification_dict[ver_result['idx']] = ver_result
    print(f"Loaded {len(verification_dict)} verification results")
    
    # Initialize counters
    correct = 0  # score >= 1.0
    rule_failures = 0  # score = 0.5 (follows rules but wrong answer)
    format_failures = 0  # score = 0.1 (correct format but rule violations or wrong answer)
    parsing_failures = 0  # score = 0 (cannot parse or other failures)
    timeout_failures = 0
    error_indices = []
    timeout_indices = []
    
    # Combine results
    print("Combining results...")
    combined_results = []
    for idx, data in enumerate(generation_results):
        if idx % 1000 == 0:
            print(f"Processing result {idx}/{len(generation_results)}...")
        
        # Add evaluation results
        data["correct"] = None
        data["score"] = None
        
        if idx in verification_dict:
            ver_result = verification_dict[idx]
            if ver_result['status'] == 'success':
                # Categorize based on actual scores from compute_score function
                # The compute_score function already handles all parsing/rule logic internally
                score = ver_result.get('score', 0)
                data["score"] = score
                data["correct"] = (score >= 1.0)
                
                if score >= 1.0:
                    correct += 1
                elif abs(score - 0.5) < 1e-6:  # score = 0.5 (rule_score)
                    rule_failures += 1
                elif abs(score - 0.1) < 1e-6:  # score = 0.1 (format_score)
                    format_failures += 1
                else:  # score = 0 (parsing failures handled by compute_score)
                    parsing_failures += 1
            elif ver_result['status'] == 'timeout':
                timeout_failures += 1
                timeout_indices.append(idx)
                data["correct"] = False
                data["timeout_error"] = True
                data["score"] = 0
            else:  # 'error' or 'parse_error' - verification process failed before compute_score
                parsing_failures += 1
                error_indices.append(idx)
                data["score"] = 0
        else:
            # No verification result - likely missing data or verification process failed
            parsing_failures += 1
            error_indices.append(idx)
            data["score"] = 0
        
        combined_results.append(data)
    
    # Save combined results
    with open(output_file, "w") as f:
        for result in combined_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Calculate accuracy - only for successfully evaluated problems (exclude parsing failures)
    total_evaluated = correct + rule_failures + format_failures
    accuracy = (correct / total_evaluated) * 100 if total_evaluated > 0 else 0.0
    
    # Total attempted includes timeouts but excludes parsing failures
    total_attempted = total_evaluated + timeout_failures
    
    # Write log file
    with open(log_file, "w") as logf:
        logf.write("Indices with parse or verify failures:\n")
        for idx in error_indices:
            logf.write(f"{idx}\n")
        
        logf.write("\nIndices with timeout failures:\n")
        for idx in timeout_indices:
            logf.write(f"{idx}\n")
        
        logf.write("\n=== Final Evaluation Summary (Countdown Task) ===\n")
        logf.write(f"Total Examples: {len(generation_results)}\n")
        logf.write(f"Total Successfully Evaluated: {total_evaluated}\n")
        logf.write(f"Total Attempted (includes timeouts): {total_attempted}\n")
        logf.write(f"\n--- Score Breakdown ---\n")
        logf.write(f"Correct (score >= 1.0): {correct}\n")
        logf.write(f"Rule Failures (score = 0.5): {rule_failures}\n")
        logf.write(f"Format Failures (score = 0.1): {format_failures}\n")
        logf.write(f"Parsing Failures (score = 0): {parsing_failures}\n")
        logf.write(f"Timeout Failures: {timeout_failures}\n")
        logf.write(f"\nAccuracy (correct/evaluated): {accuracy:.2f}%\n")
        logf.write(f"\nGeneration Time: {generation_time:.2f} seconds\n")
        logf.write(f"Evaluation Time: {eval_time:.2f} seconds\n")
        logf.write(f"Total Time: {total_time:.2f} seconds\n")
    
    # Create summary JSON
    summary = {
        "model": base_model,
        "dataset_path": dataset_path,
        "output_suffix": output_suffix,
        "output_file": output_file,
        "log_file": log_file,
        "total_examples": len(generation_results),
        "total_evaluated": total_evaluated,
        "total_attempted": total_attempted,
        "correct": correct,
        "rule_failures": rule_failures,
        "format_failures": format_failures,
        "parsing_failures": parsing_failures,
        "timeout_failures": timeout_failures,
        "accuracy": accuracy,
        "generation_time": generation_time,
        "evaluation_time": eval_time,
        "total_time": total_time
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\nCountdown Task Evaluation Results for {output_suffix}")
    print(f"Total Examples: {len(generation_results)}")
    print(f"Total Successfully Evaluated: {total_evaluated}")
    print(f"Total Attempted (includes timeouts): {total_attempted}")
    print(f"\n--- Score Breakdown ---")
    print(f"Correct (score >= 1.0): {correct}")
    print(f"Rule Failures (score = 0.5): {rule_failures}")
    print(f"Format Failures (score = 0.1): {format_failures}")
    print(f"Parsing Failures (score = 0): {parsing_failures}")
    print(f"Timeout Failures: {timeout_failures}")
    print(f"\nAccuracy (correct/evaluated): {accuracy:.2f}%")
    print(f"Generation Time: {generation_time:.2f} seconds")
    print(f"Evaluation Time: {eval_time:.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Combined results saved to {output_file}")
    print(f"Log saved to {log_file}")
    print(f"Summary saved to {summary_file}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and evaluate math solutions using a language model.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model identifier (default: {DEFAULT_BASE_MODEL})"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_TEST_DATASET_PATH,
        help=f"Path to the test dataset (default: {DEFAULT_TEST_DATASET_PATH})"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="baseline",
        help="Suffix for the output file names (default: 'baseline')"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (default: 1)"
    )
    # Additional arguments for PEFT evaluation
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["base", "checkpoints", "peft"],
        default="base",
        help="Evaluation mode: base, checkpoints, or peft (Parameter-Efficient Fine-Tuning)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing PEFT model checkpoints"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=[],
        help="List of checkpoint names to evaluate (e.g., checkpoint-1 checkpoint-2)"
    )
    parser.add_argument(
        "--shot_mode",
        type=str,
        choices=["zero_shot", "few_shot"],
        default="zero_shot",
        help="Prompting mode (only zero-shot supported for countdown task)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model for PEFT (if different from --model)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force evaluation even if output files already exist"
    )
    args = parser.parse_args()
    
    # Print the parameters being used
    print("\n=== Parameters ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Output Suffix: {args.output_suffix}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"Evaluation Mode: {args.eval_mode}")
    print(f"Shot Mode: {args.shot_mode}")
    print(f"Force Re-evaluation: {args.force}")
    if args.eval_mode in ["checkpoints", "peft"]:
        print(f"Checkpoint Directory: {args.checkpoint_dir}")
        print(f"Checkpoints: {', '.join(args.checkpoints) if args.checkpoints else 'All'}")
        print(f"Base Model (for PEFT): {args.base_model}")
    print("=================\n")
    
    # For countdown task, we only use zero_shot
    internal_shot_mode = args.shot_mode
    
    # Handle different evaluation modes
    if args.eval_mode == "peft" and args.checkpoint_dir:
        print(f"\nRunning in PEFT evaluation mode")
        
        # Get all checkpoints if not explicitly specified
        if not args.checkpoints:
            print("No specific checkpoints provided, scanning directory...")
            checkpoint_patterns = ["checkpoint-*", "adapter_model", "adapter_model.bin"]
            args.checkpoints = []
            for pattern in checkpoint_patterns:
                found = glob.glob(os.path.join(args.checkpoint_dir, pattern))
                args.checkpoints.extend([os.path.basename(p) for p in found])
            
            if not args.checkpoints:
                print("No checkpoints found. Please check the checkpoint directory.")
                exit(1)
            
            print(f"Found checkpoints: {', '.join(args.checkpoints)}")
        
        # Evaluate each checkpoint
        all_success = True
        for checkpoint in args.checkpoints:
            checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint)
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint directory {checkpoint_path} does not exist. Skipping.")
                continue
                
            print(f"\nEvaluating checkpoint: {checkpoint}")
            # Create a unique suffix for this checkpoint
            suffix = f"{args.output_suffix}_{args.shot_mode}_{checkpoint.replace('-', '_')}"
            
            success = generate_and_evaluate_solutions(
                base_model=args.base_model,  # Use the specified base model
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                output_suffix=suffix,
                tensor_parallel_size=args.tensor_parallel_size,
                eval_mode=args.eval_mode,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint=checkpoint,
                shot_mode=internal_shot_mode,
                force=args.force
            )
            
            all_success = all_success and success
            
        if all_success:
            print("\nAll PEFT checkpoint evaluations completed successfully.")
        else:
            print("\nSome PEFT checkpoint evaluations encountered errors.")
    
    elif args.eval_mode == "checkpoints" and args.checkpoint_dir:
        # Similar to PEFT but for full checkpoints
        print(f"\nRunning in regular checkpoints evaluation mode")
        
        # Get all checkpoints if not explicitly specified
        if not args.checkpoints:
            print("No specific checkpoints provided, scanning directory...")
            checkpoint_patterns = ["checkpoint-*"]
            args.checkpoints = []
            for pattern in checkpoint_patterns:
                found = glob.glob(os.path.join(args.checkpoint_dir, pattern))
                args.checkpoints.extend([os.path.basename(p) for p in found])
            
            if not args.checkpoints:
                print("No checkpoints found. Please check the checkpoint directory.")
                exit(1)
            
            print(f"Found checkpoints: {', '.join(args.checkpoints)}")
        
        # Process each checkpoint
        all_success = True
        for checkpoint in args.checkpoints:
            # For regular checkpoints, the checkpoint itself is the model path
            checkpoint_model_path = os.path.join(args.checkpoint_dir, checkpoint)
            
            print(f"\nEvaluating checkpoint: {checkpoint}")
            # Create a unique suffix for this checkpoint
            suffix = f"{args.output_suffix}_{args.shot_mode}_{checkpoint.replace('-', '_')}"
            
            # For regular checkpoints, we treat the checkpoint as the model
            success = generate_and_evaluate_solutions(
                base_model=checkpoint_model_path,
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                output_suffix=suffix,
                tensor_parallel_size=args.tensor_parallel_size,
                eval_mode="base",  # Use base mode since we're directly loading the model
                shot_mode=internal_shot_mode,
                force=args.force
            )
            
            all_success = all_success and success
        
        if all_success:
            print("\nAll checkpoint evaluations completed successfully.")
        else:
            print("\nSome checkpoint evaluations encountered errors.")
    
    else:
        # Original base model evaluation
        suffix = f"{args.output_suffix}_{args.shot_mode}"
        success = generate_and_evaluate_solutions(
            base_model=args.model,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            output_suffix=suffix,
            tensor_parallel_size=args.tensor_parallel_size,
            shot_mode=internal_shot_mode,
            force=args.force
        )
    
    if success:
        print("\nProcess completed successfully.")
    else:
        print("\nProcess encountered errors.")
