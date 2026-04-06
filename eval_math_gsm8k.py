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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams
from math_verify import parse, verify





# Default path settings pass arguments to change
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B" 
DEFAULT_OUTPUT_DIR = ""




# ------------ PROMPT TEMPLATES ------------
# Notice the doubled curly braces {{ }} for escaping them in the template

# Zero-shot prompt template
ZERO_SHOT_TEMPLATE = """You are a math expert. I am going to give you a math Problem that you need to solve. When you respond, respond only with the Solution, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct.\"\n\nProblem:\n{problem}\n\nSolution:"""


Problem:# Few-shot (4-shot) prompt template
FEW_SHOT_TEMPLATE = """You are a math expert. I am going to give you a math Problem that you need to solve. When you respond, respond only with the Solution, thinking step by step. At the end of the Solution, when you give your final answer, write it in the form "Final Answer: The final answer is $answer$. I hope it is correct.\"

Problem:
Find the domain of the expression $\\frac{{\\sqrt{{x-2}}}}{{\\sqrt{{5-x}}}}$.
Solution:
The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{{[2,5)}}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
If $\\det \\mathbf{{A}} = 2$ and $\\det \\mathbf{{B}} = 12,$ then find $\\det (\\mathbf{{A}} \\mathbf{{B}}).$
Solution:
We have that $\\det (\\mathbf{{A}} \\mathbf{{B}}) = (\\det \\mathbf{{A}})(\\det \\mathbf{{B}}) = (2)(12) = \\boxed{{24}}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$: \\begin{{align*}}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{{16}}
\\end{{align*}}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations \\begin{{align*}} 6x-4y&=a,\\\\ 6y-9x &=b. \\end{{align*}}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{{a}}{{b}},$ assuming $b$ is nonzero.
Solution:
If we multiply the first equation by $-\\frac{{3}}{{2}}$, we obtain
$6y-9x=-\\frac{{3}}{{2}}a.$Since we also know that $6y-9x=b$, we have
$-\\frac{{3}}{{2}}a=b\\Rightarrow\\frac{{a}}{{b}}=\\boxed{{-\\frac{{2}}{{3}}}}.$
Final Answer: The final answer is $-\\frac{{2}}{{3}}$. I hope it is correct.
\n\nProblem:\n{problem}\n\nSolution:"""


# Default to few-shot template (will be updated based on args)
PROMPT_TEMPLATE = ZERO_SHOT_TEMPLATE

# ------------ GENERATION PARAMETERS ------------
MAX_TOKENS = 2048
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



def generate_solutions_vllm(llm, problems, shot_mode="few_shot"):
    """Generate solutions using vLLM"""
    # Choose the appropriate template based on shot mode
    template = ZERO_SHOT_TEMPLATE if shot_mode == "zero_shot" else FEW_SHOT_TEMPLATE
    
    prompts = [template.format(problem=p) for p in problems]
    sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=42, stop=["Problem:"])
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract both prompts and solutions
    results = []
    for output in outputs:
        prompt = output.prompt
        solution = output.outputs[0].text.strip()
        results.append({"prompt": prompt, "solution": solution})
    
    return results










def extract_final_answer(text):
    try:
        parsed = parse(text)
        if parsed:
            # Filter overly long numeric strings (e.g., hallucinated huge numbers)
            for p in parsed:
                if isinstance(p, str) and len(p) > 5000:
                    return None  # Treat as unparseable
            return parsed
        else:
            return None
    except Exception:
        return None

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
    
    output_file = os.path.join(output_dir, f"math128_combined_{output_suffix}.jsonl")
    log_file = os.path.join(output_dir, f"errorindices_{output_suffix}.log")
    summary_file = os.path.join(output_dir, f"summary_{output_suffix}.json")
    
    # Check if output file already exists and force flag is not set
    if os.path.exists(output_file) and not force:
        print(f"Combined output already exists at {output_file}. Skipping...")
        return True
    
    print(f"Loading dataset with test problems from {dataset_path}...")
    test_dataset = load_jsonl_dataset(dataset_path)
    print(f"Loaded dataset with {len(test_dataset)} examples")
    
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
                    tensor_parallel_size=tensor_parallel_size
                )
            except Exception as e:
                print(f"Error loading merged model with vLLM: {e}")
                # Fallback to loading with PEFT adapter directly
                print("Falling back to using PEFT adapter with vLLM...")
                llm = LLM(
                    model=base_model,
                    tensor_parallel_size=tensor_parallel_size,
                    peft_model_path=checkpoint_path
                )
        else:
            # For baseline mode, load the base model
            print(f"Loading base model {base_model} with vLLM...")
            llm = LLM(
                model=base_model,
                tensor_parallel_size=tensor_parallel_size
            )
        
        # Extract problems
        problems = [example["problem"] for example in test_dataset]
        
        # Generate solutions
        print(f"Generating solutions using {shot_mode} prompting...")
        generation_results = generate_solutions_vllm(llm, problems, shot_mode=shot_mode)
        
        # Save results directly to a single combined file
        results = []
        for i, example in enumerate(test_dataset):
            result = example.copy()
            result["prompt"] = generation_results[i]["prompt"]
            result["generated_solution"] = generation_results[i]["solution"]
            results.append(result)
        
        generation_time = time.time() - start_time
        print(f"Solutions generated. Generation time: {generation_time:.2f} seconds")
        
        # Clean up model to free memory
        cleanup_llm(llm)
        
        # Clean up temporary merged model directory if it exists
        if eval_mode == "peft" and 'merged_model_path' in locals() and os.path.exists(merged_model_path):
            print(f"Cleaning up temporary directory: {merged_model_path}")
            try:
                shutil.rmtree(merged_model_path)
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")
        
        # Step 2: Evaluate the solutions
        print("\nStarting evaluation of generated solutions...")
        eval_start_time = time.time()
        
        # Evaluation metrics
        correct = 0
        total = 0
        parsing_failures = 0
        verification_failures = 0
        error_indices = []
        
        # Open files for writing final output and logs
        with open(output_file, "w") as outfile, \
             open(log_file, "w") as logf:
            
            for idx, data in enumerate(results):
                gt_text = data.get("solution", "")
                pred_text = data.get("generated_solution", "")
                
                gt_answer = extract_final_answer(gt_text)
                pred_answer = extract_final_answer(pred_text)
                
                # Add evaluation results directly to the original data
                data["parsed_solution"] = str(gt_answer) if gt_answer else None
                data["parsed_generated_solution"] = str(pred_answer) if pred_answer else None
                data["correct"] = None
                
                if gt_answer is None or pred_answer is None:
                    parsing_failures += 1
                    error_indices.append(idx)
                else:
                    try:
                        if verify(gt_answer, pred_answer):
                            data["correct"] = True
                            correct += 1
                        else:
                            data["correct"] = False
                            verification_failures += 1
                        total += 1
                    except Exception:
                        verification_failures += 1
                        error_indices.append(idx)
                
                # Write the combined record to output file
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            # Log errors and summary
            logf.write("Indices with parse or verify failures:\n")
            for idx in error_indices:
                logf.write(f"{idx}\n")
            
            total_attempted = correct + verification_failures
            accuracy = (correct / total_attempted) * 100 if total_attempted > 0 else 0.0
            
            # Write summary to log file
            logf.write("\n=== Final Evaluation Summary ===\n")
            logf.write(f"Total Attempted (parsed): {total_attempted}\n")
            logf.write(f"Correct: {correct}\n")
            logf.write(f"Verification Failures: {verification_failures}\n")
            logf.write(f"Parsing Failures: {parsing_failures}\n")
            logf.write(f"Accuracy: {accuracy:.2f}%\n")
            
            # Record total time
            eval_time = time.time() - eval_start_time
            total_time = time.time() - start_time
            
            logf.write(f"\nGeneration Time: {generation_time:.2f} seconds\n")
            logf.write(f"Evaluation Time: {eval_time:.2f} seconds\n")
            logf.write(f"Total Time: {total_time:.2f} seconds\n")
        
        # Create a summary JSON file
        summary = {
            "model": base_model,
            "dataset_path": dataset_path,
            "output_suffix": output_suffix,
            "output_file": output_file,
            "log_file": log_file,
            "total_examples": len(test_dataset),
            "total_attempted": total_attempted,
            "correct": correct,
            "verification_failures": verification_failures,
            "parsing_failures": parsing_failures,
            "accuracy": accuracy,
            "generation_time": generation_time,
            "evaluation_time": eval_time,
            "total_time": total_time
        }
        
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\nEvaluation Results for {output_suffix}")
        print(f"Total Attempted (parsed): {total_attempted}")
        print(f"Correct: {correct}")
        print(f"Verification Failures: {verification_failures}")
        print(f"Parsing Failures: {parsing_failures}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Generation Time: {generation_time:.2f} seconds")
        print(f"Evaluation Time: {eval_time:.2f} seconds")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Combined results saved to {output_file}")
        print(f"Log saved to {log_file}")
        print(f"Summary saved to {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"Error in generation or evaluation process: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

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
        choices=["zero_shot", "four_shot"],
        default="four_shot",
        help="Whether to use zero-shot or four-shot prompting"
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
    
    # Adjust shot mode mapping to match the expected values
    shot_mode_mapping = {
        "zero_shot": "zero_shot",
        "four_shot": "few_shot"  # Map four_shot to few_shot as used internally
    }
    internal_shot_mode = shot_mode_mapping.get(args.shot_mode, "few_shot")
    
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
