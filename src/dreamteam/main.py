import os
import importlib.util
import json
import re
from train_mps import train
from model import GPT, GPTConfig
import google.generativeai as genai
from dotenv import load_dotenv
import shutil

load_dotenv()

def parse_python_blocks(response_text, original_train_mps_code, original_model_py_code):
    """
    Parse Python code blocks from the response text.
    Returns the parsed code for train_mps.py and model.py.
    If a block is empty or invalid, uses the original code.
    """
    # Find Python code blocks
    train_mps_pattern = r'```python\s*train_mps\.py\s*\n(.*?)```'
    model_py_pattern = r'```python\s*model\.py\s*\n(.*?)```'
    
    # Try to find train_mps.py block
    train_mps_match = re.search(train_mps_pattern, response_text, re.DOTALL)
    new_train_mps_code = train_mps_match.group(1).strip() if train_mps_match else ""
    
    # Try to find model.py block
    model_py_match = re.search(model_py_pattern, response_text, re.DOTALL)
    new_model_py_code = model_py_match.group(1).strip() if model_py_match else ""
    
    # Validate Python syntax and use original if invalid/empty
    def is_valid_python(code):
        if not code.strip():
            return False
        
        # Check that the first word is 'import' (simple validation)
        first_line = code.strip().split('\n')[0].strip()
        if not first_line.startswith('import'):
            return False
        
        # Finally check Python syntax
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    # Check if both code blocks are invalid/empty - if so, return None to indicate this candidate should be ignored
    train_mps_invalid = not is_valid_python(new_train_mps_code)
    model_py_invalid = not is_valid_python(new_model_py_code)
    
    if train_mps_invalid and model_py_invalid:
        print("  Warning: Both train_mps.py and model.py code blocks are invalid or empty, ignoring this candidate")
        return None, None
    
    # Use original code if new code is invalid or empty
    if train_mps_invalid:
        print("  Warning: Invalid or empty train_mps.py code block, using original")
        new_train_mps_code = original_train_mps_code
    
    if model_py_invalid:
        print("  Warning: Invalid or empty model.py code block, using original")
        new_model_py_code = original_model_py_code
    
    return new_train_mps_code, new_model_py_code

def load_prompts_from_mode(mode):
    """
    Load prompts from a specific mode directory.
    Returns a dictionary mapping prompt names to their content.
    """
    prompts = {}
    mode_dir = f"prompts/{mode}"
    
    if not os.path.exists(mode_dir):
        print(f"Warning: Mode directory '{mode_dir}' does not exist")
        return prompts
    
    prompt_files = os.listdir(mode_dir)
    for file_name in prompt_files:
        if file_name.endswith('.txt'):
            with open(f"{mode_dir}/{file_name}", "r") as f:
                prompt_name = f"{mode}_{file_name.replace('.txt', '')}"
                prompts[prompt_name] = f.read()
    
    return prompts

def run_dreamteam_workflow(modes=None):
    """
    This is the main function that orchestrates the DreamTeam workflow.
    
    Args:
        modes: List of modes to use. If None, uses all available modes.
               Available modes: ['savant', 'collaborator', 'specialist']
    """
    shutil.copyfile("dreamteam_generations/gen_1/baseline/train_mps.py", "train_mps.py")
    shutil.copyfile("dreamteam_generations/gen_1/baseline/model.py", "model.py")

    # --- Configuration ---
    # Configure the Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    genai.configure(api_key=gemini_api_key)

    # --- Load Prompts and Code ---
    prompts = {}
    
    # If no modes specified, use all available modes
    if modes is None:
        modes = ['savant', 'collaborator', 'specialist']
    
    # Load prompts from each specified mode
    for mode in modes:
        mode_prompts = load_prompts_from_mode(mode)
        prompts.update(mode_prompts)
        print(f"Loaded {len(mode_prompts)} prompts from {mode} mode")

    if not prompts:
        raise ValueError("No prompts found in any of the specified modes")

    with open("train_mps.py", "r") as f:
        train_mps_code = f.read()
    with open("model.py", "r") as f:
        model_py_code = f.read()

    # --- Evolutionary Algorithm ---
    num_generations = 3 # TODO: Adjust as needed

    for generation in range(num_generations):
        print(f"--- Generation {generation+1} ---")
        candidates = []

        # --- Generate Candidates ---
        for name, prompt in prompts.items():
            print(f"Generating candidate from {name}...")
            # Create a Gemini client
            model = genai.GenerativeModel('gemini-2.5-flash')

            # Construct the full prompt
            full_prompt = f"{prompt}\n\nHere is the existing implementation:\n\n`train_mps.py`:\n```python\n{train_mps_code}\n```\n\n`model.py`:\n```python\n{model_py_code}\n```\n\nProvide your modified code in separate Python code blocks. If you don't modify a file, include an empty Python block for it.\n\n```python\ntrain_mps.py\n# Your modified train_mps.py code here\n```\n\n```python\nmodel.py\n# Your modified model.py code here\n```"

            # Generate the candidate code
            try:
                response = model.generate_content(full_prompt)
            except Exception as e:
                print(f"  Error generating content from {name}: {e}")
                continue

            # --- Parse the response ---
            try:
                new_train_mps_code, new_model_py_code = parse_python_blocks(response.text, train_mps_code, model_py_code)

                # Skip this candidate if both code blocks are invalid
                if new_train_mps_code is None and new_model_py_code is None:
                    print(f"  Skipping candidate from {name} due to invalid code blocks")
                    continue

                candidates.append({
                    "name": name,
                    "train_mps_code": new_train_mps_code,
                    "model_py_code": new_model_py_code
                })
            except Exception as e:
                print(f"  Error parsing response from {name}: {e}")
                print(f"  Response: {response.text}")


        # --- Evaluate Candidates ---
        results = []
        for candidate in candidates:
            print(f"Evaluating candidate from {candidate['name']}...")
            try:
                # --- Create a directory for the generation ---
                generation_dir = f"dreamteam_generations/gen_{generation+1}"
                os.makedirs(generation_dir, exist_ok=True)
                candidate_dir = f"{generation_dir}/{candidate['name']}"
                os.makedirs(candidate_dir, exist_ok=True)


                # --- Create temporary files for the candidate code ---
                train_mps_path = f"{candidate_dir}/train_mps.py"
                model_py_path = f"{candidate_dir}/model.py"
                with open(train_mps_path, "w") as f:
                    f.write(candidate["train_mps_code"])
                with open(model_py_path, "w") as f:
                    f.write(candidate["model_py_code"])

                # --- Import the temporary training script ---
                spec = importlib.util.spec_from_file_location("temp_train_mps", train_mps_path)
                temp_train_mps = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(temp_train_mps)

                # --- Run the training ---
                best_vloss, elapsed_min = temp_train_mps.train()

                results.append({
                    "name": candidate["name"],
                    "best_vloss": best_vloss,
                    "elapsed_min": elapsed_min,
                    "train_mps_code": candidate["train_mps_code"],
                    "model_py_code": candidate["model_py_code"]
                })

            except Exception as e:
                print(f"  Error running training for {candidate['name']}: {e}")
                results.append({
                    "name": candidate["name"],
                    "best_vloss": float('inf'),
                    "elapsed_min": float('inf'),
                    "train_mps_code": candidate["train_mps_code"],
                    "model_py_code": candidate["model_py_code"]
                })


        # --- Evaluate the baseline code ---
        print("Evaluating baseline...")
        #baseline_dir = f"dreamteam_generations/gen_{generation+1}/baseline"
        #os.makedirs(baseline_dir, exist_ok=True)
        #shutil.copyfile("train_mps.py", f"{baseline_dir}/train_mps.py")
        #shutil.copyfile("model.py", f"{baseline_dir}/model.py")
        #spec = importlib.util.spec_from_file_location("baseline_train_mps", f"{baseline_dir}/train_mps.py")
        #baseline_train_mps = importlib.util.module_from_spec(spec)
        #spec.loader.exec_module(baseline_train_mps)
        #base_vloss, base_elapsed = baseline_train_mps.train()
        results.append({
            "name": "baseline",
            "best_vloss": 4.6439,
            "elapsed_min": 6.89,
            "train_mps_code": train_mps_code,
            "model_py_code": model_py_code
        })


        # --- Select the Best Candidates ---
        results.sort(key=lambda x: x["best_vloss"])
        best_candidate = results[0]

        # --- Save the results ---
        generation_dir = f"dreamteam_generations/gen_{generation+1}"
        with open(f"{generation_dir}/results.txt", "w") as f:
            for res in results:
                f.write(f"{res['name']}: vloss={res['best_vloss']:.4f}, time={res['elapsed_min']:.2f} min\n")

        print(f"\n--- Generation {generation+1} Results ---")
        print(f"Best candidate: {best_candidate['name']}")
        print(f"  Validation loss: {best_candidate['best_vloss']:.4f}")
        print(f"  Elapsed time: {best_candidate['elapsed_min']:.2f} min")

        # --- Create the next generation ---
        # For now, we will just use the best candidate's code as the new base
        with open("train_mps.py", "w") as f:
            f.write(best_candidate["train_mps_code"])
        with open("model.py", "w") as f:
            f.write(best_candidate["model_py_code"])

        with open("train_mps.py", "r") as f:
            train_mps_code = f.read()
        with open("model.py", "r") as f:
            model_py_code = f.read()


if __name__ == "__main__":
    # Run with all modes by default
    run_dreamteam_workflow()
    
    # To run with specific modes only, uncomment and modify:
    # run_dreamteam_workflow(modes=['savant'])  # Only historical geniuses
    # run_dreamteam_workflow(modes=['collaborator'])  # Only research collaborators
    # run_dreamteam_workflow(modes=['specialist'])  # Only field specialists
    # run_dreamteam_workflow(modes=['savant', 'collaborator'])  # Mix of modes
