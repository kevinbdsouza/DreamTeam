import os
import importlib.util
import json
from train_mps import train
from model import GPT, GPTConfig
import google.generativeai as genai
from dotenv import load_dotenv
import shutil

load_dotenv()

def run_dreamteam_workflow():
    """
    This is the main function that orchestrates the DreamTeam workflow.
    """
    shutil.copyfile("train_mps.py", "train_mps_bck.py")
    shutil.copyfile("model.py", "model_bck.py")

    # --- Configuration ---
    # Configure the Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    genai.configure(api_key=gemini_api_key)

    # --- Load Prompts and Code ---
    prompts = {}
    prompt_files = os.listdir("prompts")
    for file_name in prompt_files[:2]:
        with open(f"prompts/{file_name}", "r") as f:
            prompts[file_name.replace(".txt", "")] = f.read()

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
            full_prompt = f"{prompt}\n\nHere is the code to modify:\n\n`train_mps.py`:\n```python\n{train_mps_code}\n```\n\n`model.py`:\n```python\n{model_py_code}\n```"

            # Generate the candidate code
            response = model.generate_content(full_prompt)

            # --- Parse the response ---
            try:
                json_start = response.text.find('{')
                json_end = response.text.rfind('}') + 1
                files_dict = json.loads(response.text[json_start:json_end])

                new_train_mps_code = files_dict.get("train_mps.py")
                new_model_py_code = files_dict.get("model.py")
                if not new_train_mps_code or not new_model_py_code:
                    raise ValueError("Missing expected keys in response JSON")

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
        baseline_dir = f"dreamteam_generations/gen_{generation+1}/baseline"
        os.makedirs(baseline_dir, exist_ok=True)
        shutil.copyfile("train_mps.py", f"{baseline_dir}/train_mps.py")
        shutil.copyfile("model.py", f"{baseline_dir}/model.py")
        spec = importlib.util.spec_from_file_location("baseline_train_mps", f"{baseline_dir}/train_mps.py")
        baseline_train_mps = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_train_mps)
        base_vloss, base_elapsed = baseline_train_mps.train()
        results.append({
            "name": "baseline",
            "best_vloss": base_vloss,
            "elapsed_min": base_elapsed,
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
    run_dreamteam_workflow()
