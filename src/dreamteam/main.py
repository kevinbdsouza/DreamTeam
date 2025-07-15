import os
import importlib.util
from train_mps import train
from model import GPT, GPTConfig
import google.generativeai as genai


def run_dreamteam_workflow():
    """
    This is the main function that orchestrates the DreamTeam workflow.
    """
    # --- Configuration ---
    # Configure the Gemini API key
    genai.configure(api_key="YOUR_API_KEY") # TODO: Replace with your API key

    # --- Load Prompts and Code ---
    prompts = {}
    prompt_files = os.listdir("src/dreamteam/prompts")
    for file_name in prompt_files:
        with open(f"src/dreamteam/prompts/{file_name}", "r") as f:
            prompts[file_name.replace(".txt", "")] = f.read()

    with open("src/dreamteam/train_mps.py", "r") as f:
        train_mps_code = f.read()
    with open("src/dreamteam/model.py", "r") as f:
        model_py_code = f.read()

    # --- Evolutionary Algorithm ---
    population_size = len(prompts)
    num_generations = 3 # TODO: Adjust as needed

    for generation in range(num_generations):
        print(f"--- Generation {generation+1} ---")
        candidates = []

        # --- Generate Candidates ---
        for name, prompt in prompts.items():
            print(f"Generating candidate from {name}...")
            # Create a Gemini client
            model = genai.GenerativeModel('gemini-pro')

            # Construct the full prompt
            full_prompt = f"{prompt}\n\nHere is the code to modify:\n\n`train_mps.py`:\n```python\n{train_mps_code}\n```\n\n`model.py`:\n```python\n{model_py_code}\n```"

            # Generate the candidate code
            response = model.generate_content(full_prompt)

            # --- Parse the response ---
            try:
                # Extract the code from the response
                # TODO: This parsing is very basic and might need to be improved
                new_train_mps_code = response.text.split("`train_mps.py`:\n```python")[1].split("```")[0]
                new_model_py_code = response.text.split("`model.py`:\n```python")[1].split("```")[0]

                candidates.append({
                    "name": name,
                    "train_mps_code": new_train_mps_code,
                    "model_py_code": new_model_py_code
                })
            except (IndexError, AttributeError) as e:
                print(f"  Error parsing response from {name}: {e}")
                print(f"  Response: {response.text}")


        # --- Evaluate Candidates ---
        results = []
        for candidate in candidates:
            print(f"Evaluating candidate from {candidate['name']}...")
            try:
                # --- Create temporary files for the candidate code ---
                with open("src/dreamteam/temp_train_mps.py", "w") as f:
                    f.write(candidate["train_mps_code"])
                with open("src/dreamteam/temp_model.py", "w") as f:
                    f.write(candidate["model_py_code"])

                # --- Import the temporary training script ---
                spec = importlib.util.spec_from_file_location("temp_train_mps", "src/dreamteam/temp_train_mps.py")
                temp_train_mps = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(temp_train_mps)

                # --- Run the training ---
                best_vloss, elapsed_min = temp_train_mps.train()

                results.append({
                    "name": candidate["name"],
                    "best_vloss": best_vloss,
                    "elapsed_min": elapsed_min
                })

            except Exception as e:
                print(f"  Error running training for {candidate['name']}: {e}")

            finally:
                # --- Clean up temporary files ---
                if os.path.exists("src/dreamteam/temp_train_mps.py"):
                    os.remove("src/dreamteam/temp_train_mps.py")
                if os.path.exists("src/dreamteam/temp_model.py"):
                    os.remove("src/dreamteam/temp_model.py")


        # --- Select the Best Candidates ---
        # TODO: Implement a more sophisticated selection strategy
        results.sort(key=lambda x: x["best_vloss"])
        best_candidate = results[0]

        print(f"\n--- Generation {generation+1} Results ---")
        print(f"Best candidate: {best_candidate['name']}")
        print(f"  Validation loss: {best_candidate['best_vloss']:.4f}")
        print(f"  Elapsed time: {best_candidate['elapsed_min']:.2f} min")

        # --- Create the next generation ---
        # For now, we will just use the best candidate's code as the new base
        with open("src/dreamteam/train_mps.py", "w") as f:
            f.write(candidates[0]["train_mps_code"])
        with open("src/dreamteam/model.py", "w") as f:
            f.write(candidates[0]["model_py_code"])

        with open("src/dreamteam/train_mps.py", "r") as f:
            train_mps_code = f.read()
        with open("src/dreamteam/model.py", "r") as f:
            model_py_code = f.read()


if __name__ == "__main__":
    run_dreamteam_workflow()
