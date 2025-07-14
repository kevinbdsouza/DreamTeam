# DreamTeam Workflow Roadmap

This document outlines how to implement the **DreamTeam** workflow. The goal is to orchestrate a population of historical expert agents that iteratively propose modifications to `train_mps.py` and `model.py`, using an evolutionary algorithm to select the best-performing candidate in each generation.

## 1. Agent Prompts
- Each agent represents a historical mathematician, physicist, or computer scientist.
- Prompts reside in `src/dreamteam/prompts/*.txt` and describe the expert's perspective.
- Every prompt instructs the agent to respect the immutable settings listed in `train_mps.py` comments and to return `(best_vloss, elapsed_min)`.
- Agents must output the full updated source code for any file they modify using the format:
  ```
  file_name.py:
  <full file contents>
  ```

## 2. Population Initialization
- The population size matches the number of prompts (25).
- For each generation, load `train_mps.py` and `model.py` and pass their contents to each agent along with its prompt.
- Agents operate sequentially: one agent receives the base code, proposes modifications, and returns updated files. The next agent uses those updated files as its starting point, and so on.

## 3. Candidate Evaluation
- After an agent produces modified files, run `train_mps.py` to obtain `best_vloss` and `elapsed_min`.
- Record these values along with the agent identity and generation number.

## 4. Evolutionary Selection
- Once all agents in a generation have produced candidates, rank the results using a fitness function combining validation loss and elapsed time.
- Select the top candidates (e.g., the best N) to seed the next generation. Their code becomes the starting point for the next round of agent modifications.

## 5. Orchestration
- Implement a controller script that:
  1. Loads prompts and initial code.
  2. Iteratively invokes each agent via the Gemini API, providing the current code and the agent’s prompt.
  3. Saves each candidate’s returned files and evaluation metrics.
  4. Applies the evolutionary selection step and prepares the next generation.
  5. Repeats for the configured number of generations.

## 6. Result Aggregation
- Store all candidate code versions and metrics for analysis.
- After the final generation, identify the overall best-performing model.

## 7. Notes
- Agents must never modify the protected hyperparameters or the return statement in `train_mps.py`.
- The workflow intentionally avoids giving agents explicit suggestions about what code to change; they decide autonomously within the provided constraints.
