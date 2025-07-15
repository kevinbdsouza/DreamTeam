import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import google.generativeai as genai

from . import config
from .utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts() -> List[Tuple[str, str]]:
    """Return list of (agent_name, prompt_text)."""
    prompts = []
    prompt_dir = Path(config.PROMPTS_DIR)
    for path in sorted(prompt_dir.glob("*.txt")):
        prompts.append((path.stem, path.read_text()))
    return prompts


def call_agent(prompt: str) -> str:
    """Invoke Gemini API with the given prompt and return the response text."""
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt, stream=False)
    return response.text


def parse_agent_response(text: str) -> Dict[str, str]:
    """Parse agent response into mapping of file name to full contents."""
    sections = {}
    current_name = None
    current_lines: List[str] = []
    for line in text.splitlines():
        if line.strip().endswith(".py:"):
            if current_name:
                sections[current_name] = "\n".join(current_lines).rstrip() + "\n"
            current_name = line.strip()[:-1]
            current_lines = []
        else:
            current_lines.append(line)
    if current_name:
        sections[current_name] = "\n".join(current_lines).rstrip() + "\n"
    return sections


def write_candidate(base_dir: Path, files: Dict[str, str]) -> Path:
    """Write candidate files to a new directory and return its path."""
    cand_dir = base_dir
    cand_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (cand_dir / name).write_text(content)
    return cand_dir


def evaluate_candidate(cand_dir: Path) -> Tuple[float, float]:
    """Run train_mps.train() from candidate directory."""
    import sys
    import importlib

    sys.path.insert(0, str(cand_dir))
    try:
        import train_mps  # type: ignore
        importlib.reload(train_mps)
        best_vloss, elapsed_min = train_mps.train()
        if hasattr(best_vloss, "item"):
            best_vloss = float(best_vloss.item())
    finally:
        sys.path.pop(0)
        if "train_mps" in sys.modules:
            del sys.modules["train_mps"]
        if "model" in sys.modules:
            del sys.modules["model"]
    return float(best_vloss), float(elapsed_min)


def fitness(vloss: float, elapsed: float) -> float:
    return vloss + 0.01 * elapsed


def run_generation(gen: int, base_files: Dict[str, str], prompts: List[Tuple[str, str]], results_dir: Path) -> Dict[str, str]:
    gen_dir = results_dir / f"gen_{gen}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    best_files = base_files
    best_fit = float("inf")

    for idx, (agent_name, prompt) in enumerate(prompts):
        logger.info("Generation %s • Agent %s", gen, agent_name)
        combined = (
            f"{prompt}\n\nHere is train_mps.py:\n```\n{best_files['train_mps.py']}\n```\n\n"
            f"Here is model.py:\n```\n{best_files['model.py']}\n```"
        )
        response = call_agent(combined)
        files = parse_agent_response(response)
        if not files:
            logger.warning("Agent %s returned no files", agent_name)
            files = best_files
        else:
            for fname in base_files:
                files.setdefault(fname, best_files[fname])

        cand_path = gen_dir / f"{idx:02d}_{agent_name}"
        write_candidate(cand_path, files)
        vloss, elapsed = evaluate_candidate(cand_path)
        (cand_path / "metrics.json").write_text(json.dumps({"vloss": vloss, "elapsed": elapsed}))
        fit = fitness(vloss, elapsed)
        if fit < best_fit:
            best_fit = fit
            best_files = files
    return best_files


def main() -> None:
    prompts = load_prompts()
    results_dir = Path("results")
    base_files = {
        "train_mps.py": Path("src/dreamteam/train_mps.py").read_text(),
        "model.py": Path("src/dreamteam/model.py").read_text(),
    }

    for gen in range(config.N_GENERATIONS):
        logger.info("Starting generation %s", gen)
        base_files = run_generation(gen, base_files, prompts, results_dir)
        for name, content in base_files.items():
            (results_dir / f"best_{name}").write_text(content)

    logger.info("DreamTeam run complete")


if __name__ == "__main__":
    main()
