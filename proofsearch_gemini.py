# Lean proof search with Google Gemini API interaction
# Adapted from original vLLM version by Kevin Dsouza (user)
# Author of adaptation: ChatGPT (OpenAI o3)
#
# Key changes:
#   • Replaces vLLM local model with Google Generative AI (Gemini) Cloud API.
#   • Removes vllm‑specific code, tensor‑parallel configs, and GPU requirements so the
#     script can run entirely on a macOS laptop/desktop (CPU‑only).
#   • Adds a lightweight sampling wrapper around the Gemini SDK that approximates the
#     score‑based best‑first search used in the original script.
#   • CLI trimmed: dropped --tp-degree (TP), CUDA env, and model choices now default to
#     "gemini-pro" (or any other model that "google-generativeai" supports).
#
# Requirements (install with pip):
#   pip install google-generativeai lean-dojo tqdm
#   # plus any existing LeanDojo deps (elan/Lean 4 binary must be on PATH).
#
# Before running set your Gemini API key once per shell session:
#   export GEMINI_API_KEY="<YOUR‑KEY‑HERE>"
#
# Example usage (single shard, CPU‑only):
#   python proofsearch_gemini.py --model-name gemini-pro \
#       --dataset-name minif2f-test --shard 0 --num-shards 1 \
#       --max-iters 100 --num-samples 8 --temperatures 0.7
#
# Notes / Caveats
#  • Gemini does not expose token‑level log‑probabilities; we therefore approximate the
#    search cost with the candidate‑rank (1st → highest score, 2nd → slightly lower).
#  • Candidate count is capped at ≤8 by Gemini today, so large NUM_SAMPLES values will
#    be clipped.
#  • If you need strict step limits, rely on the --timeout flag because the network
#    call latency is variable.

from __future__ import annotations

import json
import heapq
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import google.generativeai as genai
import tqdm  # tqdm >= 4.66 provides trange/tqdm
from lean_dojo import *  # noqa: F401, F403 (upstream requirement)
from dotenv import load_dotenv

# --------------------------------------------------------------------------------------
# Gemini helpers
# --------------------------------------------------------------------------------------
_gemini_configured = False

def _configure_gemini(api_key_env: str = "GEMINI_API_KEY") -> None:
    """Configure global Gemini credentials exactly once (idempotent)."""
    global _gemini_configured
    if _gemini_configured:
        return
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env} must be set with your Gemini API key"
        )
    # The SDK’s `configure` call is safe to run multiple times but we guard anyway
    genai.configure(api_key=api_key)
    _gemini_configured = True


class GeminiWrapper:
    """Thin wrapper exposing a generate() interface similar to vLLM."""

    def __init__(self, model_name: str = "gemini-pro") -> None:
        _configure_gemini()
        self._model = genai.GenerativeModel(model_name=model_name)

    # ------------------------------------------------------------------
    # The public method expected by the rest of the script
    # ------------------------------------------------------------------
    def generate(
            self,
            prompt: str,
            *,
            temperatures: List[float],
            num_samples: int,
            stop: List[str] | None = None,
            max_tokens: int = 256,
    ) -> Tuple[List[str], List[float]]:
        """Return (texts, scores) lists in descending score order."""

        texts: List[str] = []
        scores: List[float] = []

        if num_samples <= 0:
            return [], []

        num_samples = min(num_samples, 8)  # Gemini hard‑limit (July 2025)

        for temp in temperatures:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temp,
                    max_output_tokens=max_tokens,
                    candidate_count=num_samples,
                    stop_sequences=stop or [],
                ),
                stream=False,
            )

            # response.candidates is sorted best‑to‑worst already
            for rank, cand in enumerate(response.candidates):
                text = cand.text.strip()
                # Lack of token‑logprobs: approximate score via inverse rank
                score = 1.0 / (rank + 1)
                texts.append(text)
                scores.append(score)

        # remove dupes & keep best score for each unique text
        uniq: dict[str, float] = {}
        for t, s in zip(texts, scores):
            if t not in uniq or s > uniq[t]:
                uniq[t] = s
        sorted_pairs = sorted(uniq.items(), key=lambda kv: -kv[1])
        texts_sorted, scores_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])
        return list(texts_sorted), list(scores_sorted)


# --------------------------------------------------------------------------------------
# Original helper functions (mostly unchanged aside from vLLM removal)
# --------------------------------------------------------------------------------------


def _tactic_state(state: TacticState | ProofFinished) -> str:
    if isinstance(state, TacticState):
        return state.pp
    return state.unsolved_tactic_state


def _prompt_fewshot(ts: str) -> str:
    # Same canonical few‑shot prompt as upstream
    return (
            """Tactic state:
    ---
    α : Type u_1
    r : α → α → Prop
    inst✝¹ : DecidableEq α
    inst✝ : IsIrrefl α r
    ⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
    ---
    Next tactic:
    ---
    rintro s t ⟨u, a, hr, he⟩
    ---
    
    Tactic state:
    ---
    ι : Type u_1
    I✝ J✝ : Box ι
    x y : ι → ℝ
    I J : WithBot (Box ι)
    ⊢ ↑I = ↑J ↔ I = J
    ---
    Next tactic:
    ---
    simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
    ---
    
    Tactic state:
    ---
    m n : ℕ
    h : Nat.coprime m n
    ⊢ Nat.gcd m n = 1
    ---
    Next tactic:
    ---
    rw [← h.gcd_eq_one]
    ---
    
    Tactic state:
    ---
    %s
    ---
    Next tactic:
    ---\n"""
            % (ts)
    )


# --------------------------------------------------------------------------------------
# Best‑first proof search (unchanged aside from generate() backend call)
# --------------------------------------------------------------------------------------


def best_first_search(
        theorem: Theorem,
        llm: GeminiWrapper,
        *,
        max_iters: int,
        temperatures: List[float],
        num_samples: int,
        prompt_fn,
        timeout: int = 600,
        early_stop: bool = False,
        max_tokens: int = 256,
):
    """Run best‑first proof search using an external LLM."""

    attempt_results = []

    try:
        with Dojo(theorem, hard_timeout=timeout) as (dojo, init_state):
            start = time.time()
            proof_finished = False
            queue: List[tuple[float, list[str], TacticState | ProofFinished, list[dict[str, str]]]] = [
                (0.0, [], init_state, [])
            ]
            visited: set[str] = set()

            for iteration in tqdm.trange(max_iters):
                if not queue or proof_finished:
                    break

                total_score, steps, state, trace = heapq.heappop(queue)
                ts = _tactic_state(state)
                visited.add(ts)

                step_cands, step_scores = llm.generate(
                    prompt_fn(ts),
                    temperatures=temperatures,
                    num_samples=num_samples,
                    stop=["---", "\n"],
                    max_tokens=max_tokens,
                )

                step_cands = [s.strip() for s in step_cands]

                for step, score in zip(step_cands, step_scores):
                    result = dojo.run_tac(state, step)
                    step_trace = {"tactic": step, "state_before": ts}

                    if isinstance(result, ProofFinished):
                        attempt_results.append(
                            {
                                "theorem": theorem.full_name,
                                "proof": steps + [step],
                                "score": total_score - score,
                                "success": True,
                                "failure_reason": "",
                                "trace": trace + [step_trace],
                                "temperature": temperatures,
                                "elapsed": time.time() - start,
                                "iteration": iteration,
                            }
                        )
                        if early_stop:
                            return attempt_results
                        proof_finished = True
                        break

                    elif isinstance(result, TacticState):
                        next_ts = _tactic_state(result)
                        if next_ts not in visited:
                            new_score = total_score - score
                            heapq.heappush(
                                queue, (new_score, steps + [step], result, trace + [step_trace])
                            )

    except (
            DojoInitError,
            DojoHardTimeoutError,
            DojoCrashError,
            subprocess.CalledProcessError,
    ) as e:
        # One exception per theorem is fine—just record failure and move on.
        attempt_results.append(
            {
                "theorem": theorem.full_name,
                "success": False,
                "failure_reason": type(e).__name__,
            }
        )

    if not attempt_results:
        attempt_results.append(
            {
                "theorem": theorem.full_name,
                "success": False,
                "failure_reason": "SearchEnded",
            }
        )

    return attempt_results


# --------------------------------------------------------------------------------------
# Utility helpers for I/O, sharding, etc. (minor tweaks only)
# --------------------------------------------------------------------------------------


def _save(model_name: str, results, args_dict, output_dir: str, shard: int):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"results__{model_name.replace('/', '_')}__{shard}.json"
    )
    with open(output_file, "w") as f:
        json.dump({"results": results, "args": args_dict}, f, indent=4)
    print(f"Saved to {output_file}")


def _load_model(model_name: str) -> GeminiWrapper:
    return GeminiWrapper(model_name)


def _load_data(dataset_name: str, dataset_path: str):
    if "minif2f" in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f:
                d = json.loads(line)
                # Ensure commit matches LeanDojo dataset snapshot
                assert (
                        d["commit"] == "d00c776260c77de7e70125ef0cd119de6c0ff1de"
                ), "Dataset commit mismatch—update script if necessary."
                data.append(d)
        if "valid" in dataset_name:
            data = [x for x in data if x["split"] == "valid"]
        else:
            data = [x for x in data if x["split"] == "test"]
        repo = LeanGitRepo(data[0]["url"], data[0]["commit"])
    else:
        raise NotImplementedError(dataset_name)

    return repo, data


def print_stats(results):
    success_rate = sum(1 for x in results if x["success"]) / len(results)
    print(f"Success rate: {success_rate:.2%}")
    print("# successes:", sum(1 for x in results if x["success"]))


def resume_from(results_filename: str, data):
    results = json.load(open(results_filename))["results"]
    data = data[len(results):]
    print(f"=== Resuming from {len(results)} completed theorems")
    return results, data


def make_output_dir(output_dir: str):
    dt = datetime.now().strftime("%d-%m-%Y-%H-%M")
    full = os.path.join(output_dir, dt)
    Path(full).mkdir(parents=True, exist_ok=True)
    return full


# --------------------------------------------------------------------------------------
# CLI entry‑point (rewritten for Gemini)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default="gemini-2.0-flash",
        help="Gemini model to use (e.g. gemini-pro, gemini-1.5-pro)"
    )
    parser.add_argument(
        "--dataset-name",
        default="minif2f-test",
        choices=["minif2f-valid", "minif2f-test"],
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--dataset-path", default="data/minif2f.jsonl")
    parser.add_argument("--output-dir", default="output/minif2f")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7])
    parser.add_argument("--clear-process-hours", type=int, default=3)

    args = parser.parse_args()

    llm = _load_model(args.model_name)
    output_dir = make_output_dir(args.output_dir)

    repo, data = _load_data(args.dataset_name, args.dataset_path)
    shard_size = len(data) // args.num_shards
    start_idx = args.shard * shard_size
    end_idx = len(data) if args.shard + 1 == args.num_shards else (args.shard + 1) * shard_size
    data = data[start_idx:end_idx]
    print(f"Shard size: {len(data)} (records {start_idx}..{end_idx - 1})")

    if args.resume_from:
        results, data = resume_from(args.resume_from, data)
    else:
        results = []

    loop_start = time.time()

    for example in tqdm.tqdm(data, total=len(data)):
        file_path = example["file_path"]
        theorem_name = example["full_name"]
        theorem = Theorem(repo, file_path, theorem_name)

        attempt_results = best_first_search(
            theorem,
            llm,
            max_iters=args.max_iters,
            prompt_fn=_prompt_fewshot,
            temperatures=args.temperatures,
            num_samples=args.num_samples,
            timeout=args.timeout,
            early_stop=args.early_stop,
        )

        results.append(
            {
                "attempt_results": attempt_results,
                "success": any(x["success"] for x in attempt_results),
                "example": example,
            }
        )

        _save(
            model_name=args.model_name,
            results=results,
            args_dict=vars(args),
            output_dir=output_dir,
            shard=args.shard,
        )
        print_stats(results)

        # Workaround for Lean process leak: kill periodically
        if args.shard == 0 and time.time() - loop_start > args.clear_process_hours * 3600:
            print("=== Killing stray leanprover processes (memory leak mitigation)")
            os.system("pkill -9 -f leanprover || true")
            loop_start = time.time()
