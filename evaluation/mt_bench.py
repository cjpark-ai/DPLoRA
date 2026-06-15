"""MT-Bench harness via FastChat: spawns gen_model_answer/gen_judgment/show_result,
reads score from judgment .jsonl. Caller pre-saves the LoRA-merged HF model to ``model_path``."""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# model_type -> conv_template; unsupported type fails fast (no silent Llama fallback).
TEMPLATE_MAP = {
    "qwen2": "alpaca_dplora_qwen",
    "qwen": "alpaca_dplora_qwen",
    "llama": "alpaca_dplora_llama",
}


def _resolve_num_gpus() -> int:
    """Resolve visible GPU count by parsing CUDA_VISIBLE_DEVICES directly (PyTorch
    caches device_count() after first call, so later env changes have no effect)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        parts = [x.strip() for x in cvd.split(",") if x.strip()]
        if parts:
            return len(parts)
        # Empty CUDA_VISIBLE_DEVICES → fall through to torch detection; never
        # return 0, which would ZeroDivision in FastChat's chunk_size.
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def _tail_log(path: str, n: int = 50) -> str:
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(log file not found or unreadable)"


def _archive_artifact(src: Optional[str], dest_dir: str, dest_name: str,
                      logger) -> Optional[str]:
    """Copy a per-run artifact into output_dir so each run is self-contained
    (FastChat recycles its copy). Returns dest path, or None on miss/failure (non-fatal)."""
    if not src or not os.path.exists(src):
        return None
    dest = os.path.join(dest_dir, dest_name)
    try:
        shutil.copy2(src, dest)
        return dest
    except OSError as e:
        logger.warning(f"Failed to archive {src} -> {dest}: {e}")
        return None


def _archive_judgment_rows(src: Optional[str], dest_dir: str, dest_name: str,
                           model_id: str, logger) -> Optional[str]:
    """Write only this run's judgment rows (model==model_id) into output_dir. The
    shared gpt-4_single.jsonl is never deleted (concurrent-run race), so filter by model_id."""
    if not src or not os.path.exists(src):
        return None
    dest = os.path.join(dest_dir, dest_name)
    try:
        kept = 0
        with open(src) as fin, open(dest, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("model") == model_id:
                    fout.write(line + "\n")
                    kept += 1
        if kept == 0:
            os.remove(dest)
            return None
        return dest
    except OSError as e:
        logger.warning(f"Failed to archive judgment rows -> {dest}: {e}")
        return None


def _compute_score_from_judgment(judgment_file: str, model_id: str,
                                 logger) -> Optional[float]:
    """Compute MT-Bench Average from the judgment .jsonl (mirrors FastChat
    show_result.py: keep score!=-1 and model==model_id, mean). None if no valid rows."""
    if not os.path.exists(judgment_file):
        logger.warning(f"Judgment file not found: {judgment_file}")
        return None
    try:
        import pandas as pd
        df = pd.read_json(judgment_file, lines=True)
    except (ValueError, ImportError) as e:
        logger.warning(f"Failed to read judgment file {judgment_file}: {e}")
        return None
    if df.empty:
        logger.warning(f"Judgment file is empty: {judgment_file}")
        return None
    df = df[(df["score"] != -1) & (df["model"] == model_id)]
    if df.empty:
        logger.warning(f"No valid judgments for model_id={model_id}")
        return None
    score = float(df["score"].mean())
    if not (0 <= score <= 10):
        logger.warning(f"Computed score {score} outside [0,10] range")
        return None
    return score


def run_mt_bench_evaluation(model_path, output_dir, logger,
                            skip_judgment=False, fastchat_path=None,
                            subprocess_timeout: Optional[int] = None):
    """Run MT-Bench answer-gen + GPT-4 judgment + score report. ``model_path`` = a
    pre-saved LoRA-merged HF dir; returns results dict, or None on hard failure."""
    if subprocess_timeout is None:
        subprocess_timeout = int(os.environ.get("MT_BENCH_TIMEOUT", "10800"))

    try:
        # Resolve FastChat path: caller arg, FASTCHAT_PATH env, then script-relative defaults.
        if fastchat_path is None:
            fastchat_path = os.environ.get("FASTCHAT_PATH")

            if not fastchat_path:
                pkg_dir = os.path.dirname(os.path.abspath(__file__))
                possible_paths = [
                    os.path.join(pkg_dir, "..", "FastChat"),
                    os.path.join(pkg_dir, "..", "..", "FastChat"),
                    os.path.expanduser("~/FastChat"),
                ]
                for path in possible_paths:
                    normalized = os.path.normpath(path)
                    if os.path.exists(normalized) and os.path.exists(os.path.join(normalized, "fastchat")):
                        fastchat_path = normalized
                        break

            if fastchat_path is None:
                logger.error("FastChat repository not found!")
                logger.error("Set FASTCHAT_PATH in .env file or environment variable")
                logger.error("Or clone to current directory: git clone https://github.com/lm-sys/FastChat.git")
                return None

        fastchat_path = os.path.abspath(fastchat_path)
        llm_judge_path = os.path.join(fastchat_path, "fastchat", "llm_judge")

        if not os.path.isdir(model_path):
            logger.error(f"model_path does not exist or is not a directory: {model_path}")
            return None

        os.makedirs(output_dir, exist_ok=True)

        # model_id with collision-free suffix.
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix_src = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        short_uid = uuid.uuid4().hex[:6]
        model_id = f"alpaca_final_{ts}_{suffix_src}_{short_uid}"

        log_dir = os.path.join(output_dir, f"mt_bench_logs_{model_id}")
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"MT-Bench subprocess logs → {log_dir}")

        results = {
            "model_id": model_id,
            "model_path": model_path,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "log_dir": log_dir,
        }

        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            logger.error(f"config.json not found in {model_path}")
            return None
        with open(config_path, "r") as f:
            model_config = json.load(f)
        model_type = model_config.get("model_type", "").lower()
        # Eval in the checkpoint's saved dtype (merged model is bf16) instead of
        # FastChat's hardcoded fp16 default — faithful + avoids fp16 overflow.
        saved_dtype = str(model_config.get("torch_dtype", "")).lower()
        eval_dtype = saved_dtype if saved_dtype in (
            "float32", "float16", "bfloat16") else None
        conv_template = None
        for key, tpl in TEMPLATE_MAP.items():
            if key in model_type:
                conv_template = tpl
                break
        if conv_template is None:
            logger.error(
                f"Unsupported model_type for MT-Bench: '{model_type}'. "
                f"Supported keys: {list(TEMPLATE_MAP.keys())}"
            )
            return None
        logger.info(f"Conversation template: {conv_template} (model_type={model_type})")

        # child env: cap tqdm rate + forward parent env. Prepend FastChat root to
        # PYTHONPATH so subprocesses import the local fastchat (custom conv templates).
        child_env = {**os.environ, "TQDM_MININTERVAL": "10"}
        child_env["PYTHONPATH"] = fastchat_path + os.pathsep + os.environ.get("PYTHONPATH", "")
        num_gpus_total = _resolve_num_gpus()

        logger.info("Generating MT-Bench answers...")
        gen_answer_script = os.path.join(llm_judge_path, "gen_model_answer.py")

        cmd = [
            sys.executable, gen_answer_script,
            "--model-path", model_path,
            "--model-id", model_id,
            "--conv-template", conv_template,
            "--num-gpus-per-model", "1",
            "--num-gpus-total", str(num_gpus_total),
        ]
        if eval_dtype is not None:
            cmd += ["--dtype", eval_dtype]
        else:
            logger.warning(
                "config.json has no usable torch_dtype; FastChat will default "
                "to float16 (may not match the saved checkpoint)."
            )

        logger.info(f"Running command: {' '.join(cmd)}")

        step1_log = os.path.join(log_dir, "step1_gen_answer.log")
        try:
            with open(step1_log, "w", buffering=1) as fout:
                # stdout->file, stderr merged; avoid buffering multi-MB output.
                # stdin=DEVNULL: never inherit a TTY (no interactive prompts here).
                result = subprocess.run(
                    cmd, cwd=llm_judge_path,
                    stdin=subprocess.DEVNULL,
                    stdout=fout, stderr=subprocess.STDOUT,
                    timeout=subprocess_timeout,
                    env=child_env,
                )
        except subprocess.TimeoutExpired:
            logger.error(
                f"Answer generation timed out after {subprocess_timeout}s. "
                f"Last log lines:\n{_tail_log(step1_log, 50)}"
            )
            return results

        if result.returncode != 0:
            logger.error(
                f"Answer generation failed (rc={result.returncode}). "
                f"Last log lines:\n{_tail_log(step1_log, 50)}"
            )
            return results

        answer_file = os.path.join(llm_judge_path, "data/mt_bench/model_answer", f"{model_id}.jsonl")
        results["answer_file"] = answer_file
        logger.info(f"Answers generated: {answer_file}")

        if not skip_judgment and "OPENAI_API_KEY" in os.environ:
            logger.info("Getting GPT-4 judgments...")
            gen_judgment_script = os.path.join(llm_judge_path, "gen_judgment.py")

            # gen_judgment.py appends to this shared fixed-name file; do NOT delete
            # first (races with concurrent jobs). Scoring/archival filter by model_id.
            judgment_file = os.path.join(
                llm_judge_path, "data/mt_bench/model_judgment",
                "gpt-4_single.jsonl",
            )

            cmd = [
                sys.executable, gen_judgment_script,
                "--model-list", model_id,
                "--parallel", "2",
                "--mode", "single",
            ]

            step2_log = os.path.join(log_dir, "step2_gen_judgment.log")
            try:
                with open(step2_log, "w", buffering=1) as fout:
                    # gen_judgment.py has an interactive input(); feed a newline so it
                    # doesn't raise EOFError under non-interactive (sbatch) stdin.
                    result = subprocess.run(
                        cmd, cwd=llm_judge_path,
                        input="\n", text=True,
                        stdout=fout, stderr=subprocess.STDOUT,
                        timeout=subprocess_timeout,
                        env=child_env,
                    )
            except subprocess.TimeoutExpired:
                logger.error(
                    f"GPT-4 judgment timed out after {subprocess_timeout}s. "
                    f"Last log lines:\n{_tail_log(step2_log, 50)}"
                )
                result = None

            if result is not None and result.returncode == 0:
                # show_result.py is run only for a human-readable log; its
                # stdout is not parsed.
                show_result_script = os.path.join(llm_judge_path, "show_result.py")
                cmd = [
                    sys.executable, show_result_script,
                    "--model-list", model_id,
                    "--mode", "single",
                ]
                step3_log = os.path.join(log_dir, "step3_show_result.log")
                try:
                    with open(step3_log, "w", buffering=1) as fout:
                        subprocess.run(
                            cmd, cwd=llm_judge_path,
                            stdin=subprocess.DEVNULL,
                            stdout=fout, stderr=subprocess.STDOUT,
                            timeout=600,
                            env=child_env,
                        )
                except subprocess.TimeoutExpired:
                    logger.warning(
                        f"show_result.py timed out (600s). "
                        f"Last log lines:\n{_tail_log(step3_log, 30)}"
                    )

                # read judgment file directly (no brittle stdout regex).
                results["judgment_file"] = judgment_file
                score_found = _compute_score_from_judgment(
                    judgment_file, model_id, logger
                )
                if score_found is not None:
                    results["overall_score"] = score_found
                    logger.info(f"MT-Bench Overall Score: {score_found:.2f}")
                else:
                    logger.warning(
                        "Could not compute MT-Bench score from judgment file"
                    )
            else:
                logger.warning("GPT-4 judgment failed or skipped")
        else:
            if skip_judgment:
                logger.info("Skipping GPT-4 judgment as requested")
                results["judgment_skipped_reason"] = "requested"
            else:
                logger.warning("Skipping GPT-4 judgment - OpenAI API key not set")
                # Distinguish "wanted judgment but no key" so the CLI can exit
                # non-zero (a no-score run is a failure, not a silent success).
                results["judgment_skipped_reason"] = "no_api_key"
            results["judgment_skipped"] = True

        # Archive per-run artifacts next to the score so each run is
        # self-contained.
        archived_answer = _archive_artifact(
            results.get("answer_file"), output_dir,
            f"model_answer_{model_id}.jsonl", logger)
        if archived_answer:
            results["archived_answer_file"] = archived_answer
        archived_judgment = _archive_judgment_rows(
            results.get("judgment_file"), output_dir,
            f"gpt4_judgment_{model_id}.jsonl", model_id, logger)
        if archived_judgment:
            results["archived_judgment_file"] = archived_judgment

        results_file = os.path.join(output_dir, f"mt_bench_results_{model_id}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"MT-Bench results saved to {results_file}")

        return results

    except Exception as e:
        logger.error(f"MT-Bench evaluation failed with error: {e}")
        return None


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MT-Bench (FastChat llm_judge) on a merged HF model dir "
                    "and emit the results dict as JSON on stdout."
    )
    p.add_argument(
        "model_path",
        help="Already-merged vanilla HF model dir (config.json + weights), "
             "e.g. output/.../final_model",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Where to write results/logs "
             "(default: <dirname(model_path)>/mt_bench_results)",
    )
    p.add_argument(
        "--skip-judgment", action="store_true",
        help="Generate answers only; skip GPT-4 judgment.",
    )
    p.add_argument(
        "--fastchat-path", default=None,
        help="FastChat repo path (auto-detected if omitted).",
    )
    return p


def main(argv=None) -> int:
    """CLI entry. Logs → stderr; final stdout line is the results JSON.
    Exit 0 on success, 1 on hard failure."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cli_logger = logging.getLogger("mt_bench")

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.model_path)), "mt_bench_results"
    )
    results = run_mt_bench_evaluation(
        model_path=args.model_path,
        output_dir=output_dir,
        logger=cli_logger,
        skip_judgment=args.skip_judgment,
        fastchat_path=args.fastchat_path,
    )
    if results is None:
        cli_logger.error("MT-Bench evaluation failed (see logs above).")
        return 1
    # Machine-readable result → stdout (always, so artifacts are discoverable).
    print(json.dumps(results))
    # "Wanted judgment but no OPENAI_API_KEY" is a failure (no score) → exit non-zero
    # so automation catches it; explicit --skip-judgment stays exit 0.
    if results.get("judgment_skipped_reason") == "no_api_key":
        cli_logger.error(
            "No MT-Bench score: OPENAI_API_KEY not set (use --skip-judgment to "
            "intentionally generate answers only)."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
