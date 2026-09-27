import argparse
import os

from .eval_coding import eval_coding
from .eval_misalignment import eval_misalignment

# Resolve data/prompt files relative to this file rather than a hardcoded /root/caft.
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_EVAL_DIR)
JUDGE_PROMPTS_CODING = os.path.join(_EVAL_DIR, "judge_prompts_coding.yaml")
OOD_QUESTIONS = os.path.join(_PKG_DIR, "data", "first_plot_questions.yaml")
CODE_DATASET = os.path.join(_PKG_DIR, "data", "insecure_val.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, default="eval_result.csv")
    parser.add_argument("--code_dataset", type=str, default=CODE_DATASET)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--n_per_ood_question", type=int, default=100)
    parser.add_argument("--n_per_code_question", type=int, default=1)
    # eval_coding and eval_misalignment each build their own vLLM engine at
    # gpu_memory_utilization=0.95, so running both in one process OOMs on the
    # second. Run them as separate invocations instead.
    parser.add_argument("--only", choices=["coding", "misalignment", "both"],
                        default="both")

    args = parser.parse_args()
    output_coding = args.output.split(".")[0] + "_coding.csv"
    output_misalignment = args.output.split(".")[0] + "_misalignment.csv"

    if args.only == "both":
        raise SystemExit(
            "Refusing to run both evals in one process: each builds a vLLM engine at\n"
            "gpu_memory_utilization=0.95, so the second fails to allocate. Run:\n"
            "  --only coding\n"
            "  --only misalignment\n"
            "as two separate commands."
        )

    if args.only == "coding":
        eval_coding(args.model,
                    args.code_dataset,
                    n_per_question=args.n_per_code_question,
                    output=output_coding,
                    lora_path=args.lora,
                    judge_prompts_path=JUDGE_PROMPTS_CODING)

    if args.only == "misalignment":
        eval_misalignment(args.model,
                          questions=OOD_QUESTIONS,
                          n_per_question=args.n_per_ood_question,
                          output=output_misalignment,
                          lora_path=args.lora)
