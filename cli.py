import argparse
from pathlib import Path

from run_experiments import (
    run_rank1_experiment,
    run_relora_experiment,
    run_capacity_analysis,
    run_attention_analysis,
    run_sbd_benchmark,
    run_full_pipeline,
)


def main():
    parser = argparse.ArgumentParser(description="Gemma Lab CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-path", type=str, default="models/gemma-3-4b")

    sub.add_parser("rank1", parents=[common])
    sub.add_parser("relora", parents=[common])
    sub.add_parser("capacity", parents=[common])
    sub.add_parser("attention", parents=[common])
    sub.add_parser("sbd", parents=[common])
    sub.add_parser("full", parents=[common])

    args = parser.parse_args()

    model_path = args.model_path
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Run: python scripts/download_model.py --model gemma-3-4b")
        return

    if args.cmd == "rank1":
        run_rank1_experiment(model_path)
    elif args.cmd == "relora":
        run_relora_experiment(model_path)
    elif args.cmd == "capacity":
        run_capacity_analysis(model_path)
    elif args.cmd == "attention":
        run_attention_analysis(model_path)
    elif args.cmd == "sbd":
        run_sbd_benchmark(model_path)
    elif args.cmd == "full":
        run_full_pipeline(model_path)


if __name__ == "__main__":
    main()


