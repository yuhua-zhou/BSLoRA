import argparse
import json
import logging

import lm_eval

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = ["piqa"]
    else:
        task_names = args.tasks.split(",")

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    task_manager = lm_eval.tasks.TaskManager()
    results = lm_eval.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        task_manager=task_manager,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        check_integrity=args.check_integrity,
    )

    dumped = json.dumps(results, indent=2)
    # print(dumped)

    if args.output_path:
        import os
        directory_path = os.path.dirname(args.output_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(args.output_path, "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
