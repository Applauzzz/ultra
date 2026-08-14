import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Shuffle a JSONL file using Huggingface Datasets with multiprocessing.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42).")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of processes for reading and writing.")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input_file} ...")
    dataset = load_dataset(
        "json",
        data_files=args.input_file,
        split="train",
        num_proc=args.num_workers
    )

    print("Shuffling dataset ...")
    shuffled_dataset = dataset.shuffle(seed=args.seed)

    print(f"Writing shuffled dataset to {args.output_file} ...")
    shuffled_dataset.to_json(
        args.output_file,
        orient="records",
        lines=True,
        num_proc=args.num_workers
    )

    print("Done.")

if __name__ == "__main__":
    main()
