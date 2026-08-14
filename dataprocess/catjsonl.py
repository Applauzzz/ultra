import os
import glob
import argparse
from datasets import load_dataset, concatenate_datasets

def main():
    parser = argparse.ArgumentParser(description="Batch merge jsonl files, keep only the 'text' field, and output a single large jsonl file.")
    parser.add_argument('--input_folder', '-i', required=True, help='Path to the folder containing input jsonl files')
    parser.add_argument('--output_file', '-o', required=True, help='Path to the output merged jsonl file')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_file = args.output_file

    pattern = os.path.join(input_folder, "*.jsonl")
    files = glob.glob(pattern)

    if not files:
        print(f"No jsonl files found in {input_folder}")
        return

    datasets_list = []
    total_count = 0

    for file in files:
        try:
            ds = load_dataset("json", data_files=file, split="train")
        except Exception as e:
            print(f"Warning: Failed to load file {file}, skipping. Reason: {e}")
            continue

        if "text" in ds.column_names:
            ds = ds.remove_columns([col for col in ds.column_names if col != "text"])
            datasets_list.append(ds)
            total_count += len(ds)
        else:
            print(f"Warning: File {file} does not contain 'text' field, skipping.")

    if not datasets_list:
        print("No valid datasets found. Exiting.")
        return

    import pdb
    pdb.set_trace()
    merged_dataset = concatenate_datasets(datasets_list)
    # shuffled = merged_dataset.shuffle(seed=42)
    # shuffled.save_to_disk(output_file, num_proc=64)
    merged_dataset.to_json(output_file, num_proc=40, orient="records", lines=True, force_ascii=False)
    print(f"Merging complete. Total {len(merged_dataset)} entries written to {output_file}")

if __name__ == "__main__":
    main()
