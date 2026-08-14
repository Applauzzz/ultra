import os
import json
import argparse
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
def write_jsonl_chunk(dataset, start, end, out_path):
    """
    Write a chunk of the dataset to a JSONL file.
    """
    with open(out_path, 'w', encoding='utf-8') as f:
        for i in range(start, end):
            record = dataset[i]
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def dataset_to_jsonl_threadpool(dataset, out_dir, n_chunks=8):
    """
    Split the dataset into n_chunks and write each chunk to a JSONL file in parallel using threads.
    """
    os.makedirs(out_dir, exist_ok=True)
    total = len(dataset)
    chunk_size = total // n_chunks
    futures = []
    with ThreadPoolExecutor(max_workers=n_chunks) as executor:
        for i in range(n_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_chunks - 1 else total
            out_path = os.path.join(out_dir, f'data_chunk_{i}.jsonl')
            futures.append(
                executor.submit(write_jsonl_chunk, dataset, start, end, out_path)
            )
        # Optionally, wait for all threads to finish and raise exceptions if any
        for future in futures:
            future.result()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Huggingface dataset to JSONL files in parallel using threads.")
    parser.add_argument("--repo_name", type=str, required=True, help="Huggingface dataset name, e.g., HuggingFaceFW/fineweb-edu")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for JSONL files")
    parser.add_argument("--num_chunks", type=int, default=64, help="Number of output files/threads (default: 8)")
    parser.add_argument("--num_proc", type=int, default=40, help="Number of processes for loading dataset (default: 1)")
    
    args = parser.parse_args()
    dataset = load_dataset(args.repo_name, split=args.split, num_proc=args.num_proc)
    dataset_to_jsonl_threadpool(dataset, args.out_dir, n_chunks=args.num_chunks)