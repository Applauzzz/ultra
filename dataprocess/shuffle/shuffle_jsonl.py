import os
import glob
import random
import argparse

def merge_jsonl_files(input_dir, merged_file):
    with open(merged_file, 'w', encoding='utf-8') as fout:
        for file in glob.glob(os.path.join(input_dir, '*.jsonl')):
            with open(file, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)

def split_file_by_lines(input_file, out_dir, n_blocks):
    """Split a file into n_blocks equal line chunks."""
    os.makedirs(out_dir, exist_ok=True)
    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    total = len(lines)
    block_size = total // n_blocks
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else total
        with open(os.path.join(out_dir, f'block_{i}.jsonl'), 'w', encoding='utf-8') as fout:
            fout.writelines(lines[start:end])
    return total

def row_shuffle(row_dir, n_blocks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_blocks):
        with open(os.path.join(row_dir, f'block_{i}.jsonl'), 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        random.shuffle(lines)
        block_size = len(lines) // n_blocks
        for j in range(n_blocks):
            start = j * block_size
            end = (j + 1) * block_size if j < n_blocks - 1 else len(lines)
            with open(os.path.join(out_dir, f'row_{i}_{j}.jsonl'), 'w', encoding='utf-8') as fout:
                fout.writelines(lines[start:end])

def col_shuffle(row_block_dir, n_blocks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for j in range(n_blocks):
        # Gather all row_i_j blocks for fixed j
        lines = []
        for i in range(n_blocks):
            with open(os.path.join(row_block_dir, f'row_{i}_{j}.jsonl'), 'r', encoding='utf-8') as fin:
                lines.extend(fin.readlines())
        random.shuffle(lines)
        block_size = len(lines) // n_blocks
        for k in range(n_blocks):
            start = k * block_size
            end = (k + 1) * block_size if k < n_blocks - 1 else len(lines)
            with open(os.path.join(out_dir, f'col_{j}_{k}.jsonl'), 'w', encoding='utf-8') as fout:
                fout.writelines(lines[start:end])

def final_merge(col_block_dir, n_blocks, num_chunks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_lines = []
    # Gather all col_j_k blocks
    for j in range(n_blocks):
        for k in range(n_blocks):
            with open(os.path.join(col_block_dir, f'col_{j}_{k}.jsonl'), 'r', encoding='utf-8') as fin:
                all_lines.extend(fin.readlines())
    random.shuffle(all_lines)
    total = len(all_lines)
    chunk_size = total // num_chunks
    for c in range(num_chunks):
        start = c * chunk_size
        end = (c + 1) * chunk_size if c < num_chunks - 1 else total
        with open(os.path.join(output_dir, f'final_chunk_{c}.jsonl'), 'w', encoding='utf-8') as fout:
            fout.writelines(all_lines[start:end])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing jsonl files')
    parser.add_argument('--work_dir', type=str, required=True, help='Working directory for intermediate files')
    parser.add_argument('--n_blocks', type=int, default=8, help='Number of splits along each axis')
    parser.add_argument('--num_chunks', type=int, default=8, help='Number of final output chunks')
    args = parser.parse_args()

    merged_file = os.path.join(args.work_dir, 'all_data.jsonl')
    row_dir = os.path.join(args.work_dir, 'row_blocks')
    row_shuffle_dir = os.path.join(args.work_dir, 'row_shuffled')
    col_shuffle_dir = os.path.join(args.work_dir, 'col_shuffled')
    output_dir = os.path.join(args.work_dir, 'final_chunks')

    print("Merging all JSONL files...")
    merge_jsonl_files(args.input_dir, merged_file)

    print("Splitting merged file into row blocks...")
    split_file_by_lines(merged_file, row_dir, args.n_blocks)

    print("Row shuffle and split...")
    row_shuffle(row_dir, args.n_blocks, row_shuffle_dir)

    print("Column shuffle and split...")
    col_shuffle(row_shuffle_dir, args.n_blocks, col_shuffle_dir)

    print("Final merge and output...")
    final_merge(col_shuffle_dir, args.n_blocks, args.num_chunks, output_dir)

    print("Done.")

if __name__ == "__main__":
    main()
