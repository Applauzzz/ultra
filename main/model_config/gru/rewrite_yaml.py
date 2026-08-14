import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="FreeLong chunk size. If omitted, the YAML's `model.chunk_size` "
             "is left as-is (which means train.py will use its built-in default).",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["dump_dir"] = args.dump_dir
    cfg["name"] = args.name
    cfg["data"]["seq_len"] = args.seq_len
    if args.chunk_size is not None:
        cfg.setdefault("model", {})["chunk_size"] = args.chunk_size

    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    main()
