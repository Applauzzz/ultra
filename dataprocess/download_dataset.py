
# NOTE this is usesd for fking tt merlin devbox
import socket
_original_getaddrinfo = socket.getaddrinfo
def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):

    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = _ipv4_only_getaddrinfo

import argparse
import pdb
from datasets import load_dataset
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_name', type=str, required=True, help='HuggingFace dataset repo name')
    parser.add_argument('--name', type=str, required=True, help='HuggingFace dataset repo name')
    args = parser.parse_args()

    dataset = load_dataset(
        args.repo_name,
        name=args.name,
        keep_in_memory=False,
        num_proc=64,
    )
if __name__ == "__main__":
    main()