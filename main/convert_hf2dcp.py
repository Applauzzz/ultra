import argparse
import os
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from ultra.model import load_model_from_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert DCP checkpoint to Hugging Face format")
    parser.add_argument("--model_name", required=True, help="Name of the model")
    parser.add_argument("--hf_path", required=True, help="Path to the DCP checkpoint")
    parser.add_argument("--save_path", required=True, help="Path to save the Hugging Face model")
    return parser.parse_args()

def main():
    """Main function to convert DCP checkpoint to Hugging Face format."""
    args = parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.hf_path):
        raise FileNotFoundError(f"Config file not found at {args.hf_path}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # load the model
    model = load_model_from_path(args.model_name, args.hf_path)
    
    # Load DCP checkpoint
    print(f"Saving checkpoint to {args.save_path}")
    model_dict = {"model": get_model_state_dict(model)}
    dcp.save(model_dict, checkpoint_id=args.save_path)

if __name__ == "__main__":
    main()