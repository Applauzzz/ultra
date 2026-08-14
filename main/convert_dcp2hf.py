import argparse
import os
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from ultra.model import load_model_from_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert DCP checkpoint to Hugging Face format")
    parser.add_argument("--model_name", required=True, help="Name of the model")
    parser.add_argument("--config_path", required=True, help="Path to the model configuration file")
    parser.add_argument("--dcp_path", required=True, help="Path to the DCP checkpoint")
    parser.add_argument("--save_path", required=True, help="Path to save the Hugging Face model")
    return parser.parse_args()

def main():
    """Main function to convert DCP checkpoint to Hugging Face format."""
    args = parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found at {args.config_path}")
    
    if not os.path.exists(args.dcp_path):
        raise FileNotFoundError(f"DCP checkpoint not found at {args.dcp_path}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize model on meta device
    with torch.device("meta"):
        model = load_model_from_config(args.model_name, args.config_path)
    
    # Move model to GPU
    model.to_empty(device="cuda")
    
    # Load DCP checkpoint
    print(f"Loading checkpoint from {args.dcp_path}")
    model_dict = {"model": get_model_state_dict(model)}
    dcp.load(model_dict, checkpoint_id=args.dcp_path)
    
    # Save model in Hugging Face format
    print(f"Saving model to {args.save_path}")
    model.save_pretrained(args.save_path)

if __name__ == "__main__":
    main()