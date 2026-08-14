import argparse
from ultra.model import get_config

def main():
    parser = argparse.ArgumentParser(description="Save model configuration to specified path")
    parser.add_argument("--model_name", required=True, help="Model name to retrieve configuration")
    parser.add_argument("--save_path", required=True, help="Path to save the configuration")
    args = parser.parse_args()

    # Retrieve and save configuration
    config = get_config(args.model_name)
    config.save_pretrained(args.save_path)
    print(f"Configuration saved to {args.save_path}")

if __name__ == "__main__":
    main()