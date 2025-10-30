import os
import sys
import fire
from huggingface_hub import snapshot_download, HfFolder

# Download model from Hugging Face
# HF cache is stored in /dev/shm (117GB) to avoid filling workspace or /tmp
def main(model_name: str="meta-llama/Llama-2-7b-chat-hf"):
    # Set HF cache to /dev/shm (117GB) to avoid filling workspace or /tmp
    os.environ["HF_HOME"] = "/dev/shm/huggingface"
    os.environ["HF_HUB_CACHE"] = "/dev/shm/huggingface/hub"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/dev/shm/huggingface/hub"

    # Create cache directories if they don't exist
    os.makedirs("/dev/shm/huggingface/hub", exist_ok=True)

    print(f"Downloading {model_name} to HuggingFace cache")
    print(f"HuggingFace cache location: {os.environ['HF_HOME']}")

    try:
        # Download to cache (NO local_dir - let HF manage it properly)
        # Use ignore_patterns to skip safetensors if you want only .bin files
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir="/dev/shm/huggingface/hub",
            ignore_patterns=["*.safetensors"]  # Only download .bin files
        )
        print(f"\nModel downloaded successfully to cache: {model_path}")
        print(f"You can now load it using: AutoModelForCausalLM.from_pretrained('{model_name}')")
    except Exception as e:
        print(f"\nERROR: Failed to download model: {e}")
        print("\nPlease ensure:")
        print("1. You've accepted the license at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
        print("2. Your token has access to this model")
        sys.exit(1)

if __name__ == "__main__":
    fire.Fire(main)
