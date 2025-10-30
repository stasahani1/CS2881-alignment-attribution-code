import os
import fire
from huggingface_hub import snapshot_download

# Download model from Hugging Face
# HF cache is stored in /tmp/ to avoid filling workspace volume
def main(model_name: str="meta-llama/Llama-2-7b-chat-hf", local_dir: str="models/llama-2-7b-chat-hf"):
    # Set HF cache to /tmp/ to save workspace storage
    os.environ["HF_HOME"] = "/tmp/huggingface"

    print(f"Downloading {model_name} to {local_dir}")
    print(f"HuggingFace cache location: {os.environ['HF_HOME']}")

    model_path = snapshot_download(
        repo_id=model_name,
        local_dir=local_dir
    )

    print(f"Model downloaded successfully to: {model_path}")

if __name__ == "__main__":
    fire.Fire(main)
