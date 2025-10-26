import fire
from huggingface_hub import snapshot_download

# Download model from Hugging Face
def main(model_name: str="meta-llama/Llama-2-7b-chat-hf", local_dir: str="models/llama-2-7b-chat-hf"):
    model_path = snapshot_download(
        repo_id=model_name,
        local_dir=local_dir
    )

if __name__ == "__main__":
    fire.Fire(main)
