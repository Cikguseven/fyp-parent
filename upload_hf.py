

import os
from huggingface_hub import HfApi, login

login(token=os.getenv("HF_TOKEN")) # set HF_TOKEN env variable using "export HF_TOKEN="

# Enable hf_transfer for significantly faster uploads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

api = HfApi()

REPO_ID = "tokenizer-project/byte-level-bpe-38b"  # org-name (DON'T CHANGE) + new/existing repo-name, e.g. "my-username/my-repo"
LOCAL_FOLDER = "" # path to your local folder with files to upload, e.g. "./my-weights"

# Step 1: Create the repo if it doesn't exist yet
api.create_repo(
    repo_id=REPO_ID,
    repo_type="model",
    exist_ok=True,        # won't error if repo already exists
    private=False,         # set False for public
)

# Step 2: Upload with upload_large_folder — resumable, won't re-upload already-committed files
api.upload_large_folder(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=LOCAL_FOLDER,
    num_workers=4,        # parallel upload workers; tune based on your bandwidth
)

print("Upload complete!")
