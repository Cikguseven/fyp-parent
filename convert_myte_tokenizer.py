import json
import os
from myte.src.myt5.myt5_tokenizer import MyT5Tokenizer
import shutil

# 1. Initialize the tokenizer
# Ensure the map files exist at these paths or update the paths
tokenizer = MyT5Tokenizer(
    decompose_map="/scratch/Projects/CFP-01/CFP01-CF-060/kieron/8192_myte_SEA_1m/decompose.json",
    merge_map="/scratch/Projects/CFP-01/CFP01-CF-060/kieron/8192_myte_SEA_1m/morf_map_mc4_8192.json",
)

# 3. Save the tokenizer to a directory
output_dir = "/scratch/Projects/CFP-01/CFP01-CF-060/kieron/8192_myte_SEA_1m"
tokenizer.save_pretrained(output_dir)

# 4. Manually update tokenizer_config.json to include auto_map
# This tells generic AutoTokenizer to look for your python file.
config_path = os.path.join(output_dir, "tokenizer_config.json")

with open(config_path, "r") as f:
    config = json.load(f)

config["auto_map"] = {
    "AutoTokenizer": [
        "myt5_tokenizer.MyT5Tokenizer",
        None  # None indicates there is no "Fast" (Rust) version available
    ]
}

# 5. Copy the python source file into the output directory
# This is crucial so the model is self-contained
shutil.copy("/scratch/Projects/CFP-01/CFP01-CF-060/kieron/myte/src/myt5/myt5_tokenizer.py", os.path.join(output_dir, "myt5_tokenizer.py"))

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Tokenizer saved to {output_dir}. You can now load it using AutoTokenizer.")
