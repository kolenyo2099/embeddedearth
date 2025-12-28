
from huggingface_hub import hf_hub_download
import json
import os

model_id = "earthflow/GeoLB-ViT-14-SigLIP-so400m-384-EO"
print(f"Fetching config for {model_id}...")

try:
    config_path = hf_hub_download(repo_id=model_id, filename="open_clip_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Config keys:", config.keys())
    print(json.dumps(config, indent=2))
    
except Exception as e:
    print(f"Failed to fetch config: {e}")
