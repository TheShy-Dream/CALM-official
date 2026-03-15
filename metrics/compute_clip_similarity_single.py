import json
import sys
from dataclasses import dataclass
from pathlib import Path

import clip
import numpy as np
import pyrallis
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from metrics.imagenet_utils import get_embedding_for_prompt, imagenet_templates


@dataclass
class EvalConfig:
    output_path: Path = Path("XXX")
    metrics_save_path: Path = Path("XXX")

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)


@pyrallis.wrap()
def run(config: EvalConfig):
    print("Loading CLIP model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()
    print("Done.")

    prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    print(f"Running on {len(prompts)} prompts...")

    results_per_prompt = {}
    for prompt in tqdm(prompts):
        print(f'Running on: "{prompt}"')

        image_paths = [p for p in (config.output_path / prompt).rglob('*') if p.suffix in ['.png', '.jpg']]
        images = [Image.open(p) for p in image_paths]
        image_names = [p.name for p in image_paths]
        queries = [preprocess(image).unsqueeze(0).to(device) for image in images]

        with torch.no_grad():
            text_features = get_embedding_for_prompt(model, prompt, templates=imagenet_templates)
            image_features = [model.encode_image(img).float() for img in queries]
            image_features = [feat / feat.norm(dim=-1, keepdim=True) for feat in image_features]
            similarities = [(feat @ text_features.T).item() for feat in image_features]

            results_per_prompt[prompt] = {
                'full_text': similarities,
                'image_names': image_names,
            }

    aggregated_result = aggregate_by_full_text(results_per_prompt)

    if "sd1.4" in config.output_path.name:
        file_name = config.output_path.name
    elif "sdxl1.0" in config.output_path.name:
        file_name = config.output_path.name
    elif "animals_objects" in config.output_path.name:
        file_name = "animals_objects"
    elif "objects" in config.output_path.name:
        file_name = "objects"
    elif "animals" in config.output_path.name:
        file_name = "animals"
    elif "DVMP" in config.output_path.name:
        file_name = "DVMP"
    elif "ABC" in config.output_path.name:
        file_name = "ABC"

    with open(config.metrics_save_path / f"clip_raw_{file_name}_metrics.json", 'w') as f:
        json.dump(results_per_prompt, f, indent=4, sort_keys=True)
    with open(config.metrics_save_path / f"clip_aggregated_{file_name}_metrics.json", 'w') as f:
        json.dump({'full_text_aggregation': aggregated_result}, f, indent=4, sort_keys=True)


def aggregate_by_full_text(results):
    all_scores = [sim for v in results.values() for sim in v['full_text']]
    return float(np.mean(all_scores)) if all_scores else 0.0

if __name__ == '__main__':
    run()
