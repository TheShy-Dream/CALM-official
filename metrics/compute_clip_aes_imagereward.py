import json
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import gc

import clip
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pyrallis
import ImageReward as IR
from imagenet_utils import get_embedding_for_prompt, imagenet_templates


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


@dataclass
class EvalConfig:
    output_path: Path = Path("XXX")
    metrics_save_path: Path = Path("XXX")

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)


@pyrallis.wrap()
def run(config: EvalConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading CLIP models...")
    clip_model_sim, preprocess = clip.load("ViT-B/16", device=device)
    clip_model_sim.eval()
    clip_model_aes, preprocess2 = clip.load("ViT-L/14", device=device)
    clip_model_aes.eval()
    print("Done.")

    print("Loading AES MLP model...")
    aes_model = MLP(768).to(device)
    aes_model.load_state_dict(torch.load("XXX")) #sac+logos+ava1-l14-linearMSE.pth
    aes_model.eval()
    print("Done.")

    print("Loading ImageReward model...")
    ir_model = IR.load("ImageReward-v1.0")
    print("Done.")

    prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    print(f"Running on {len(prompts)} prompts...")

    results_per_prompt = {}

    for prompt in tqdm(prompts):
        prompt_dir = config.output_path / prompt
        image_paths = [p for p in prompt_dir.rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.webp']]
        image_names = [p.name for p in image_paths]

        images = [Image.open(p).convert('RGB') for p in image_paths]
        queries_sim = torch.cat([preprocess(img).unsqueeze(0) for img in images]).to(device)
        queries_aes = torch.cat([preprocess2(img).unsqueeze(0) for img in images]).to(device)

        with torch.no_grad():
            if ' and ' in prompt:
                prompt_parts = prompt.split(' and ')
            elif ' with ' in prompt:
                prompt_parts = prompt.split(' with ')
            else:
                prompt_parts = [prompt, prompt]  # fallback

            full_text_features = get_embedding_for_prompt(clip_model_sim, prompt, templates=imagenet_templates).to(device)
            first_half_features = get_embedding_for_prompt(clip_model_sim, prompt_parts[0], templates=imagenet_templates).to(device)
            second_half_features = get_embedding_for_prompt(clip_model_sim, prompt_parts[1], templates=imagenet_templates).to(device)

            image_features_sim = clip_model_sim.encode_image(queries_sim)
            image_features_sim = image_features_sim / image_features_sim.norm(dim=-1, keepdim=True)

            full_text_similarities = [(feat.float() @ full_text_features.float().T).item() for feat in
                                      image_features_sim]
            first_half_similarities = [(feat.float() @ first_half_features.float().T).item() for feat in
                                       image_features_sim]
            second_half_similarities = [(feat.float() @ second_half_features.float().T).item() for feat in
                                        image_features_sim]

            image_features_aes = clip_model_aes.encode_image(queries_aes)
            image_features_aes = normalized(image_features_aes.cpu().numpy())
            aes_scores = aes_model(torch.from_numpy(image_features_aes).to(device).float()).squeeze().cpu().tolist()

            ir_scores = [ir_model.score(prompt, str(p)) for p in image_paths]

            results_per_prompt[prompt] = {
                'full_text': full_text_similarities,
                'first_half': first_half_similarities,
                'second_half': second_half_similarities,
                'aes': aes_scores,
                'image_reward': ir_scores,
                'image_names': image_names,
            }

    aggregated_results = {
        'full_text_aggregation': aggregate_by_full_text(results_per_prompt),
        'min_first_second_aggregation': aggregate_by_min_half(results_per_prompt),
        'aes_aggregation': float(np.mean([v['aes'] for v in results_per_prompt.values()])),
        'image_reward_aggregation': float(np.mean([v['image_reward'] for v in results_per_prompt.values()])),
    }

    file_name = config.output_path.name
    with open(config.metrics_save_path / f"clip_raw_{file_name}_metrics.json", 'w') as f:
        json.dump(results_per_prompt, f, sort_keys=True, indent=4)
    with open(config.metrics_save_path / f"clip_aggregated_{file_name}_metrics.json", 'w') as f:
        json.dump(aggregated_results, f, sort_keys=True, indent=4)


def aggregate_by_min_half(d):
    min_per_half_res = [[min(a, b) for a, b in zip(d[p]["first_half"], d[p]["second_half"])] for p in d]
    return np.average(np.array(min_per_half_res).flatten())


def aggregate_by_full_text(d):
    full_text_res = [v['full_text'] for v in d.values()]
    return np.average(np.array(full_text_res).flatten())


if __name__ == '__main__':
    run()
