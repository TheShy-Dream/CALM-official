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
import ImageReward as IR
import pyrallis
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

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        return F.mse_loss(self.layers(x), y)

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        return F.mse_loss(self.layers(x), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


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

def calculate_aesthetic_scores(images, aes_model, clip_model_aes, device):
    aes_model.eval()
    clip_model_aes[0].eval()
    scores = []
    with torch.no_grad():
        for img in images:
            img_tensor = clip_model_aes[1](img).unsqueeze(0).to(device)
            img_feature = clip_model_aes[0].encode_image(img_tensor)
            img_emb_arr = normalized(img_feature.cpu().numpy())
            pred = aes_model(torch.from_numpy(img_emb_arr).to(device).float())
            scores.append(pred.item())
    return scores

def calculate_image_reward_scores(image_paths, prompts, ir_model):
    return [ir_model.score(prompt, str(img_path)) for img_path, prompt in zip(image_paths, prompts)]

def aggregate_by_metric(results, key):
    all_scores = [v[key] for v in results.values()]
    return float(np.mean(all_scores)) if all_scores else 0.0

@pyrallis.wrap()
def run(config: EvalConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading CLIP model for similarity (ViT-B/16)...")
    clip_model_sim = clip.load("ViT-B/16", device=device)
    clip_model_sim[0].eval()
    print("Done.")

    print("Loading CLIP model for AES (ViT-L/14)...")
    clip_model_aes = clip.load("ViT-L/14", device=device)
    clip_model_aes[0].eval()
    print("Done.")

    print("Loading AES MLP model...")
    aes_model = MLP(768).to(device)
    aes_model.load_state_dict(torch.load("XXX"))#sac+logos+ava1-l14-linearMSE.pth
    aes_model.eval()
    print("Done.")

    print("Loading ImageReward model...")
    ir_model = IR.load("ImageReward-v1.0")
    print("Done.")

    prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    print(f"Running on {len(prompts)} prompts...")

    results_per_prompt = {}

    for prompt in tqdm(prompts):
        print(f'Running on: "{prompt}"')

        image_paths = [p for p in (config.output_path / prompt).rglob('*') if
                       p.suffix.lower() in ['.png', '.jpg', '.webp']]
        image_names = [p.name for p in image_paths]

        images = [Image.open(p).convert('RGB') for p in image_paths]

        queries_sim = [clip_model_sim[1](img).unsqueeze(0).to(device) for img in images]
        queries_aes = [clip_model_aes[1](img).unsqueeze(0).to(device) for img in images]

        with torch.no_grad():
            text_features = get_embedding_for_prompt(clip_model_sim[0], prompt, templates=imagenet_templates)
            text_features = text_features.to(device)

            image_features_sim = torch.cat([clip_model_sim[0].encode_image(img).float() for img in queries_sim])
            image_features_sim = image_features_sim / image_features_sim.norm(dim=-1, keepdim=True)
            similarities = (image_features_sim @ text_features.T).squeeze().cpu().tolist()

            image_features_aes = torch.cat([clip_model_aes[0].encode_image(img).float() for img in queries_aes])
            image_features_aes = image_features_aes.cpu().numpy()
            image_features_aes = normalized(image_features_aes)
            preds = aes_model(torch.from_numpy(image_features_aes).to(device).float())
            aes_scores = preds.squeeze().cpu().tolist()

            ir_scores = [ir_model.score(prompt, str(p)) for p in image_paths]

            results_per_prompt[prompt] = {
                'full_text': similarities,
                'aes': aes_scores,
                'image_reward': ir_scores,
                'image_names': image_names,
            }

    aggregated_result = {
        'full_text_aggregation': aggregate_by_metric(results_per_prompt, 'full_text'),
        'aes_aggregation': aggregate_by_metric(results_per_prompt, 'aes'),
        'image_reward_aggregation': aggregate_by_metric(results_per_prompt, 'image_reward'),
    }

    file_name = config.output_path.name
    with open(config.metrics_save_path / f"clip_raw_{file_name}_metrics.json", 'w') as f:
        json.dump(results_per_prompt, f, indent=4, sort_keys=True)
    with open(config.metrics_save_path / f"clip_aggregated_{file_name}_metrics.json", 'w') as f:
        json.dump(aggregated_result, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    run()
