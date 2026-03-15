import json
import os
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pyrallis
from open_clip import create_model, get_tokenizer
from transformers import CLIPProcessor, CLIPModel

class HPSScorer(nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        
        base_model_path = "XXX" #CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin
        hps_checkpoint_path = "XXX" #HPS_v2.1_compressed.pt

        self.model = create_model(
            "ViT-H-14",
            pretrained=base_model_path,
            precision='fp32' if self.dtype == torch.float32 else 'fp16',
            device=self.device
        )

        if os.path.exists(hps_checkpoint_path):
            checkpoint = torch.load(hps_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
        
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.model.eval().requires_grad_(False)

        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073],device=self.device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711],device=self.device).view(1, 3, 1, 1))

    def preprocess(self, x):
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def score_batch(self, prompt, images_tensor):
        tokens = self.tokenizer([prompt]).to(self.device)
        txt_feat = self.model.encode_text(tokens)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        img_input = self.preprocess(images_tensor.to(self.device, dtype=self.dtype))
        img_feat = self.model.encode_image(img_input)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        scores = (img_feat * txt_feat).sum(dim=-1)
        return scores.cpu().tolist()


class PickScoreScorer(nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        processor_path = "XXX" #CLIP-ViT-H-14-laion2B-s32B-b79K
        model_path = "XXX" #PickScore_v1
        
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device, dtype=dtype)

    @torch.no_grad()
    def score_batch(self, prompt, images_pil):
        inputs = self.processor(
            text=[prompt], 
            images=images_pil, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        img_feats = self.model.get_image_features(pixel_values=inputs['pixel_values'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)

        txt_feats = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (txt_feats * img_feats).sum(dim=-1)
        return (scores / 26).cpu().tolist()


@dataclass
class EvalConfig:
    output_path: Path = Path("XXX")
    metrics_save_path: Path = Path("XXX")
    device: str = "cuda"

    def __post_init__(self):
        self.metrics_save_path.mkdir(parents=True, exist_ok=True)

@pyrallis.wrap()
def run(config: EvalConfig):
    device = config.device
    
    hps_scorer = HPSScorer(device=device)
    pick_scorer = PickScoreScorer(device=device)
    
    to_tensor = transforms.ToTensor()
    prompts = [p.name for p in config.output_path.glob("*") if p.is_dir()]
    
    results_per_prompt = {}

    for prompt in tqdm(prompts, desc="Evaluating"):
        img_paths = [p for p in (config.output_path / prompt).rglob('*') if 
                     p.suffix.lower() in ['.png', '.jpg', '.webp']]
        if not img_paths: continue

        images_pil = [Image.open(p).convert('RGB') for p in img_paths]
        
        images_tensor = torch.stack([to_tensor(img) for img in images_pil])

        hps_scores = hps_scorer.score_batch(prompt, images_tensor)
        pick_scores = pick_scorer.score_batch(prompt, images_pil)

        results_per_prompt[prompt] = {
            'hps_v2': hps_scores,
            'pick_score': pick_scores,
            'image_names': [p.name for p in img_paths],
        }

    def get_avg(key):
        vals = [s for res in results_per_prompt.values() for s in res[key]]
        return float(sum(vals)/len(vals)) if vals else 0.0

    aggregated = {
        'avg_hps_v2': get_avg('hps_v2'),
        'avg_pick_score': get_avg('pick_score')
    }

    file_tag = config.output_path.name
    with open(config.metrics_save_path / f"raw_{file_tag}_hps_pickscore.json", 'w') as f:
        json.dump(results_per_prompt, f, indent=4)
    with open(config.metrics_save_path / f"agg_{file_tag}_hps_pickscore .json", 'w') as f:
        json.dump(aggregated, f, indent=4)

if __name__ == '__main__':
    run()