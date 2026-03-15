import pprint
from typing import List

import pyrallis
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, BitsAndBytesConfig, SD3Transformer2DModel, UNet2DConditionModel, \
    PNDMScheduler, StableDiffusion3Pipeline, StableDiffusionXLPipeline,FluxTransformer2DModel

from config import RunConfig
from pipeline_CALM_SD import CALM_Pipeline
from pipeline_CALM_SDXL import CALM_XLPipeline
from pipeline_CALM_SD3 import CALM_SD3Pipeline
from pipeline_CALM_Flux import CALM_FluxPipeline
from utils import ptp_utils, vis_utils

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if config.model_id == "sd2.1":
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
        stable = CALM_Pipeline.from_pretrained(stable_diffusion_version).to(device)
    elif config.model_id == "sd1.5":
        stable_diffusion_version = "stabilityai/stable-diffusion-v1-5"
        stable = CALM_Pipeline.from_pretrained(stable_diffusion_version).to(device)
    elif config.model_id == "sdxl1.0":
        stable_diffusion_version = "stabilityai/stable-diffusion-xl-base-1.0"
        stable = CALM_XLPipeline.from_pretrained(stable_diffusion_version,torch_dtype=torch.bfloat16).to(device)
    elif config.model_id == "sd3.5":
        stable_diffusion_version = "stabilityai/stable-diffusion-3.5-medium"
        stable = CALM_SD3Pipeline.from_pretrained(stable_diffusion_version).to(device)
        stable.transformer.enable_gradient_checkpointing()
        stable.enable_attention_slicing()
    elif config.model_id == "sd1.4":
        stable_diffusion_version = "/data0/cjl/StableDiffusionBackbone/stable-diffusion-v1-4/"
        stable = CALM_Pipeline.from_pretrained(stable_diffusion_version).to(device)
        stable.unet.enable_gradient_checkpointing()
        stable.enable_attention_slicing()
    elif config.model_id == "flux1.0":
        stable_diffusion_version = "black-forest-labs/FLUX.1-dev"
        stable = CALM_FluxPipeline.from_pretrained(stable_diffusion_version).to(device)
        stable.transformer.enable_gradient_checkpointing()
        stable.enable_attention_slicing()
    return stable

def run_on_prompt(prompt: List[str],
                  negative_prompt: List[str],
                  model: StableDiffusionPipeline,
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if config.model_id in ["sd2.1", "sd1.5","sd1.4"]:
        outputs = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            attention_res=config.attention_res,
            guidance_scale=config.guidance_scale,
            generator=seed,
            num_inference_steps=config.n_inference_steps,
            run_standard_sd=config.run_standard_sd,
            scale_factor=config.scale_factor,
            num_intervention_steps=config.num_intervention_steps,
            beta=config.beta,
            temperature=config.temperature,
            max_intervention_steps_per_iter=config.max_intervention_steps_per_iter,
            patience=config.patience,
            kernel_size= config.kernel_size,
            modifier_threshold= config.modifier_threshold,
            noun_threshold= config.noun_threshold,
            noun_alpha = config.noun_alpha,
            modifier_alpha= config.modifier_alpha,
            whether_enhance=config.whether_enhance
        )
    elif config.model_id == "sdxl1.0":
        outputs = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.n_inference_steps,
            generator=seed,
            run_standard_sd=config.run_standard_sd,
            num_intervention_steps=config.num_intervention_steps,
            scale_factor=config.scale_factor,
            beta=config.beta,
            temperature=config.temperature,
            max_intervention_steps_per_iter=config.max_intervention_steps_per_iter,
            patience=config.patience,
            kernel_size=config.kernel_size,
            modifier_threshold=config.modifier_threshold,
            noun_threshold=config.noun_threshold,
            noun_alpha=config.noun_alpha,
            modifier_alpha=config.modifier_alpha,
            whether_enhance=config.whether_enhance
        )
    elif config.model_id == "sd3.5":
        outputs = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.n_inference_steps,
            generator=seed,
            run_standard_sd=config.run_standard_sd,
            num_intervention_steps=config.num_intervention_steps,
            scale_factor=config.scale_factor,
            beta=config.beta,
            temperature=config.temperature,
            max_intervention_steps_per_iter=config.max_intervention_steps_per_iter,
            patience=config.patience,
            max_sequence_length=config.max_seqlen,
            kernel_size=config.kernel_size,
            modifier_threshold=config.modifier_threshold,
            noun_threshold=config.noun_threshold,
            noun_alpha=config.noun_alpha,
            modifier_alpha=config.modifier_alpha,
            whether_enhance=config.whether_enhance
        )
    elif config.model_id == "flux1.0":
        outputs = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.n_inference_steps,
            generator=seed,
            run_standard_sd=config.run_standard_sd,
            num_intervention_steps=config.num_intervention_steps,
            scale_factor=config.scale_factor,
            beta=config.beta,
            temperature=config.temperature,
            max_intervention_steps_per_iter=config.max_intervention_steps_per_iter,
            patience=config.patience,
            max_sequence_length=config.max_seqlen,
            kernel_size=config.kernel_size,
            modifier_threshold=config.modifier_threshold,
            noun_threshold=config.noun_threshold,
            noun_alpha=config.noun_alpha,
            modifier_alpha=config.modifier_alpha,
            whether_enhance=config.whether_enhance
        )
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)
    images = []
    for seed in config.seeds:
        print(f"Seed: {seed}")
        g = torch.Generator('cuda').manual_seed(seed)
        image = run_on_prompt(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            model=stable,
            seed=g,
            config=config
        )
        prompt_output_path = config.output_path / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f'{seed}.png')
        images.append(image)

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(images)
    joined_image.save(config.output_path / f'{config.prompt}.png')


if __name__ == '__main__':
    main()
