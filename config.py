from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RunConfig:
    # Guiding text prompt
    prompt: str = "Large white bird with yellow beak flying over the water."
    negative_prompt: str = "ugly,low resolution,blurry image,bad composition,disfigured,oversaturated"
    model_id: str = "sd1.4"
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [42])
    # Path to save all outputs to
    output_path: Path = Path('./outputs/singlefile')
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 1
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Whether to use the attention maps
    load_in_bit4: bool = False
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 15
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False
    # negative loss weight
    beta: float = 0.25
    # temperature
    temperature: float = 0.1
    # invention steps
    num_intervention_steps: int = 25
    # balance neg and pos loss
    max_intervention_steps_per_iter: int =10
    #patience
    patience: int = 1
    #length of text model
    max_seqlen: int = 256
    # modifier threshold
    modifier_threshold: float = 0.2
    # noun threshold
    noun_threshold: float = 0.3
    # noun_alpha
    noun_alpha: float = 1.0
    # modifier_alpha
    modifier_alpha: float = 1.0
    # kernel_size
    kernel_size: int=5
    # whether augment
    whether_enhance: bool = True

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
    
