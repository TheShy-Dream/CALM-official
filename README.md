# CALM: Context-Aware Latent Space Mediation for Inference-Time Unbiased Semantic Alignment in Text-to-Image Models

## Quick Start

### 1. Environment Setup

Install the required dependencies using the `requirements.txt` file located in the `environment` directory:

Bash

```
pip install -r environment/requirements.txt
```

### 2. Configuration

Modify the parameters in `config.py` to customize your generation. Key settings include the model selection, prompt, and CFG scale:

Python

```
# config.py
model_id = "sd3.5"       # Supported: sd1.4, sdxl1.0, sd3.5, flux.1
prompt = "A traditional Chinese ink painting of a dragon"
guidance_scale = 7.5     # CFG scale
beta = 0.3               # Bias correction coefficient
output_path = "./results"
```

### 3. Execution

Run the model with the following command:

Bash

```
python run.py
```

All generated images will be saved automatically in the `output_path` directory.

---

## Configuration Parameters


| **Parameter**      | **Description**                      | **Default Value**    |
| ------------------ | ------------------------------------ | -------------------- |
| **model_id**       | Selected model version               | `"sd3.5"`            |
| **prompt**         | Text prompt for image generation     | `"A traditional..."` |
| **guidance_scale** | Classifier-Free Guidance (CFG) scale | `7.5`                |
| ···                | ···                                  | ···                  |


## Supported Models


| **Model ID** | **Hugging Face Identifier**              | **Key Features**             |
| ------------ | ---------------------------------------- | ---------------------------- |
| **sd1.4**    | CompVis/stable-diffusion-v1-4            | Classic lightweight model    |
| **sdxl1.0**  | stabilityai/stable-diffusion-xl-base-1.0 | High-resolution base model   |
| **sd3.5**    | stabilityai/stable-diffusion-3.5-medium  | Enhanced latent resolution   |
| **flux.1**   | black-forest-labs/FLUX.1-dev             | State-of-the-art performance |


## Evaluation Tool

Quantitative evaluation scripts are available in the `metrics` directory to verify semantic alignment and image quality.