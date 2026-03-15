import inspect
import random
from typing import Union, Optional, List, Dict, Any, Callable

import numpy as np
import spacy
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.utils import replace_example_docstring, is_torch_xla_available, logging

from utils.parser import extract_attribution_indices, extract_attribution_indices_with_verb_root, \
    extract_attribution_indices_with_verbs, extract_entities_only, unify_lists, align_wordpieces_indices, start_token, \
    end_token, get_indices, split_nested_lists, split_indices, _flatten_indices
from utils.ptp_utils import AttendExciteAttnProcessorDiT, AttentionStore, AttentionStoreDiT
from utils.gaussian_smoothing import GaussianSmoothingPatch
from torch.nn import functional as F
from utils.parser import supervised_contrastive_loss

random.seed(42)

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""

# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class CALM_SD3Pipeline(StableDiffusion3Pipeline):
    # ---------------------------------------------------------------- register
    def _register_attention_control(self, store: AttentionStoreDiT,indices=None,modifier_threshold=0.2,noun_threshold=0.3,noun_alpha=1.0,modifier_alpha=1.0,whether_enhance=False) -> None:
        procs, count = {}, 0
        for name in self.transformer.attn_processors.keys():
            if "attn2" in name:
                procs[name] = self.transformer.attn_processors[name]
            else:
                count += 1
                if count in [20,21,22,23,24]:
                    procs[name] = AttendExciteAttnProcessorDiT(store,indices=indices,modifier_threshold=modifier_threshold,noun_threshold=noun_threshold,
                                                               noun_alpha=noun_alpha,modifier_alpha=modifier_alpha,whether_enhance=whether_enhance)
                else:
                    procs[name] = self.transformer.attn_processors[name]

        self.transformer.set_attn_processor(procs)
        store.num_att_layers = 5

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_sequence_length: int = 256,
    ):
        return super().encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length
        )

    def syntactic_extractor(self, positive_prompt: Union[str, List[str]] = None
                            , include_entities: bool = True):
        self.parser = spacy.load("en_core_web_lg")
        self.subtrees_indices = None
        self.doc = None
        self.include_entities = include_entities
        self.doc = self.parser(positive_prompt)
        modifier_indices = []
        # extract standard attribution indices
        modifier_sets_1 = extract_attribution_indices(self.doc)
        modifier_indices_1 = self._align_indices(positive_prompt, modifier_sets_1)
        if modifier_indices_1:
            modifier_indices.append(modifier_indices_1)

        # extract attribution indices with verbs in between
        modifier_sets_2 = extract_attribution_indices_with_verb_root(self.doc)
        modifier_indices_2 = self._align_indices(positive_prompt, modifier_sets_2)
        if modifier_indices_2:
            modifier_indices.append(modifier_indices_2)

        modifier_sets_3 = extract_attribution_indices_with_verbs(self.doc)
        modifier_indices_3 = self._align_indices(positive_prompt, modifier_sets_3)
        if modifier_indices_3:
            modifier_indices.append(modifier_indices_3)

        # entities only
        if self.include_entities:
            modifier_sets_4 = extract_entities_only(self.doc)
            modifier_indices_4 = self._align_indices(positive_prompt, modifier_sets_4)
            modifier_indices.append(modifier_indices_4)

        # make sure there are no duplicates
        modifier_indices = unify_lists(modifier_indices)  
        # print(f"Final modifier indices collected:{modifier_indices}")
        return modifier_indices

    def _align_indices(self, prompt,
                       spacy_pairs): 
        wordpieces2indices = get_indices(self.tokenizer, prompt)
        paired_indices = []
        collected_spacy_indices = (
            set()
        )  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)

        for pair in spacy_pairs:
            curr_collected_wp_indices = (
                []
            )  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
            for member in pair:
                for idx, wp in wordpieces2indices.items():
                    if wp in [start_token, end_token]:
                        continue

                    wp = wp.replace("</w>", "")
                    if member.text.lower() == wp.lower():
                        if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                            curr_collected_wp_indices.append(idx)
                            break
                    # take care of wordpieces that are split up
                    elif member.text.lower().startswith(
                            wp.lower()) and wp.lower() != member.text.lower():  # can maybe be while loop
                        wp_indices = align_wordpieces_indices(
                            wordpieces2indices, idx, member.text
                        )
                        # check if all wp_indices are not already in collected_spacy_indices
                        if wp_indices and (wp_indices not in curr_collected_wp_indices) and all(
                                [wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                            curr_collected_wp_indices.append(wp_indices)
                            break

            for collected_idx in curr_collected_wp_indices:
                if isinstance(collected_idx, list):
                    for idx in collected_idx:
                        collected_spacy_indices.add(idx)
                else:
                    collected_spacy_indices.add(collected_idx)

            if curr_collected_wp_indices:
                paired_indices.append(curr_collected_wp_indices)
            else:
                print(f"No wordpieces were aligned for {pair} in _align_indices")

        return paired_indices

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        scale_factor: int = 20,
        run_standard_sd: bool = False,
        num_intervention_steps: int = 25,
        beta: float = 0.5,
        temperature: float = 0.1,
        max_intervention_steps_per_iter: int = 10,
        patience: int = 3,
        kernel_size: int = 5,
        modifier_threshold: float = 0.2,
        noun_threshold: float = 0.3,
        noun_alpha: float = 1.0,
        modifier_alpha: float = 1.0,
        whether_enhance: bool = True,
    ):
        r"""
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in all the text-encoders.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            Examples:
    """
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        self.beta = beta
        self.temperature = temperature

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds = prompt_embeds
                original_pooled_prompt_embeds = pooled_prompt_embeds
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        if not run_standard_sd:
            #prepare refine for intervention
            text_embeds_refine = (
                prompt_embeds[batch_size * num_images_per_prompt:]
                if self.do_classifier_free_guidance
                else prompt_embeds
            )
            add_txt_refine = (
                pooled_prompt_embeds[batch_size * num_images_per_prompt:]
                if self.do_classifier_free_guidance
                else pooled_prompt_embeds
            )

            self.attention_store = AttentionStoreDiT()
            self.modifier_noun_indices = self.syntactic_extractor(prompt, include_entities=True)
            self._register_attention_control(self.attention_store,self.modifier_noun_indices,modifier_threshold=modifier_threshold,noun_threshold=noun_threshold,
                                         noun_alpha=noun_alpha,modifier_alpha=modifier_alpha,whether_enhance=whether_enhance)


        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        scale_range = np.linspace(1.0, 0.5, len(timesteps))
        step_sizes = scale_factor * np.sqrt(scale_range)

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # Attend‑and‑Excite latent update ----------------------------------
                if i < num_intervention_steps and not run_standard_sd:
                    latents = self._upgrade_step(
                        latents,
                        text_embeds_refine,
                        t,
                        i,
                        step_sizes[i],
                        add_txt_refine,
                        num_intervention_steps=num_intervention_steps,
                        max_intervention_steps_per_iter=max_intervention_steps_per_iter,
                        patience=patience,
                    )

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    should_skip_layers = (
                        True
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep = t.expand(latents.shape[0])
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                        )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)

    def _upgrade_step(
            self,
            latents,
            text_embeddings,
            t,
            i,
            step_size,
            add_txt_refine,
            num_intervention_steps,
            max_intervention_steps_per_iter,
            patience,
    ):
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            updated_latents = []
            for latent, text_embedding in zip(latents, text_embeddings):
                # Forward pass of denoising with text conditioning
                latent = latent.unsqueeze(0)
                timestep = t.expand(latents.shape[0])
                text_embedding = text_embedding.unsqueeze(0)

                for param in self.transformer.parameters():
                    param.requires_grad = False
                #
                # for param in self.transformer.transformer_blocks[23].parameters():
                #     param.requires_grad = True

                best_loss = torch.inf
                while patience > 0 and max_intervention_steps_per_iter > 0:
                    self.transformer(
                        hidden_states=latent,
                        timestep=timestep,
                        encoder_hidden_states=text_embedding,
                        pooled_projections=add_txt_refine,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    self.transformer.zero_grad()
                    # Get attention maps
                    clip_attention_maps = self.attention_store.aggregate(["clip_text"])
                    T5_attention_maps = self.attention_store.aggregate(["T5_text"])
                    loss_clip = self._compute_loss(attention_maps=clip_attention_maps, relation_tree=self.modifier_noun_indices,
                                              temperature=self.temperature, beta=self.beta,is_T5=False)
                    loss_T5 = self._compute_loss(attention_maps=T5_attention_maps, relation_tree=self.modifier_noun_indices,
                                              temperature=self.temperature, beta=self.beta,is_T5=True)
                    loss = (loss_clip + loss_T5)/2
                    # Perform gradient update
                    if i < num_intervention_steps:
                        if loss != 0:
                            latent = self._update_latent(
                                latents=latent, loss=loss, step_size=step_size
                            )
                        print(f"Iteration {i} | Loss: {loss:0.4f}")
                        logger.info(f"Iteration {i} | Loss: {loss:0.4f}")

                    if loss < best_loss:
                        best_loss = loss
                        patience = 5
                    else:
                        patience -= 1
                    max_intervention_steps_per_iter -= 1

            updated_latents.append(latent)

        latents = torch.cat(updated_latents, dim=0)

        return latents

    def _compute_attention_per_index(self, maps: torch.Tensor, indices: List[int], is_T5: bool) -> List[torch.Tensor]:
        if not is_T5:
            attention_for_text = maps[:, :, 1:-1].clone()
        else:
            attention_for_text = maps.clone()
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        indices = [index - 1 for index in indices]
        max_indices_list = []
        if len(indices) == 1:
            max_indices_list.append(attention_for_text[:, :, indices[0]])
            return max_indices_list
        for i in indices:
            image = attention_for_text[:, :, i]
            height, width = image.shape[0], image.shape[1]
            smoothing = GaussianSmoothingPatch().to(maps.device)
            _input = F.pad(
                image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
            )
            image = smoothing(_input).squeeze(0).squeeze(0)
            max_indices_list.append(image)
        return max_indices_list

    def _compute_loss(
        self,
        attention_maps: torch.Tensor,
        relation_tree: List[List[int]] = None,
        temperature: float = 0.1,
        kernel_size: int = 5,
        beta: float = 0.2,
        is_T5: bool = False
    ) -> torch.Tensor:
        positive_loss = self.compute_positive_loss(attention_maps, relation_tree=relation_tree,kernel_size=kernel_size,is_T5=is_T5)
        negative_loss = self.compute_negative_loss(attention_maps, relation_tree=relation_tree,temperature=temperature,is_T5=is_T5)
        return positive_loss + beta * negative_loss

    def gaussian_kernel(self, kernel_size: int, sigma: float, device=None):
        """生成一个 2D 高斯卷积核"""
        ax = torch.arange(kernel_size, device=device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def compute_positive_loss(self, attention_maps: torch.Tensor, relation_tree: list[list[int]] = None,
                              kernel_size: int = 5, is_T5=False) -> torch.Tensor:
        noun_losses = []

        kernel = self.gaussian_kernel(kernel_size, sigma=kernel_size / 3,
                                      device=attention_maps.device).to(dtype=attention_maps.dtype)

        for sub_list in split_nested_lists(relation_tree):
            attn_value_list = self._compute_attention_per_index(attention_maps, sub_list, is_T5 = is_T5)

            if len(sub_list) > 1:
                noun_attn = attn_value_list[-1].view(1, 1, -1)  # (1,1,N)
                n = int(noun_attn.numel() ** 0.5)
                noun_2d = noun_attn.view(1, 1, n, n)
                prod_conv_list = []
                noun_sq = noun_2d ** 2
                for v in attn_value_list[:-1]:
                    mod_attn = v.view(1, 1, n, n)
                    prod = noun_sq * mod_attn  
                    prod_conv = F.conv2d(prod, kernel,
                                         stride=1, padding=kernel_size // 2)
                    prod_conv_list.append(prod_conv)

                if prod_conv_list:
                    mod_agg = torch.stack(prod_conv_list).mean(dim=0)
                    co_occurrence = mod_agg.max()
                    noun_losses.append(1 - co_occurrence)

            else:
                noun_attn = attn_value_list[0].view(1, 1, -1)
                n = int(noun_attn.numel() ** 0.5)
                noun_2d = noun_attn.view(1, 1, n, n)
                noun_conv = F.conv2d(noun_2d ** 2, kernel,
                                     stride=1, padding=kernel_size // 2)
                noun_focus = noun_conv.max()
                noun_losses.append(1 - noun_focus)

        total_loss = torch.stack(noun_losses).mean()
        return total_loss

    def compute_negative_loss(self,attention_maps: torch.Tensor, relation_tree: List[List[int]] = None, temperature: float=0.1, beta: float=0.2) -> torch.Tensor:
        noun_list,modifier_list = split_indices(split_nested_lists(relation_tree))
        noun_list,modifier_list = _flatten_indices(noun_list), _flatten_indices(modifier_list)
        feature_list = []
        label_list = []
        for label_idx, sub_list in enumerate(split_nested_lists(relation_tree)):
            attn_value_list = self._compute_attention_per_index(attention_maps, sub_list)
            feature_list.extend(attn_value_list)
            label_list.extend([label_idx] * len(attn_value_list) )
        feature_stacked = torch.stack(feature_list)
        label_stacked = torch.tensor(label_list).to(feature_stacked.device)
        modifier_scl_loss = supervised_contrastive_loss(feature_stacked.view(feature_stacked.shape[0],-1),label_stacked,temperature=temperature)
        #negative_loss = noun_loss + beta * modifier_scl_loss
        return modifier_scl_loss/len(label_list)

    @staticmethod
    def _update_latent(
            latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents
