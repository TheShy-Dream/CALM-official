import random
from diffusers.utils import logging
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import spacy
import torch
from diffusers import StableDiffusionPipeline
from torch.nn import functional as F

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from utils.gaussian_smoothing import GaussianSmoothing, GaussianSmoothingPatch
from utils.parser import extract_entities_only, extract_attribution_indices_with_verbs, \
    extract_attribution_indices_with_verb_root, extract_attribution_indices, start_token, end_token, get_indices, \
    align_wordpieces_indices, unify_lists, _flatten_indices, gumbel_softmax, split_indices, mapping_noun_modifier, \
    supervised_contrastive_loss, split_nested_lists
from utils.ptp_utils import AttentionStore, AttendExciteAttnProcessor

random.seed(42)

logger = logging.get_logger(__name__)

class CALM_Pipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]


        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds

    # ---------------------------------------------------------------- register
    def _register_attention_control(self, store: AttentionStore , indices=None,modifier_threshold=0.2,noun_threshold=0.3,noun_alpha=1.0,modifier_alpha=1.0,whether_enhance=False) -> None:
        procs, count = {}, 0
        for name in self.unet.attn_processors.keys():
            place = (
                "mid"
                if name.startswith("mid_block")
                else "up" if name.startswith("up_blocks") else "down"
            )
            procs[name] = AttendExciteAttnProcessor(store, place, indices=indices,modifier_threshold=modifier_threshold,noun_threshold=noun_threshold,
                                                    noun_alpha=noun_alpha,modifier_alpha=modifier_alpha,whether_enhance=whether_enhance)
            count += 1
        self.unet.set_attn_processor(procs)
        store.num_att_layers = count


    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond
        return latents 

    def _align_indices(self, prompt, spacy_pairs):
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
                    elif member.text.lower().startswith(wp.lower()) and wp.lower() != member.text.lower():  # can maybe be while loop
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

    def syntactic_extractor(self, positive_prompt:Union[str, List[str]]=None
                                ,include_entities:bool=True):
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
        #print(f"Final modifier indices collected:{modifier_indices}")
        return modifier_indices

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            run_standard_sd: bool = False,
            scale_factor: int = 20,
            num_intervention_steps: int = 25,
            beta: float = 0.1,
            temperature: float = 0.1,
            max_intervention_steps_per_iter: int = 10,
            patience: int = 3,
            kernel_size: int = 5,
            modifier_threshold: float = 0.2,
            noun_threshold: float = 0.3,
            noun_alpha: float = 1.0,
            modifier_alpha: float = 1.0,
            whether_enhance: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        self.beta = beta
        self.temperature = temperature
        self.max_intervention_steps_per_iter = max_intervention_steps_per_iter
        self.patience = patience

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        scale_range = np.linspace(1.0, 0.5, len(timesteps))
        step_sizes = scale_factor * np.sqrt(scale_range)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
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

        # 5. split embeds for refinement batch --------------------------------
        text_embeds_refine = (
            prompt_embeds[batch_size * num_images_per_prompt :]
            if do_classifier_free_guidance
            else prompt_embeds
        )

        """
        if isinstance(indices_to_alter[0], int):
            token_indices = [indices_to_alter]
        indices_batched: List[List[int]] = []
        for ind in token_indices:
            indices_batched += [ind] * num_images_per_prompt
        """
        self.modifier_noun_indices =self.syntactic_extractor(prompt, include_entities=True)

        if isinstance(attention_res, int):
            attention_res = (attention_res, attention_res)
        self.attention_store = AttentionStore(attention_res)
        original_procs = self.unet.attn_processors
        if not run_standard_sd:
            self._register_attention_control(self.attention_store,self.modifier_noun_indices,modifier_threshold=modifier_threshold,noun_threshold=noun_threshold,
                                         noun_alpha=noun_alpha,modifier_alpha=modifier_alpha,whether_enhance=whether_enhance)


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        """
        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1
        """

        # 7. denoise loop ------------------------------------------------------
        progress = self.progress_bar(total=num_inference_steps)
        for i, t in enumerate(timesteps):
            # Attend‑and‑Excite latent update ----------------------------------
            if i < num_intervention_steps and not run_standard_sd:
                latents = self._upgrade_step(
                    latents,
                    text_embeds_refine,
                    t,
                    i,
                    step_sizes[i],
                    num_intervention_steps=num_intervention_steps,
                    max_intervention_steps_per_iter=max_intervention_steps_per_iter,
                    patience=patience,
                    kernel_size=kernel_size
                )
            # diffusion step --------------------------------------------------
            model_in = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            model_in = self.scheduler.scale_model_input(model_in, t)

            noise_pred = self.unet(
                model_in,
                t,
                encoder_hidden_states=prompt_embeds,
            ).sample
            if do_classifier_free_guidance:
                uncond, cond = noise_pred.chunk(2)
                noise_pred = uncond + guidance_scale * (cond - uncond)

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

            if callback and i % callback_steps == 0:
                callback(i, t, latents)
            progress.update(1)

        # 8. Post-processing
        self.unet.set_attn_processor(original_procs)  # restore
        image = self.decode_latents(latents)
        progress.close()

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def _upgrade_step(
            self,
            latents,
            text_embeddings,
            t,
            i,
            step_size,
            num_intervention_steps,
            max_intervention_steps_per_iter,
            patience,
            kernel_size,
    ):
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            updated_latents = []
            for latent, text_embedding in zip(latents, text_embeddings):
                # Forward pass of denoising with text conditioning
                latent = latent.unsqueeze(0)
                current_patience = patience
                text_embedding = text_embedding.unsqueeze(0)
                best_loss = torch.inf
                while current_patience > 0 and max_intervention_steps_per_iter > 0:
                    self.unet(
                        latent,
                        t,
                        encoder_hidden_states=text_embedding,
                    ).sample
                    self.unet.zero_grad()
                    # Get attention maps
                    attention_maps = self.attention_store.aggregate(("up", "down", "mid")) 
                    loss = self._compute_loss(attention_maps=attention_maps,relation_tree=self.modifier_noun_indices,
                                              temperature=self.temperature,kernel_size=kernel_size,beta=self.beta)
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
                        current_patience = patience
                    else:
                        current_patience -= 1
                    max_intervention_steps_per_iter -= 1
            updated_latents.append(latent)

        latents = torch.cat(updated_latents, dim=0)

        return latents

    # ---------------------------------------------------------------- helpers
    def _compute_attention_per_index(self, maps: torch.Tensor, indices: List[int]) -> List[torch.Tensor]:
        attention_for_text = maps[:, :, 1:-1].clone()
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        indices = [index - 1 for index in indices]
        max_indices_list = []
        if len(indices)==1:
            max_indices_list.append(attention_for_text[:,:,indices[0]])
            return max_indices_list
        for i in indices:
            image = attention_for_text[:, :, i]
            height,width=image.shape[0],image.shape[1]
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
        kernel_size: int = 5,
        temperature: float = 0.1,
        beta: float = 0.2,
    ) -> torch.Tensor:

        positive_loss = self.compute_positive_loss(attention_maps, relation_tree=relation_tree,kernel_size=kernel_size)
        negative_loss = self.compute_negative_loss(attention_maps, relation_tree=relation_tree,temperature=temperature)
        return positive_loss + beta* negative_loss

    def gaussian_kernel(self,kernel_size: int, sigma: float, device=None):
        ax = torch.arange(kernel_size, device=device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def compute_positive_loss(self, attention_maps: torch.Tensor, relation_tree: list[list[int]] = None,
                              kernel_size: int = 5) -> torch.Tensor:
        noun_losses = []

        kernel = self.gaussian_kernel(kernel_size, sigma=kernel_size / 3,
                                      device=attention_maps.device).to(dtype=attention_maps.dtype)

        for sub_list in split_nested_lists(relation_tree):
            attn_value_list = self._compute_attention_per_index(attention_maps, sub_list)

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

    def compute_negative_loss(self,attention_maps: torch.Tensor, relation_tree: List[List[int]] = None, temperature: float=0.1) -> torch.Tensor:
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
