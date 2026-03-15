from copy import deepcopy
import os
import numpy as np
import spacy
import torch
from diffusers import FluxPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps, XLA_AVAILABLE
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from typing import Union, Optional, List, Dict, Any, Callable
from diffusers.utils import replace_example_docstring, is_torch_xla_available, logging
from utils.gaussian_smoothing import GaussianSmoothingPatch
from utils.parser import extract_attribution_indices, extract_attribution_indices_with_verb_root, \
    extract_attribution_indices_with_verbs, extract_entities_only, unify_lists, get_indices, start_token, end_token, \
    align_wordpieces_indices, supervised_contrastive_loss, split_nested_lists, split_indices, _flatten_indices
from utils.ptp_utils import AttendExciteAttentionProcessorFlux, AttentionStoreFlux

from torch.nn import functional as F


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""


class CALM_FluxPipeline(FluxPipeline):
    r"""
    The Flux pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """
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

    def _register_attention_control(self, store: AttentionStoreFlux,indices=None,modifier_threshold=0.2,noun_threshold=0.3,noun_alpha=1.0,modifier_alpha=1.0,whether_enhance=False,txt_length=512) -> None:
        procs, count = {}, 0
        for name in self.transformer.attn_processors.keys():
            if not "single" in name:
                procs[name] = self.transformer.attn_processors[name]
            else:
                count += 1
                if count in [34,35,36,37,38]:
                    procs[name] = AttendExciteAttentionProcessorFlux(store,indices=indices,modifier_threshold=modifier_threshold,noun_threshold=noun_threshold,
                                                               noun_alpha=noun_alpha,modifier_alpha=modifier_alpha,whether_enhance=whether_enhance,txt_length=txt_length)
                else:
                    procs[name] = self.transformer.attn_processors[name]

        self.transformer.set_attn_processor(procs)
        store.num_att_layers = 5

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
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
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                True classifier-free guidance (guidance scale) is enabled when `true_cfg_scale` > 1 and
                `negative_prompt` is provided.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Embedded guiddance scale is enabled by setting `guidance_scale` > 1. Higher `guidance_scale` encourages
                a model to generate images more aligned with `prompt` at the expense of lower image quality.

                Guidance-distilled models approximates true classifer-free guidance for `guidance_scale` > 1. Refer to
                the [paper](https://huggingface.co/papers/2210.03142) to learn more.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self.beta = beta
        self.temperature = temperature
        self.max_intervention_steps_per_iter = max_intervention_steps_per_iter
        self.patience = patience
        self.kernel_size = kernel_size
        self.modifier_threshold = modifier_threshold
        self.noun_threshold = noun_threshold
        self.noun_alpha = noun_alpha
        self.modifier_alpha = modifier_alpha
        self.whether_enhance = whether_enhance

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

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
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        if not run_standard_sd:
            self.attention_store = AttentionStoreFlux()
            self.modifier_noun_indices = self.syntactic_extractor(prompt, include_entities=True)
            self._register_attention_control(self.attention_store, self.modifier_noun_indices,
                                             modifier_threshold=modifier_threshold, noun_threshold=noun_threshold,
                                             noun_alpha=noun_alpha, modifier_alpha=modifier_alpha,
                                             whether_enhance=whether_enhance, txt_length=max_sequence_length)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
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
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        
        scale_range = np.linspace(1.0, 0.5, len(timesteps))
        step_sizes = scale_factor * np.sqrt(scale_range)
        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if not run_standard_sd and i < num_intervention_steps:
                    current_step_size = step_sizes[i]
                    
                    latents = self._upgrade_step(
                        latents=latents,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        latent_image_ids=latent_image_ids,
                        guidance=guidance,
                        t=t,
                        i=i,
                        step_size=current_step_size,
                        num_intervention_steps=num_intervention_steps,
                        max_intervention_steps_per_iter=max_intervention_steps_per_iter,
                        patience=patience
                    )

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)



                #with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=negative_pooled_prompt_embeds,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

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

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


    def _upgrade_step(
            self,
            latents,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            latent_image_ids,
            guidance,
            t,
            i,
            step_size,
            num_intervention_steps,
            max_intervention_steps_per_iter,
            patience,
    ):
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            
            best_loss = torch.inf
            curr_patience = patience
            curr_iters = max_intervention_steps_per_iter

            while curr_patience > 0 and curr_iters > 0:
                self.transformer(
                    hidden_states=latents,
                    timestep=(t / 1000).expand(latents.shape[0]).to(latents.dtype),
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                self.transformer.zero_grad()

                attn_maps = self.attention_store.aggregate(["text"]) 

                loss = self._compute_loss(
                    attention_maps=attn_maps, 
                    relation_tree=self.modifier_noun_indices,
                    temperature=self.temperature, 
                    beta=self.beta
                )

                if i < num_intervention_steps:
                    if loss != 0:
                        latents = self._update_latent(latents=latents, loss=loss, step_size=step_size)
                    # print(f"Flux Intervene Step {i} | Iter {max_intervention_steps_per_iter - curr_iters} | Loss: {loss:0.4f}")

                if loss < best_loss:
                    best_loss = loss
                    curr_patience = patience
                else:
                    curr_patience -= 1
                curr_iters -= 1

        return latents.detach()

    def _update_latent(self, latents, loss, step_size):
        grad = torch.autograd.grad(loss, [latents], retain_graph=True)[0]
        latents = latents - step_size * grad
        return latents

    
    def _compute_attention_per_index(self, maps: torch.Tensor, indices: List[int]) -> List[torch.Tensor]:
        attention_for_text = maps[:, :, :].clone() 
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

    """
    @staticmethod
    def _compute_loss(max_vals: List[torch.Tensor]) -> torch.Tensor:
        losses = [max(0, 1.0 - curr_max) for curr_max in max_vals]
        loss = max(losses)
        return loss
    """

    def _compute_loss(
        self,
        attention_maps: torch.Tensor,
        relation_tree: List[List[int]] = None,
        temperature: float = 0.1,
        kernel_size: int = 5,
        beta: float = 0.2,
    ) -> torch.Tensor:
        positive_loss = self.compute_positive_loss(attention_maps, relation_tree=relation_tree,kernel_size=kernel_size)
        negative_loss = self.compute_negative_loss(attention_maps, relation_tree=relation_tree,temperature=temperature)
        return positive_loss + beta * negative_loss

    def gaussian_kernel(self, kernel_size: int, sigma: float, device=None):
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

    def compute_negative_loss(self,attention_maps: torch.Tensor, relation_tree: List[List[int]] = None, temperature: float=0.1, beta: float=0.2) -> torch.Tensor:
        noun_list,modifier_list = split_indices(split_nested_lists(relation_tree))
        noun_list,modifier_list = _flatten_indices(noun_list), _flatten_indices(modifier_list)
        #mapping_dict = mapping_noun_modifier(relation_tree)
        #modifier_loss
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