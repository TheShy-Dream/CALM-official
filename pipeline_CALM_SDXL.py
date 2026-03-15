from diffusers.utils import logging
import random
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import spacy
import torch
from diffusers import StableDiffusionXLPipeline
from torch.nn import functional as F

from PIL import Image as _PIL_Image  # only for type hints
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from utils.gaussian_smoothing import GaussianSmoothingPatch
from utils.parser import extract_entities_only, extract_attribution_indices_with_verbs, \
    extract_attribution_indices_with_verb_root, extract_attribution_indices, start_token, end_token, get_indices, \
    align_wordpieces_indices, unify_lists, _flatten_indices,  split_indices, \
    supervised_contrastive_loss, split_nested_lists
from utils.ptp_utils import AttentionStore, AttendExciteAttnProcessor
random.seed(42)

logger = logging.get_logger(__name__)


class CALM_XLPipeline(StableDiffusionXLPipeline):
    """SD‑XL pipeline augmented with Attend‑and‑Excite latent steering."""

    # ------------------------------------------------------------------ utils
    # Keep original helpers but expose wrappers so our code looks like v1.5 ----
    # (They simply forward to the private methods of SDXL base pipeline.)
    def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            **kwargs,
    ):
        return super().encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )

    def get_add_time_ids(
            self,
            orig_size: Tuple[int, int],
            target_size: Tuple[int, int],
            dtype: torch.dtype,
            text_embed_dim: int,
    ) -> torch.Tensor:
        return self._get_add_time_ids(
            orig_size,
            (0, 0),  # crop top‑left
            target_size,
            dtype,
            text_embed_dim,
        )

    # ---------------------------------------------------------------- register
    def _register_attention_control(self, store: AttentionStore,indices=None,modifier_threshold=0.2,noun_threshold=0.3,noun_alpha=1.0,modifier_alpha=1.0,whether_enhance=False) -> None:
        procs, count = {}, 0
        for name in self.unet.attn_processors.keys():
            place = (
                "mid"
                if name.startswith("mid_block")
                else "up" if name.startswith("up_blocks") else "down"
            )
            procs[name] = AttendExciteAttnProcessor(store, place,indices=indices,modifier_threshold=modifier_threshold,noun_threshold=noun_threshold,
                                                    noun_alpha=noun_alpha,modifier_alpha=modifier_alpha,whether_enhance=whether_enhance)
            count += 1
        self.unet.set_attn_processor(procs)
        store.num_att_layers = count


    # ----------------------------------------------------------- public helper
    def get_indices(self, prompt: str) -> Dict[str, int]:
        ids = self.tokenizer(prompt).input_ids
        return {
            tok: i for i, tok in enumerate(self.tokenizer.convert_ids_to_tokens(ids))
        }

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
        return modifier_indices

    # ------------------------------------------------------------------- core
    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            *,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: int = 1,
            eta: float = 0.0,
            generator: Optional[torch.Generator] = None,
            latents: Optional[torch.Tensor] = None,
            output_type: str = "pil",
            return_dict: bool = True,
            callback: Optional[Any] = None,
            callback_steps: int = 1,
            max_iter_to_alter: int = 25,
            scale_factor: int = 20,
            attn_res: Optional[Tuple[int, int]] = None,
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
            whether_enhance: bool = False,
            **unused,
    ):
        """Generate images while enforcing token saliency via Attend‑and‑Excite."""
        # 0. default resolution ------------------------------------------------
        height = height or self.unet.config.sample_size * 8
        width = width or self.unet.config.sample_size * 8

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_cfg = guidance_scale > 1.0

        self.beta = beta
        self.temperature = temperature
        self.patience = patience
        self.num_intervention_steps = num_intervention_steps

        # 1. text enc ----------------------------------------------------------
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_cfg,
            negative_prompt,
        )

        add_time_ids = self.get_add_time_ids(
            (height, width),
            (height, width),
            prompt_embeds.dtype,
            pooled_prompt_embeds.shape[-1],
        )

        # Concatenate for CFG --------------------------------------------------
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds]
            )
            add_time_ids = torch.cat([add_time_ids, add_time_ids])
        else:
            add_text_embeds = pooled_prompt_embeds

        prompt_embeds, add_text_embeds, add_time_ids = [
            x.to(device) for x in (prompt_embeds, add_text_embeds, add_time_ids)
        ]

        # 2. timesteps & latents ----------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 3. attention capture -------------------------------------------------
        if attn_res is None:
            attn_res = (int(np.ceil(width / 32)), int(np.ceil(height / 32)))
        self.attention_store = AttentionStore(attn_res)
        original_procs = self.unet.attn_processors
        if not run_standard_sd:
            self.modifier_noun_indices = self.syntactic_extractor(prompt, include_entities=True)
            self._register_attention_control(self.attention_store,self.modifier_noun_indices,modifier_threshold=modifier_threshold,noun_threshold=noun_threshold,
                                             noun_alpha=noun_alpha,modifier_alpha=modifier_alpha,whether_enhance=whether_enhance)

        # 4. step size schedule -----------------------------------------------
        scale_range = np.linspace(1.0, 0.5, len(timesteps))
        step_sizes = scale_factor * np.sqrt(scale_range)

        # 5. split embeds for refinement batch --------------------------------
        text_embeds_refine = (
            prompt_embeds[batch_size * num_images_per_prompt:]
            if do_cfg
            else prompt_embeds
        )
        add_txt_refine = (
            add_text_embeds[batch_size * num_images_per_prompt:]
            if do_cfg
            else add_text_embeds
        )
        tids_refine = (
            add_time_ids[batch_size * num_images_per_prompt:]
            if do_cfg
            else add_time_ids
        )

        """
        if isinstance(token_indices[0], int):
            token_indices = [token_indices]
        indices_batched: List[List[int]] = []
        for ind in token_indices:
            indices_batched += [ind] * num_images_per_prompt
        """

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6. denoise loop ------------------------------------------------------
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
                    add_txt_refine,
                    tids_refine,
                    num_intervention_steps=num_intervention_steps,
                    max_intervention_steps_per_iter=max_intervention_steps_per_iter,
                    patience=patience,
                    kernel_size=kernel_size
                )
            # diffusion step --------------------------------------------------
            model_in = torch.cat([latents] * 2) if do_cfg else latents
            model_in = self.scheduler.scale_model_input(model_in, t)
            added_all = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            noise_pred = self.unet(
                model_in,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_all,
            ).sample
            if do_cfg:
                uncond, cond = noise_pred.chunk(2)
                noise_pred = uncond + guidance_scale * (cond - uncond)

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

            if callback and i % callback_steps == 0:
                callback(i, t, latents)
            progress.update(1)

        # 7. decode -----------------------------------------------------------
        image = self.decode_latents(latents) if output_type == "pil" else latents
        self.unet.set_attn_processor(original_procs)  # restore
        progress.close()

        if not return_dict:
            return (image, None)
        return StableDiffusionPipelineOutput(images=[image], nsfw_content_detected=None)

    def _upgrade_step(
            self,
            latents,
            text_embeddings,
            t,
            i,
            step_size,
            add_txt_refine,
            tids_refine,
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
                for latent, text_embedding in zip(latents, text_embeddings):
                    # Forward pass of denoising with text conditioning
                    latent = latent.unsqueeze(0)
                    text_embedding = text_embedding.unsqueeze(0)
                    best_loss = torch.inf
                    added_cond_kwargs = {"text_embeds": add_txt_refine, "time_ids": tids_refine}
                    while patience > 0 and max_intervention_steps_per_iter > 0:
                        self.unet(
                            latent,
                            t,
                            encoder_hidden_states=text_embedding,
                            added_cond_kwargs=added_cond_kwargs,
                        ).sample
                        self.unet.zero_grad()
                        # Get attention maps
                        attention_maps = self.attention_store.aggregate(("down","mid","up"))
                        loss = self._compute_loss(attention_maps=attention_maps, relation_tree=self.modifier_noun_indices,
                                                  temperature=self.temperature, kernel_size=kernel_size,beta=self.beta)
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
                            patience = self.patience
                        else:
                            patience -= 1
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
        return positive_loss +beta * negative_loss

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

    def compute_negative_loss(self, attention_maps: torch.Tensor, relation_tree: List[List[int]] = None,
                              temperature: float = 0.1, beta: float = 0.2) -> torch.Tensor:
        noun_list, modifier_list = split_indices(split_nested_lists(relation_tree))
        noun_list, modifier_list = _flatten_indices(noun_list), _flatten_indices(modifier_list)
        feature_list = []
        label_list = []
        for label_idx, sub_list in enumerate(split_nested_lists(relation_tree)):
            attn_value_list = self._compute_attention_per_index(attention_maps, sub_list)
            feature_list.extend(attn_value_list)
            label_list.extend([label_idx] * len(attn_value_list))
        feature_stacked = torch.stack(feature_list)
        label_stacked = torch.tensor(label_list).to(feature_stacked.device)
        modifier_scl_loss = supervised_contrastive_loss(feature_stacked.view(feature_stacked.shape[0], -1),
                                                        label_stacked, temperature=temperature)
        # negative_loss = noun_loss + beta * modifier_scl_loss
        return modifier_scl_loss / len(label_list)

    @staticmethod
    def _update_latent(
            latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def decode_latents(self, latents: torch.Tensor) -> _PIL_Image.Image:
        needs_upcasting = (
                self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.upcast_vae()
            latents = latents.to(
                next(iter(self.vae.post_quant_conv.parameters())).dtype
            )
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")[0]
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        return image