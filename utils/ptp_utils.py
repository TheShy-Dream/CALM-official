import abc
import math

import cv2
import numpy as np
import torch
from IPython.display import display
from PIL import Image
from typing import Union, Tuple, List, Dict, Sequence, Optional
from diffusers.models.attention_processor import Attention
from torch.nn import functional as F
from diffusers.models.embeddings import apply_rotary_emb

from utils.parser import split_nested_lists


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                display_image: bool = True) -> Image.Image:
    """ Displays a list of images in a grid. """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if display_image:
        display(pil_img)
    return pil_img

def scaled_dot_product_attention_new(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool,device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight,attn_weight @ value


class AttendExciteAttnProcessor:
    """Wraps the native processor, still performs attention, but stores maps."""

    def __init__(self, attnstore, place: str, indices=None, modifier_threshold: float = 0.5,
                 noun_threshold: float = 0.5,noun_alpha=1.0,modifier_alpha=1.0,whether_enhance=True):
        self.attnstore = attnstore
        self.place = place
        self.indices = indices
        self.modifier_threshold = modifier_threshold
        self.noun_threshold = noun_threshold
        self.noun_alpha = noun_alpha
        self.modifier_alpha = modifier_alpha
        self.whether_enhance = whether_enhance

    def apply_sublist_attention_mask(self, encoder_hidden_states):
        original_dtype = encoder_hidden_states.dtype
        noun_indices_list = []
        modifier_indices_list = []
        for sub_list in split_nested_lists(self.indices):
            if len(sub_list) == 0:
                continue
            noun_idx = sub_list[-1]
            modifier_idx = sub_list[:-1]
            noun_indices_list.append(noun_idx)
            modifier_indices_list.append(modifier_idx)

        all_noun_indices = [idx for idx in noun_indices_list if idx is not None]
        all_modifier_indices = [idx for sub in modifier_indices_list for idx in sub if idx is not None]
        all_indices = all_noun_indices + all_modifier_indices

        attn_scores = torch.matmul(
            encoder_hidden_states, encoder_hidden_states.transpose(-1, -2)
        )
        attn_probs = torch.softmax(attn_scores, dim=-1)

        mask_factor = torch.ones(
            encoder_hidden_states.shape[:2], device=encoder_hidden_states.device
        ) 

        for mod_list, noun_idx in zip(modifier_indices_list, noun_indices_list):
            if len(mod_list) == 0:
                continue
            for m in mod_list: 
                local_indices = mod_list + [noun_idx]
                numerator = attn_probs[0, m, local_indices].sum()
                denominator = attn_probs[0, m, all_indices].sum() + 1e-6

                ratio = numerator / denominator
                if ratio < self.modifier_threshold:
                    mask_factor[0, m] += self.modifier_alpha * torch.tanh(self.modifier_threshold - ratio)

        for noun_idx in all_noun_indices:
            numerator = attn_probs[0, noun_idx, noun_idx]
            denominator = attn_probs[0, noun_idx, all_noun_indices].sum() + 1e-6

            ratio = numerator / denominator
            if ratio < self.noun_threshold:
                mask_factor[0, noun_idx] += self.noun_alpha * torch.tanh(self.noun_threshold - ratio)

        encoder_hidden_states = encoder_hidden_states * mask_factor.unsqueeze(-1)
        return encoder_hidden_states.to(original_dtype)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        b, seq_len, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, b)
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        enc = encoder_hidden_states if is_cross else hidden_states
        if is_cross and self.indices and self.whether_enhance:
            enc = self.apply_sublist_attention_mask(enc)
        key, value = attn.to_k(enc), attn.to_v(enc)
        query, key, value = map(attn.head_to_batch_dim, (query, key, value))
        probs = attn.get_attention_scores(query, key, attention_mask)
        self.attnstore(probs, is_cross, self.place)

        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class AttendExciteAttnProcessorDiT:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        attnstore,
        indices=None,
        modifier_threshold: float = 0.5,
        noun_threshold: float = 0.5,
        noun_alpha: float = 1.0,
        modifier_alpha: float = 1.0,
        whether_enhance=False,
    ):
        self.attnstore = attnstore
        self.indices = indices
        self.modifier_threshold = modifier_threshold
        self.noun_threshold = noun_threshold
        self.noun_alpha = noun_alpha
        self.modifier_alpha = modifier_alpha
        self.whether_enhance = whether_enhance

    def enhance_encoder_hidden_states(self, encoder_hidden_states):
        """Apply AttendExcite-style attention mask using a single indices set."""
        if self.indices is None:
            return encoder_hidden_states

        original_dtype = encoder_hidden_states.dtype

        noun_indices_list, modifier_indices_list = [], []
        for sub_list in split_nested_lists(self.indices):
            if not sub_list:
                continue
            noun_idx = sub_list[-1]
            modifier_idx = sub_list[:-1]
            noun_indices_list.append(noun_idx)
            modifier_indices_list.append(modifier_idx)

        all_noun_indices = [idx for idx in noun_indices_list if idx is not None]
        all_modifier_indices = [idx for sub in modifier_indices_list for idx in sub if idx is not None]
        all_indices = all_noun_indices + all_modifier_indices

        attn_scores = torch.matmul(
            encoder_hidden_states, encoder_hidden_states.transpose(-1, -2)
        ).float()
        attn_probs = torch.softmax(attn_scores, dim=-1)

        mask_factor = torch.ones(encoder_hidden_states.shape[:2], device=encoder_hidden_states.device)

        for mod_list, noun_idx in zip(modifier_indices_list, noun_indices_list):
            if not mod_list:
                continue
            for m in mod_list:
                local_indices = mod_list + [noun_idx]
                numerator = attn_probs[:, m, local_indices].sum(dim=-1)
                denominator = attn_probs[:, m, all_indices].sum(dim=-1) + 1e-6
                ratio = numerator / denominator
                mask_factor[:, m] += self.modifier_alpha * torch.tanh(self.modifier_threshold - ratio)

        # Noun enhancement
        for noun_idx in all_noun_indices:
            numerator = attn_probs[:, noun_idx, noun_idx]
            denominator = attn_probs[:, noun_idx, all_noun_indices].sum(dim=-1) + 1e-6
            ratio = numerator / denominator
            mask_factor[:, noun_idx] += self.noun_alpha * torch.tanh(self.noun_threshold - ratio)

        return (encoder_hidden_states * mask_factor.unsqueeze(-1)).to(original_dtype)


    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            *args,
            **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None and self.whether_enhance:
            clip_enc, t5_enc = encoder_hidden_states[:, :77], encoder_hidden_states[:, 77:]

            clip_enc = self.enhance_encoder_hidden_states(clip_enc)
            t5_enc = self.enhance_encoder_hidden_states(t5_enc)

            encoder_hidden_states = torch.cat([clip_enc, t5_enc], dim=1)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)
        attn_weight, hidden_states = scaled_dot_product_attention_new(query, key, value, dropout_p=0.0, is_causal=False) #[2,24,4429,4429]
        if encoder_hidden_states is not None: 
            attn_to_store = attn_weight[:, :, residual.shape[1]:, :residual.shape[1]] #[2,24,333,4096]
            self.attnstore(attn_to_store.reshape(-1,attn_to_store.shape[2],attn_to_store.shape[3]).permute(0,2,1))

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

#FluxAttnProcessor2_0
class AttendExciteAttentionProcessorFlux:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        attnstore,
        indices=None,
        modifier_threshold: float = 0.5,
        noun_threshold: float = 0.5,
        noun_alpha: float = 1.0,
        modifier_alpha: float = 1.0,
        whether_enhance=False,
        txt_length=512,
    ):
        self.attnstore = attnstore
        self.indices = indices
        self.modifier_threshold = modifier_threshold
        self.noun_threshold = noun_threshold
        self.noun_alpha = noun_alpha
        self.modifier_alpha = modifier_alpha
        self.whether_enhance = whether_enhance
        self.txt_length = txt_length

        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        """

    def enhance_encoder_hidden_states(self, encoder_hidden_states):
        """Apply AttendExcite-style attention mask using a single indices set."""
        if self.indices is None:
            return encoder_hidden_states

        original_dtype = encoder_hidden_states.dtype

        noun_indices_list, modifier_indices_list = [], []
        for sub_list in split_nested_lists(self.indices):
            if not sub_list:
                continue
            noun_idx = sub_list[-1]
            modifier_idx = sub_list[:-1]
            noun_indices_list.append(noun_idx)
            modifier_indices_list.append(modifier_idx)

        all_noun_indices = [idx for idx in noun_indices_list if idx is not None]
        all_modifier_indices = [idx for sub in modifier_indices_list for idx in sub if idx is not None]
        all_indices = all_noun_indices + all_modifier_indices

        attn_scores = torch.matmul(
            encoder_hidden_states, encoder_hidden_states.transpose(-1, -2)
        ).float() 
        attn_probs = torch.softmax(attn_scores, dim=-1)

        mask_factor = torch.ones(encoder_hidden_states.shape[:2], device=encoder_hidden_states.device)

        # Modifier enhancement
        for mod_list, noun_idx in zip(modifier_indices_list, noun_indices_list):
            if not mod_list:
                continue
            for m in mod_list:
                local_indices = mod_list + [noun_idx]
                numerator = attn_probs[:, m, local_indices].sum(dim=-1)
                denominator = attn_probs[:, m, all_indices].sum(dim=-1) + 1e-6
                ratio = numerator / denominator
                mask_factor[:, m] += self.modifier_alpha * torch.tanh(self.modifier_threshold - ratio)

        # Noun enhancement
        for noun_idx in all_noun_indices:
            numerator = attn_probs[:, noun_idx, noun_idx]
            denominator = attn_probs[:, noun_idx, all_noun_indices].sum(dim=-1) + 1e-6
            ratio = numerator / denominator
            mask_factor[:, noun_idx] += self.noun_alpha * torch.tanh(self.noun_threshold - ratio)

        return (encoder_hidden_states * mask_factor.unsqueeze(-1)).to(original_dtype)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape


        if hidden_states is not None and self.whether_enhance:
            text_enc,latent_enc = hidden_states[:,:self.txt_length, :], hidden_states[:,self.txt_length:, :]
            text_enc = self.enhance_encoder_hidden_states(text_enc)

            hidden_states = torch.cat([text_enc, latent_enc], dim=1)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:


            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        attn_weight,hidden_states = scaled_dot_product_attention_new(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        attn_to_store = attn_weight[:, :, self.txt_length:, :self.txt_length]  # torch.Size([1, 24, 4096, 256])
        self.attnstore(attn_to_store.reshape(-1, attn_to_store.shape[2], attn_to_store.shape[3]).permute(0, 1,2))
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class AttentionStore:
    """Collects the cross‑attention tensors at each U‑Net layer."""

    @staticmethod
    def _blank() -> Dict[str, list]:
        return {"down": [], "mid": [], "up": []}

    def __init__(self, attn_res: Tuple[int, int]):
        self.attn_res = attn_res
        self.num_att_layers: int = -1
        self.cur_att_layer: int = 0
        self.step_store: Dict[str, list[torch.Tensor]] = self._blank()
        self.attention_store: Dict[str, list[torch.Tensor]] = {}

    # Called by custom AttnProcessor -----------------------------------------
    def __call__(self, attn: torch.Tensor, is_cross: bool, place: str):
        if (
            is_cross
            and self.cur_att_layer >= 0
            and attn.shape[1] == math.prod(self.attn_res)
        ):
            self.step_store[place].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.attention_store = self.step_store
            self.step_store = self._blank()

    # Queries -----------------------------------------------------------------
    def aggregate(self, where: Sequence[str]) -> torch.Tensor:
        maps = []
        for loc in where:
            for m in self.attention_store.get(loc, []):
                maps.append(
                    m.reshape(-1, self.attn_res[0], self.attn_res[1], m.shape[-1])
                )
        if not maps:
            raise ValueError("No attention maps collected; check attn_res.")
        maps = torch.cat(maps, 0)
        return maps.sum(0) / maps.shape[0]

class AttentionStoreDiT:
    """Collects the cross‑attention tensors at each U‑Net layer."""

    @staticmethod
    def _blank() -> list:
        return {"clip_text":[], "T5_text": []}

    def __init__(self):
        self.num_att_layers: int = -1
        self.cur_att_layer: int = 0
        self.attn_res: Tuple[int, int] = (64,64)
        self.step_store: Dict[str, list[torch.Tensor]] = self._blank()
        self.attention_store: Dict[str, list[torch.Tensor]] = {}

    # Called by custom AttnProcessor -----------------------------------------
    def __call__(self, attn: torch.Tensor):
        if (
            self.cur_att_layer>=0
        ):
            self.step_store["clip_text"].append(attn[:,:,:77])
            self.step_store["T5_text"].append(attn[:,:,77:]) 
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers: 
            self.cur_att_layer = 0
            self.attention_store = self.step_store
            self.step_store = self._blank()

    # Queries -----------------------------------------------------------------
    def aggregate(self, what: Sequence[str]) -> torch.Tensor:
        maps = []
        for loc in what: #what in ["clip_text", "T5_text"]
            for m in self.attention_store.get(loc, []):
                maps.append(
                    m.reshape(-1, self.attn_res[0], self.attn_res[1], m.shape[-1])
                )
        if not maps:
            raise ValueError("No attention maps collected; check attn_res.")
        maps = torch.cat(maps, 0) 
        return maps.sum(0) / maps.shape[0]

class AttentionStoreFlux:
    """Collects the cross‑attention tensors at each U‑Net layer."""
    @staticmethod
    def _blank() -> list:
        return {"text":[]}

    def __init__(self):
        self.num_att_layers: int = -1
        self.cur_att_layer: int = 0
        self.attn_res: Tuple[int, int] = (64,64)
        self.step_store: Dict[str, list[torch.Tensor]] = self._blank()
        self.attention_store: Dict[str, list[torch.Tensor]] = {}

    # Called by custom AttnProcessor -----------------------------------------
    def __call__(self, attn: torch.Tensor):
        if (
            self.cur_att_layer>=0
        ):
            self.step_store["text"].append(attn)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers: 
            self.cur_att_layer = 0
            self.attention_store = self.step_store
            self.step_store = self._blank()

    # Queries -----------------------------------------------------------------
    def aggregate(self, what: Sequence[str]) -> torch.Tensor:
        maps = []
        for loc in what: #what in ["text"]
            for m in self.attention_store.get(loc, []):
                maps.append(
                    m.reshape(-1, self.attn_res[0], self.attn_res[1], m.shape[-1])
                )
        if not maps:
            raise ValueError("No attention maps collected; check attn_res.")
        maps = torch.cat(maps, 0)
        return maps.sum(0) / maps.shape[0]
