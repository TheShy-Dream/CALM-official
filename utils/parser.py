import torch.distributions as dist
from typing import List, Dict
import itertools
import torch
from torch.nn import functional as F

from torch import random

start_token = "<|startoftext|>"
end_token = "<|endoftext|>"


def split_nested_lists(data):
    result = []

    def process(sublist):
        flat_part = []
        for item in sublist:
            if isinstance(item, list):
                process(item)
            else:
                flat_part.append(item)
        if flat_part:
            result.append(flat_part)

    for item in data:
        if isinstance(item, list):
            process(item)
        else:
            result.append([item])

    return result


def _get_outside_indices(subtree_indices, attn_map_idx_to_wp):
    flattened_subtree_indices = _flatten_indices(subtree_indices)
    outside_indices = [
        map_idx
        for map_idx in attn_map_idx_to_wp.keys() if (map_idx not in flattened_subtree_indices)
    ]
    return outside_indices


def _flatten_indices(related_indices): 
    flattened_related_indices = []
    for item in related_indices:
        if isinstance(item, list):
            flattened_related_indices.extend(item)
        else:
            flattened_related_indices.append(item)
    return flattened_related_indices

def mapping_noun_modifier(related_indices):
    mapping_dict = {}
    for sub_indice in related_indices:
        if isinstance(sub_indice, list):
            if len(sub_indice) > 1:
                mapping_dict[sub_indice[-1]] = sub_indice[:-1]
            else:
                mapping_dict[sub_indice[0]] = []
    return mapping_dict

def split_indices(related_indices: List[int]):
    noun = [related_indices[-1]]
    modifier = related_indices[:-1]
    if isinstance(modifier, int):
        modifier = [modifier]
    return noun, modifier


def supervised_contrastive_loss(features, labels, temperature=0.1):

    device = features.device
    batch_size = features.shape[0]

    features = F.normalize(features, p=2, dim=1)

    similarity_matrix = torch.matmul(features, features.T) / temperature

    max_val = torch.max(similarity_matrix, dim=1, keepdim=True).values
    similarity_matrix = similarity_matrix - max_val

    label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)

    self_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    positive_mask = label_mask - self_mask

    numerator = torch.sum(torch.exp(similarity_matrix) * positive_mask, dim=1)

    denominator = torch.sum(torch.exp(similarity_matrix) * (1 - self_mask), dim=1)

    loss = -torch.log(numerator / denominator + 1e-8).mean()
    return loss


def _symmetric_kl(attention_map1, attention_map2):
    if len(attention_map1.shape) > 1:
        attention_map1 = attention_map1.reshape(-1)
    if len(attention_map2.shape) > 1:
        attention_map2 = attention_map2.reshape(-1)

    p = dist.Categorical(probs=attention_map1)
    q = dist.Categorical(probs=attention_map2)

    kl_divergence_pq = dist.kl_divergence(p, q)
    kl_divergence_qp = dist.kl_divergence(q, p)

    avg_kl_divergence = (kl_divergence_pq + kl_divergence_qp) / 2
    return avg_kl_divergence


def align_wordpieces_indices(
        wordpieces2indices, start_idx, target_word
):
    wp_indices = [start_idx]
    wp = wordpieces2indices[start_idx].replace("</w>", "")

    for wp_idx in range(start_idx + 1, len(wordpieces2indices)):
        if wp.lower() == target_word.lower():
            break

        wp2 = wordpieces2indices[wp_idx].replace("</w>", "")
        if target_word.lower().startswith(wp.lower() + wp2.lower()) and wp2.lower() != target_word.lower():
            wp += wordpieces2indices[wp_idx].replace("</w>", "")
            wp_indices.append(wp_idx)
        else:
            wp_indices = (
                []
            )
            break

    return wp_indices


def extract_attribution_indices(doc):
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                subtree.append(child)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees

def extract_attribution_indices_with_verbs(doc):
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp",
                 'relcl']

    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
       
        subtree = []
        stack = []
        for child in w.children:
            if child.dep_ in modifiers:
                if child.pos_ not in ['AUX', 'VERB']:
                    subtree.append(child) 
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                if node.pos_ not in ['AUX', 'VERB']:
                    subtree.append(node)
                stack.extend(node.children)
        if subtree:
            subtree.append(w)
            subtrees.append(subtree)
    return subtrees 

def extract_attribution_indices_with_verb_root(doc):

    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"] 

    for w in doc:
        subtree = []
        stack = []

        if w.pos_ != 'AUX' or w.dep_ in modifiers:
            continue

        for child in w.children:
            if child.dep_ in modifiers or child.pos_ in ['NOUN', 'PROPN']:
                if child.pos_ not in ['AUX', 'VERB']:
                    subtree.append(child)
                stack.extend(child.children)
        if len(subtree) < 2:
            continue

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                if node.pos_ not in ['AUX']:
                    subtree.append(node)
                stack.extend(node.children)

        if subtree:
            if w.pos_ not in ['AUX']:
                subtree.append(w)
            subtrees.append(subtree)
    return subtrees

def get_indices(tokenizer, prompt: str) -> Dict[str, int]:
    ids = tokenizer(prompt).input_ids
    indices = {
        i: tok
        for tok, i in zip(
            tokenizer.convert_ids_to_tokens(ids), range(len(ids))
        )
    }
    return indices

def extract_entities_only(doc):
    entities = []
    for w in doc:
        if w.pos_ in ['NOUN', 'PROPN']:
            entities.append([w])
    return entities

def align_wordpieces_indices(
        wordpieces2indices, start_idx, target_word
):

    wp_indices = [start_idx]
    wp = wordpieces2indices[start_idx].replace("</w>", "")

    for wp_idx in range(start_idx + 1, len(wordpieces2indices)):
        if wp.lower() == target_word.lower():
            break

        wp2 = wordpieces2indices[wp_idx].replace("</w>", "")
        if target_word.lower().startswith(wp.lower() + wp2.lower()) and wp2.lower() != target_word.lower():
            wp += wordpieces2indices[wp_idx].replace("</w>", "")
            wp_indices.append(wp_idx)
        else:
            wp_indices = (
                []
            )
            break

    return wp_indices



def unify_lists(list_of_lists):
    def flatten(lst):
        for elem in lst:
            if isinstance(elem, list):
                yield from flatten(elem)
            else:
                yield elem

    def have_common_element(lst1, lst2):
        flat_list1 = set(flatten(lst1))
        flat_list2 = set(flatten(lst2))
        return not flat_list1.isdisjoint(flat_list2)

    lst = []
    for l in list_of_lists:
        lst += l
    changed = True
    while changed:
        changed = False
        merged_list = []
        while lst:
            first = lst.pop(0)
            was_merged = False
            for index, other in enumerate(lst):
                if have_common_element(first, other):
                    new_merged = first + [item for item in other if item not in first]
                    lst[index] = new_merged
                    changed = True
                    was_merged = True
                    break
            if not was_merged:
                merged_list.append(first)
        lst = merged_list

    return lst


def shuffle_modifiers(indices_list):
    shuffled_list = []
    for indices in indices_list:
        if len(indices) > 1:
            modifiers = indices[:-1]  
            noun = indices[-1]      
            random.shuffle(modifiers) 
            shuffled_list.append(modifiers + [noun]) 
        else:
            shuffled_list.append(indices)  
    return shuffled_list


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret