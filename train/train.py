# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import pandas as pd

import torch

from torch_geometric.data import Data
import transformers

from utils.constants import IGNORE_INDEX, DEFAULT_GRAPH_TOKEN, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN, DEFAULT_GRAPH_PAD_ID
from torch.utils.data import Dataset
from reconglm_trainer import ReconGLMTrainer

from model import *
import tokenizers
import random
from utils import conversation as conversation_lib
from utils.utils import tokenizer_graph_token, setup_seed
import scipy.sparse as sp
import numpy as np


local_rank = None

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    # freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_graph_start_end: bool = field(default=False)
    mm_use_graph_patch_token: bool = field(default=True)
    
    mm_inv_projector_type: Optional[str] = field(default=None)
    pretrain_mm_inv_mlp_adapter: Optional[str] = field(default=None)
    similarity_loss_weight: Optional[float] = field(default=0.0)
    denoised_loss_weight: Optional[float] = field(default=0.0)
    feat_loss_weight: Optional[float] = field(default=0.0)
    topo_loss_weight: Optional[float] = field(default=0.0)
    topo_recon_ratio: Optional[float] = field(default=0.5)
    agg_method: Optional[str] = field(default="mean")


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    pretrained_embedding_type: Optional[str] = field(default='sbert')
    denoised_embedding_type: Optional[str] = field(default=None)
    use_hop: Optional[int] = field(default=2)
    sample_neighbor_size: Optional[int] = field(default=-1)
    use_task:Optional[str] = field(default="nc")
    use_dataset:Optional[str] = field(default="arxiv")
    template: Optional[str] = field(default="ND")



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    mm_projector_lr: Optional[float] = field(default=None)
    mm_inv_projector_lr: Optional[float] = field(default=None)
    model_max_length: int = field(
        default=4096,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    seed: int = field(default=0)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_graph_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_graph:
        input_ids = torch.stack(
            [tokenizer_graph_token(prompt.replace(conv.sep2, ''), tokenizer, return_tensors='pt') for prompt in
             conversations], dim=0)
    else:
        input_ids = tokenizer(
            [prompt.replace(conv.sep2, '') for prompt in conversations],
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = f'<|start_header_id|>{conv.roles[1]}<|end_header_id|>\n\n'
    for conversation, target in zip(conversations, targets):
        total_len = target.shape[0]

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        rounds_len = []
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break

            parts[0] += sep

            if has_graph:
                # len_parts0_then = tokenizer_image_token(parts[0], tokenizer)
                round_len = len(tokenizer_graph_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer))
            else:
                # len_parts0_then = tokenizer(parts[0])
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids)

            if i != 0 and not getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            # print(target)
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # print(conversations[0])
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_graph:
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_graph:
                round_len = len(tokenizer_graph_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_graph:
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())
        total_len = target.shape[0]

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_graph:
                round_len = len(tokenizer_graph_token(rou, tokenizer))
                instruction_len = len(tokenizer_graph_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_graph=has_graph)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_graph=has_graph)
    if conversation_lib.default_conversation.version.startswith("llama3"):
        return preprocess_llama3(sources, tokenizer, has_graph=has_graph)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_graph_token(prompt, tokenizer)) for prompt in prompts]

    if has_graph:
        input_ids = [tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_graph:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

class LazySupervisedGraphDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedGraphDataset, self).__init__()
        self.use_dataset = data_args.use_dataset.split('-')
        self.use_hop = data_args.use_hop
        self.template = data_args.template
        self.datas={}
        list_data_dict = []
        self.pretrained_embs={}
        self.denoised_embs = {}
        self.denoised_embedding_type = data_args.denoised_embedding_type
        for d, dataset in enumerate(self.use_dataset):
            repeat=1
            if "." in dataset:
                # cora: use cora dataset
                # cora.3: repeat cora 3 times
                ds = dataset.split('.')
                repeat = int(ds[1])
                dataset = ds[0]
            if "arxiv" in dataset:
                data_path = "dataset/ogbn-arxiv/processed_data.pt"
            elif "pubmed"  in dataset:
                data_path = "dataset/pubmed/processed_data.pt"
            elif "cora"  in dataset:
                data_path = "dataset/cora/processed_data.pt"
            elif "reddit"  in dataset:
                data_path = "dataset/reddit/processed_data.pt"
            else:
                print(f"{dataset} not exists!!!!")
                raise ValueError
            data = torch.load(data_path)
            self.datas[dataset] = data
            data_dir = os.path.dirname(data_path)
            if data_args.template == "ND":
                pretrained_emb = self.load_pretrain_embedding_graph(data_dir, data_args.pretrained_embedding_type).detach()
                self.structure_emb = torch.load(
                    f"dataset/laplacian_{data_args.use_hop}_{data_args.sample_neighbor_size}.pt")
                if data_args.denoised_embedding_type is not None:
                    denoised_emb = self.load_pretrain_embedding_graph(data_dir, data_args.denoised_embedding_type).detach()
            else:
                raise ValueError

            self.pretrained_embs[dataset] = pretrained_emb
            if data_args.denoised_embedding_type is not None:
                self.denoised_embs[dataset] = denoised_emb

            self.use_task = data_args.use_task.split('-')

            for task in self.use_task:
                task_list_data_dict = []
                if task == "nc":
                    data_path = os.path.join(data_dir, f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl")
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l['task'] = task
                                l["dataset"]=dataset
                                task_list_data_dict.append(l)
                    else:
                        raise ValueError
                elif task == "lp":
                    data_path = os.path.join(data_dir, f"edge_sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_only_train.jsonl")
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l["dataset"] = dataset
                                l['task'] = task
                                l["conversations"][0][
                                    'value'] = f"Given two node-centered subgraphs: {DEFAULT_GRAPH_TOKEN} and {DEFAULT_GRAPH_TOKEN}, we need to predict whether these two nodes connect with each other. Please tell me whether two center nodes in the subgraphs should connect to each other."
                                task_list_data_dict.append(l)
                    else:
                        raise ValueError
                elif task == "nd":
                    data_path = os.path.join(data_dir, f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl")
                    user_prompt = f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l["dataset"] = dataset
                                l['task'] = task
                                id = l['id']
                                label = data.label_texts[data.y[id]]
                                if dataset in ["arxiv", "cora", "pubmed"]:
                                    title = data.title[id]
                                    if title == "":
                                        assistant_prompt = f"This is a paper in {label} domain"
                                    else:
                                        assistant_prompt = f"This is a paper in {label} domain, it's about {title}."
                                else:
                                    raise ValueError
                                l["conversations"] = [{'from': 'human', 'value': user_prompt},
                                                      {'from': 'gpt', 'value': assistant_prompt}]
                                task_list_data_dict.append(l)
                elif task == "nda":
                    data_path = os.path.join(data_dir, f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl")
                    user_prompt = f"Please briefly describe the center node of {DEFAULT_GRAPH_TOKEN}."
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l["dataset"] = dataset
                                l['task'] = task
                                id = l['id']
                                label = data.label_texts[data.y[id]]
                                if dataset in ["arxiv", "cora", "pubmed"]:
                                    title = data.title[id]
                                    ab = data.abs[id]
                                    if title == "" and ab == "":
                                        assistant_prompt = f"This is a paper in {label} domain"
                                    elif title == "":
                                        assistant_prompt = f"This is a paper in {label} domain, its title is {title}."
                                    elif ab == "":
                                        assistant_prompt = f"This is a paper in {label} domain, its abstract is {ab}."
                                    else:
                                        assistant_prompt = f"This is a paper in {label} domain, its title is {title}, its abstract is {ab}."
                                else:
                                    raise ValueError

                                l["conversations"] = [{'from': 'human', 'value': user_prompt},
                                                      {'from': 'gpt', 'value': assistant_prompt}]
                                task_list_data_dict.append(l)
                elif task == "nctext":
                    data_path = os.path.join(data_dir, f"sampled_{data_args.use_hop}_{data_args.sample_neighbor_size}_train.jsonl")
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as file:
                            for line in file:
                                l = json.loads(line)
                                l["dataset"]=dataset
                                l['task'] = task
                                text = data.raw_texts[l['id']]
                                text = text[:2000]
                                if dataset == "arxiv":
                                    qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to? Direct tell me the class name."
                                elif dataset == "pubmed":
                                    qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers about Diabetes and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 3 classes: Diabetes Mellitus Experimental, Diabetes Mellitus Type1, Diabetes Mellitus Type2, please tell me which class the center node belongs to? Direct tell me the class name."
                                elif dataset == "cora":
                                    qs = f"Given a node-centered graph: {DEFAULT_GRAPH_TOKEN}, where nodes represent papers and edges represent co-citations, the node feature of center node is {text}. We need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to? Direct tell me the class name."
                                l["conversations"][0]['value'] = qs
                                task_list_data_dict.append(l)
                    else:
                        raise ValueError
                else:
                    print(f"{task} not exist!!!")
                    raise ValueError

                if repeat > 1:
                    base_task_list_data_dict = copy.copy(task_list_data_dict)
                    for _ in range(repeat-1):
                        task_list_data_dict += base_task_list_data_dict
                rank0_print(f"Dataset {dataset} Task {task}, size {len(task_list_data_dict)}")
                list_data_dict.extend(task_list_data_dict)


        random.shuffle(list_data_dict)
        rank0_print(f"Formatting inputs...Skip in lazy mode, size {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def load_pretrain_embedding_graph(self, data_dir, pretrained_embedding_type):
        if pretrained_embedding_type == "simteg":
            simteg_sbert = torch.load(os.path.join(data_dir, "simteg_sbert_x.pt"))
            simteg_roberta = torch.load(os.path.join(data_dir, "simteg_roberta_x.pt"))
            simteg_e5 = torch.load(os.path.join(data_dir, "simteg_e5_x.pt"))
            pretrained_emb = torch.concat([simteg_sbert, simteg_roberta, simteg_e5], dim=-1)
        else:
            pretrained_emb = torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))
        return pretrained_emb

    def load_pretrain_embedding_hop(self, data_dir, pretrained_embedding_type, hop):
        if pretrained_embedding_type == "simteg":
            simteg_sbert=[torch.load(os.path.join(data_dir, f"simteg_sbert_x.pt"))] + [torch.load(os.path.join(data_dir, f"simteg_sbert_{i}hop_x.pt")) for i in range(1, hop + 1)]
            simteg_roberta = [torch.load(os.path.join(data_dir, f"simteg_roberta_x.pt"))] + [torch.load(os.path.join(data_dir, f"simteg_roberta_{i}hop_x.pt")) for i in range(1, hop + 1)]
            simteg_e5 = [torch.load(os.path.join(data_dir, f"simteg_e5_x.pt"))] + [torch.load(os.path.join(data_dir, f"simteg_e5_{i}hop_x.pt")) for i in range(1, hop + 1)]
            pretrained_embs = [torch.cat([simteg_sbert[i], simteg_roberta[i], simteg_e5[i]], dim=-1) for i in range(hop + 1)]
        else:
            pretrained_embs = [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_x.pt"))]+  [torch.load(os.path.join(data_dir, f"{pretrained_embedding_type}_{i}hop_x.pt")) for i in range(1, hop+1)]

        return pretrained_embs


    def __len__(self):
        return len(self.list_data_dict)



    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            graph_token_size = len(sample['graphs']) if 'graphs' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + graph_token_size)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'graph' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        edge_index, node_list, task = sources['edge_index'], sources['node_list'], sources['task']
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_graph=('graph' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        # graph exists in the data
        if 'graph' in self.list_data_dict[i]:
            if not isinstance(self.list_data_dict[i]['graph'][0], list):
                self.list_data_dict[i]['graph'] = [self.list_data_dict[i]['graph']]
            if self.template == "ND":
                graph = torch.LongTensor(self.list_data_dict[i]['graph'])
                mask = graph != DEFAULT_GRAPH_PAD_ID
                masked_graph_emb = self.pretrained_embs[self.list_data_dict[i]["dataset"]][graph[mask]]
                s, n, d = graph.shape[0], graph.shape[1], masked_graph_emb.shape[1]
                graph_emb = torch.zeros((s, n, d))
                graph_emb[mask] = masked_graph_emb
                if self.structure_emb is not None:
                    graph_emb = torch.cat([graph_emb, self.structure_emb.unsqueeze(0).expand(s, -1, -1)], dim=-1)
            else:
                raise ValueError

            if task == "lp":
                graph_data = []
                for cur_graph, cur_edge_index, cur_node_list in zip(graph, edge_index, node_list):
                    # construct aggregation dictionary
                    agg_dict = {}
                    for node in cur_node_list:
                        index = torch.where(cur_graph == node)[0]
                        agg_dict[node] = index
                    node_feature = self.pretrained_embs[self.list_data_dict[i]["dataset"]][cur_node_list]
                    cur_graph_data = Data(x=node_feature, edge_index=torch.Tensor(cur_edge_index).long(), 
                                    node_list=torch.tensor(cur_node_list), agg_dict=agg_dict)
                    if self.denoised_embedding_type is not None:
                        denoised_feature = self.denoised_embs[self.list_data_dict[i]["dataset"]][cur_node_list]
                        cur_graph_data.denoised_x = denoised_feature
                    graph_data.append(cur_graph_data)
            else:
                # construct aggregation dictionary
                agg_dict = {}
                for node in node_list:
                    index = torch.where(graph[0] == node)[0]
                    agg_dict[node] = index
                node_feature = self.pretrained_embs[self.list_data_dict[i]["dataset"]][node_list]
                graph_data = Data(x=node_feature, edge_index=torch.Tensor(edge_index).long(), 
                                node_list=torch.tensor(node_list), agg_dict=agg_dict)
                if self.denoised_embedding_type is not None:
                    denoised_feature = self.denoised_embs[self.list_data_dict[i]["dataset"]][node_list]
                    graph_data.denoised_x = denoised_feature
                graph_data = [graph_data]
            
            data_dict['graph'] = graph
            data_dict['graph_emb'] = graph_emb
            data_dict['graph_data'] = graph_data


        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'graph' in instances[0]:
            graph = [instance['graph'] for instance in instances]
            graph_emb = [instance['graph_emb'] for instance in instances]
            graph_data_batch = [instance['graph_data'] for instance in instances]
            batch['graph'] = torch.cat(graph, dim=0)
            batch['graph_emb'] = torch.cat(graph_emb, dim=0)
            batch['graph_data'] = graph_data_batch

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedGraphDataset(tokenizer=tokenizer,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def _train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    setup_seed(training_args.seed)
    
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    if "tmp" not in training_args.output_dir and os.path.exists(training_args.output_dir):
        if bool(os.listdir(training_args.output_dir)):
            print(f"{training_args.output_dir} already exists and not empty!!!!")
            return
        print(f"{training_args.output_dir} already exists!!!!")

    if data_args.pretrained_embedding_type in ['sbert', 'simteg_sbert']:
        model_args.mm_hidden_size = 384
    elif data_args.pretrained_embedding_type in ["simteg_e5", "simteg_roberta", "roberta"]:
        model_args.mm_hidden_size = 1024
    elif data_args.pretrained_embedding_type in ["simteg"]:
        model_args.mm_hidden_size = 1024*2+384
    else:
        raise ValueError
    
    if  getattr(model_args, 'mm_inv_projector_type', None) is not None:
        mm_inv_projector_type = getattr(model_args, 'mm_inv_projector_type')
        print(f"mm_inv_projector_type: {mm_inv_projector_type}")
        if mm_inv_projector_type.startswith("decoder"):
            model_args.mm_inv_input_size = model_args.mm_hidden_size
        elif mm_inv_projector_type.startswith("denoiser") or mm_inv_projector_type.startswith("similarizer"):
            if data_args.denoised_embedding_type in ['sbert', 'simteg_sbert', "simteg_e5", "simteg_roberta", "roberta", "simteg"]:
                model_args.mm_inv_input_size = model_args.mm_hidden_size
            if data_args.denoised_embedding_type in ['lgd_sbert', 'lgd_155_sbert']:
                model_args.mm_inv_input_size = 96
            if data_args.denoised_embedding_type in ['infomax_sbert', 'grace_sbert']:
                model_args.mm_inv_input_size = 128
            if data_args.denoised_embedding_type in ['gcn_sbert']:
                model_args.mm_inv_input_size = 64

    if data_args.template == "ND":
        data_args.structure_embedding_dim = int((data_args.sample_neighbor_size ** (data_args.use_hop + 1) - 1) / (data_args.sample_neighbor_size - 1))
        model_args.mm_hidden_size += data_args.structure_embedding_dim
    print(f"mm_hidden_size: {model_args.mm_hidden_size}")

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    model = ReconglmLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
    )

    model.config.use_cache = False

    # if model_args.freeze_backbone:
    #     model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            # target_modules=find_all_linear_names(model),
            target_modules=["q_proj", "v_proj"],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif "llama3" in model_args.model_name_or_path.lower():
            tokenizer.pad_token = '<|end_of_text|>'
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # if model_args.vision_tower is not None:
    model.get_model().initialize_graph_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )

    data_args.is_multimodal = True

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        if hasattr(model.get_model(), "mm_inv_projector"):
            for p in model.get_model().mm_inv_projector.parameters():
                p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        if hasattr(model.get_model(), "mm_inv_projector"):
            for p in model.get_model().mm_inv_projector.parameters():
                p.requires_grad = False
    
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_graph_start_end = data_args.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    training_args.mm_use_graph_start_end = model_args.mm_use_graph_start_end
    model.initialize_graph_tokenizer(model_args, tokenizer=tokenizer)

    # print model parameters information
    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    train_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() if p.requires_grad else 0 for p in model.parameters())
    print(f">> Total params: {total_params / 1.e6}M")
    print(f">> Train params: {train_params / 1.e6}M, Ratio {train_params / total_params * 100.:.2f}%")
    print(f">> Save every {training_args.save_steps} steps.")
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = ReconGLMTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    _train()
