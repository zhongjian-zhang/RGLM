#    Copyright 2023 Haotian Liu
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
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from model import *
from utils.constants import DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN
from huggingface_hub import hf_hub_download




def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", cache_dir="../../checkpoint"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    def _has_peft_adapter(path: str) -> bool:
        return any(os.path.exists(os.path.join(path, fname)) for fname in [
            'adapter_config.json',
            'adapter_model.bin',
            'adapter_model.safetensors',
        ])

    if 'reconglm' in model_name.lower():
        # Load ReconGLM model
        peft_present = _has_peft_adapter(model_path)
        if peft_present and model_base is None:
            warnings.warn('Detected PEFT adapter files in the model directory but no `model_base` is provided. Please specify `--model_base` to load and merge LoRA adapters.')
        if peft_present and model_base is not None:
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading ReconGLM from base model...')
            model = ReconglmLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, cache_dir=cache_dir,  **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional ReconGLM weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading ReconGLM from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            # First try to load from model_path (trained model), fallback to model_base
            if os.path.exists(os.path.join(model_path, 'pytorch_model.bin')) or \
                os.path.exists(os.path.join(model_path, 'model.safetensors')) or \
                any(f.startswith('model-') and f.endswith('.safetensors') for f in os.listdir(model_path)):
                print(f"Loading trained model from {model_path}")
                model = ReconglmLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=cfg_pretrained, cache_dir=cache_dir, **kwargs)
            else:
                print(f"Model files not found in {model_path}, loading base model from {model_base}")
                model = ReconglmLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, cache_dir=cache_dir, **kwargs)
            
            if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                print("Load mm_projector from local path")
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
            else:
                # only try to load mm_projector from HuggingFace Hub if not loaded from full training weights
                if not locals().get('loaded_trained_from_path', False):
                    try:
                        model_path_hf = hf_hub_download(repo_id=model_path, filename='mm_projector.bin')
                        mm_projector_weights = torch.load(model_path_hf, map_location='cpu')
                        print("Load mm_projector from huggingface")
                        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                        model.load_state_dict(mm_projector_weights, strict=False)
                    except Exception as e:
                        print(f"Warning: failed to load mm_projector.bin from HuggingFace Hub: {e}. Skipping.")
                else:
                    print("mm_projector.bin not found, but full model weights are already loaded. Skip loading mm_projector.")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = ReconglmLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", cache_dir=cache_dir)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, cache_dir=cache_dir, **kwargs)


    if 'reconglm' in model_name.lower():
        mm_use_graph_start_end = getattr(model.config, "mm_use_graph_start_end", False)
        if mm_use_graph_start_end:
            tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len
