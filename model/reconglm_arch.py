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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import re
from model.denoiser.graph_denoiser import GraphDenoiser
from utils.constants import IGNORE_INDEX, GRAPH_TOKEN_INDEX, DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN, DEFAULT_GRAPH_PAD_ID
from model.decoder.decoder import GraphDecoder
from model.similarizer.similarizer import Similarizer
from utils.utils import initialize_weights

def build_graph_inv_projector(config):
    projector_type = getattr(config, 'mm_inv_projector_type')
    if projector_type.startswith("decoder"):
        decoder_match = re.match(r'^decoder_mlp(\d+)x$', projector_type)
        depth = int(decoder_match.group(1))
        return GraphDecoder(
            input_dim=config.hidden_size,
            output_dim=config.mm_inv_input_size,
            depth=depth,
        )
    elif projector_type.startswith("similarizer"):
        decoder_match = re.match(r'^similarizer_mlp(\d+)x$', projector_type)
        depth = int(decoder_match.group(1))
        return Similarizer(
            input_dim=config.hidden_size,
            output_dim=config.mm_inv_input_size,
            depth=depth,
        )
    elif projector_type.startswith("denoiser"):
        git_match = re.match(r'^denoiser_git(\d+)x$', projector_type)
        depth = int(git_match.group(1))
        if depth == 8:
            width = 1280
        elif depth == 12:
            width = 1536
        else:
            width = 512
        return GraphDenoiser(
            x_dim=config.mm_inv_input_size,  # x
            z_dim=config.hidden_size,  # context
            embed_dim=width,  
            depth=depth,
            timesteps='500',
        )    
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')

def build_graph_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    hidden_dim = getattr(config, 'word_embed_proj_dim', getattr(config, 'hidden_size', 'linear'))

    if projector_type == 'linear':
        return initialize_weights(nn.Linear(config.mm_hidden_size, hidden_dim))
    mlp_gelu_match = re.match(r'^(\d+)-layer-mlp$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, hidden_dim)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_dim, hidden_dim))
        return initialize_weights(nn.Sequential(*modules))
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')


class ReconglmMetaModel:

    def __init__(self, config):
        super(ReconglmMetaModel, self).__init__(config)

        if hasattr(config, "mm_hidden_size"):
            self.mm_projector = build_graph_projector(config)
        if hasattr(config, "mm_use_graph_special_token") and getattr(config, 'mm_use_graph_special_token', False):
            self.special_token_emb = self.build_special_tokens()


    def initialize_graph_modules(self, model_args, fsdp=None):
        pretrain_mm_mlp_adapter = getattr(model_args, 'pretrain_mm_mlp_adapter', None)
        pretrain_mm_inv_mlp_adapter = getattr(model_args, 'pretrain_mm_inv_mlp_adapter', None)

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = getattr(model_args, 'mm_hidden_size')
        self.config.mm_inv_projector_type = getattr(model_args, 'mm_inv_projector_type', None)
        self.config.mm_inv_input_size = getattr(model_args, 'mm_inv_input_size', None)

        self.config.similarity_loss_weight = getattr(model_args, 'similarity_loss_weight', 0.0)
        self.config.denoised_loss_weight = getattr(model_args, 'denoised_loss_weight', 0.0)
        self.config.feat_loss_weight = getattr(model_args, 'feat_loss_weight', 0.0)
        self.config.topo_loss_weight = getattr(model_args, 'topo_loss_weight', 0.0)
        self.config.topo_recon_ratio = getattr(model_args, 'topo_recon_ratio', 0.0)
        self.config.agg_method = getattr(model_args, 'agg_method', "mean")


        self.mm_projector = build_graph_projector(self.config)
        # print("projector", self.mm_projector.weight.mean().item(), self.mm_projector.weight.std().item())
        if self.config.mm_inv_projector_type is not None:
            self.mm_inv_projector = build_graph_inv_projector(self.config)
            print(self.mm_inv_projector)
        if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
            self.special_token_emb = self.build_special_tokens()

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        if pretrain_mm_inv_mlp_adapter is not None:
            mm_inv_projector_weights = torch.load(pretrain_mm_inv_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_inv_projector.load_state_dict(get_w(mm_inv_projector_weights, 'mm_inv_projector'))

    def build_special_tokens(self):
        if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
            num_token=self.config.use_hop+2
            input_embeddings = self.get_input_embeddings().weight.data
            input_embeddings_avg = input_embeddings.mean(dim=0, keepdim=True).unsqueeze(1).detach()
            special_token_emb=torch.nn.parameter.Parameter(data=input_embeddings_avg.repeat(num_token, 1, 1), requires_grad=True)
            return special_token_emb
        return None


class ReconglmMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def encode_graphs(self, graph, graph_emb):
        graph_features = self.get_model().mm_projector(graph_emb)
        graph_features[graph==DEFAULT_GRAPH_PAD_ID] = 0.
        return graph_features

    def inject_special_token(self, graph_emb):
        use_hop=self.config.use_hop
        sample_size = self.config.sample_neighbor_size
        assert graph_emb.shape[-2] == int((sample_size ** (use_hop + 1) - 1) / (sample_size - 1))
        assert self.model.special_token_emb.shape[0] == use_hop + 2
        new_graph_emb = []
        new_graph_emb.append(self.model.special_token_emb[0])
        cur=0
        for i in range(use_hop+1):
            cur_size = sample_size**i
            new_graph_emb.append(graph_emb[cur:cur+cur_size])
            cur+=cur_size
            new_graph_emb.append(self.model.special_token_emb[i+1])
        new_graph_emb = torch.concat(new_graph_emb, dim=0)
        return new_graph_emb

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, graphs, graph_emb
    ):
        if past_key_values is not None and graphs is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels, None, None

        graph_features = self.encode_graphs(graphs, graph_emb)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        # record the start and end position of the k-th graph aligned to the batch:
        boi_ids_per_rank, eoi_ids_per_rank = [], []
        cur_graph_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == GRAPH_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_graph_features = graph_features[cur_graph_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_graph_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_graph_idx += 1
                continue
            graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            cur_embed_pos = 0  # track the current embedding position
            # record the start and end position of the graphs in the current sample in the order of appearance
            batch_boi_list, batch_eoi_list = [], []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while graph_token_indices.numel() > 0:
                cur_graph_features = graph_features[cur_graph_idx]
                if hasattr(self.config, "mm_use_graph_special_token") and getattr(self.config, 'mm_use_graph_special_token', False):
                    cur_graph_features = self.inject_special_token(cur_graph_features)

                graph_token_start = graph_token_indices[0]
                # record the start and end position of the current graph
                cur_boi = cur_embed_pos + graph_token_start
                cur_eoi = cur_embed_pos + graph_token_start + cur_graph_features.shape[0] - 1
                batch_boi_list.append(cur_boi)
                batch_eoi_list.append(cur_eoi)
                
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[graph_token_start-1:graph_token_start]))
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[graph_token_start+1:graph_token_start+2]))
                    cur_embed_pos += graph_token_start + 1 + cur_graph_features.shape[0] + 1
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[graph_token_start:graph_token_start+1])
                        cur_labels = cur_labels[graph_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:graph_token_start]))
                    cur_new_input_embeds.append(cur_graph_features)
                    cur_embed_pos += graph_token_start + cur_graph_features.shape[0]
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:graph_token_start])
                        cur_new_labels.append(torch.full((cur_graph_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[graph_token_start+1:]
                cur_graph_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_input_ids = cur_input_ids[graph_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[graph_token_start+1:]
                graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_graph_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

            # summarize the positions of the graphs in the current sample to the structure of "aligned to the batch by the k-th graph"
            if len(batch_boi_list) > 0:
                # if the current maximum graph index is not enough, expand it to the sufficient k dimension
                while len(boi_ids_per_rank) < len(batch_boi_list):
                    boi_ids_per_rank.append([None for _ in range(len(input_ids))])
                    eoi_ids_per_rank.append([None for _ in range(len(input_ids))])
                # write the position of the k-th graph in the current batch
                for k, (cur_boi_k, cur_eoi_k) in enumerate(zip(batch_boi_list, batch_eoi_list)):
                    boi_ids_per_rank[k][batch_idx] = cur_boi_k
                    eoi_ids_per_rank[k][batch_idx] = cur_eoi_k

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels, boi_ids_per_rank, eoi_ids_per_rank


    def initialize_graph_tokenizer(self, model_args, tokenizer):

        if model_args.mm_use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_GRAPH_START_TOKEN, DEFAULT_GRAPH_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
