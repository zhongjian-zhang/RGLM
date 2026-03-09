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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..reconglm_arch import ReconglmMetaModel, ReconglmMetaForCausalLM
from torch_geometric.data import Data
from utils.constants import IGNORE_INDEX
from dataclasses import dataclass
from utils.constants import DEFAULT_GRAPH_PAD_ID, IGNORE_INDEX
from transformers.modeling_outputs import ModelOutput
import torch.nn.functional as F


@dataclass
class CausalLMOutputWithPastWithGraph(ModelOutput):
    lm_loss: Optional[torch.FloatTensor] = None
    gm_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    feat_loss: Optional[torch.FloatTensor] = None
    topo_loss: Optional[torch.FloatTensor] = None
    

class ReconglmConfig(LlamaConfig):
    model_type = "reconglm"


class ReconglmLlamaModel(ReconglmMetaModel, LlamaModel):
    config_class = ReconglmConfig

    def __init__(self, config: LlamaConfig):
        super(ReconglmLlamaModel, self).__init__(config)


class ReconglmLlamaForCausalLM(LlamaForCausalLM, ReconglmMetaForCausalLM):
    config_class = ReconglmConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ReconglmLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        graph: Optional[torch.FloatTensor] = None,
        graph_emb: Optional[torch.FloatTensor] = None,
        graph_data: Optional[Data] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, boi_ids_per_rank, eoi_ids_per_rank = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, graph, graph_emb)
        # print("graph_emb", graph_emb.mean().item(), graph_emb.std().item(), "inputs_embeds", inputs_embeds.mean().item(), inputs_embeds.std().item()) 
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss, gm_loss, feat_loss, topo_loss, lm_loss = None, None, None, None, None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            lm_loss = loss_fct(shift_logits, shift_labels)
            loss = lm_loss.clone()
            
        if self.training and getattr(self.config, 'mm_inv_projector_type', None) is not None:
            model_type = self.config.mm_inv_projector_type
            num_ranks = len(boi_ids_per_rank)
            gm_loss_total = 0
            feat_loss_total = 0 if model_type.startswith("decoder") else None
            topo_loss_total = 0 if model_type.startswith("decoder") else None
            for rank_idx in range(num_ranks):
                boi_ids = boi_ids_per_rank[rank_idx]
                eoi_ids = eoi_ids_per_rank[rank_idx]
                cur_graph_data_list = []
                for b_idx in range(len(boi_ids)):
                    if boi_ids[b_idx] == None or eoi_ids[b_idx] == None:
                        cur_graph_data_list.append(None)
                        continue
                    cur_graph = graph_data[b_idx][rank_idx]
                    cur_graph_data_list.append(cur_graph)

                if model_type.startswith("denoiser"):
                    cur_gm_loss = self.reconstruct_latent_noisy_loss(cur_graph_data_list, hidden_states.clone(), boi_ids, eoi_ids)
                    gm_loss_total += self.config.denoised_loss_weight * cur_gm_loss if self.config.denoised_loss_weight >0 else 0
                elif model_type.startswith("similarizer"):
                    cur_gm_loss = self.reconstruct_latent_similarity_loss(cur_graph_data_list, hidden_states.clone(), boi_ids, eoi_ids)
                    gm_loss_total += self.config.similarity_loss_weight * cur_gm_loss if self.config.similarity_loss_weight >0 else 0
                elif model_type.startswith("decoder"):
                    cur_feat_loss, cur_topo_loss = self.reconstruct_input_graph_loss(cur_graph_data_list, hidden_states.clone(), boi_ids, eoi_ids)
                    cur_feat_loss = self.config.feat_loss_weight * cur_feat_loss if self.config.feat_loss_weight >0 else 0
                    cur_topo_loss = self.config.topo_loss_weight * cur_topo_loss if self.config.topo_loss_weight >0 else 0
                    if cur_feat_loss > 0:
                        gm_loss_total += cur_feat_loss 
                        feat_loss_total += cur_feat_loss
                    if cur_topo_loss > 0:
                        gm_loss_total += cur_topo_loss 
                        topo_loss_total += cur_topo_loss
            
            gm_loss = gm_loss_total/num_ranks if gm_loss_total > 0 else None
            feat_loss = feat_loss_total/num_ranks if model_type.startswith("decoder") and feat_loss_total > 0 else None
            topo_loss = topo_loss_total/num_ranks if model_type.startswith("decoder") and topo_loss_total > 0 else None
            if gm_loss_total > 0:
                loss += gm_loss 
                

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastWithGraph(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            lm_loss=lm_loss,
            gm_loss=gm_loss,
            feat_loss=feat_loss,
            topo_loss=topo_loss,
        )


    def aggregate_node_hidden_states(self, node_list, agg_dict, hidden_states, agg_method="mean"):
        res_hidden_states = []
        for node in node_list:
            hidden_state = hidden_states[agg_dict[node.item()]]
            if agg_method == "mean":
                agg_hidden_state = torch.mean(hidden_state, dim=0)
            elif agg_method == "max":
                agg_hidden_state = torch.max(hidden_state, dim=0).values
            elif agg_method == "sum":
                agg_hidden_state = torch.sum(hidden_state, dim=0)
            elif agg_method == "last":
                agg_hidden_state = hidden_state[-1]
            else:
                raise ValueError(f"Invalid aggregation method: {agg_method}")
            res_hidden_states.append(agg_hidden_state)
        return torch.stack(res_hidden_states, dim=0)


    def reconstruct_latent_noisy_loss(self, graph_data, hidden_states, boi_ids, eoi_ids):
        new_denoised_x, new_hidden_states = [], []
        for cur_boi_id, cur_eoi_id, cur_graph_data, cur_hidden_state in zip(boi_ids, eoi_ids, graph_data, hidden_states):
            if cur_boi_id is None or cur_eoi_id is None:
                continue
            node_list = cur_graph_data.node_list
            agg_dict = cur_graph_data.agg_dict
            denoised_x = cur_graph_data.denoised_x
            cur_all_graph_hidden_states = cur_hidden_state[cur_boi_id: cur_eoi_id + 1]
            cur_graph_hidden_states = self.aggregate_node_hidden_states(node_list, agg_dict, cur_all_graph_hidden_states, self.config.agg_method)
            assert cur_graph_hidden_states.shape[0] == denoised_x.shape[0]
            new_denoised_x.append(denoised_x)
            new_hidden_states.append(cur_graph_hidden_states)

        if any(x.shape != new_denoised_x[0].shape for x in new_denoised_x):
            max_len = max(x.shape[0] for x in new_denoised_x)
            new_denoised_x_align, new_hidden_states_align, pad_mask = [], [], []
            for cur_x, cur_hidden_state in zip(new_denoised_x, new_hidden_states):
                valid_mask = torch.full((cur_x.shape[0],), True, dtype=torch.bool, device=cur_x.device)
                padding_mask = torch.full((max_len - cur_x.shape[0],), False, dtype=torch.bool, device=cur_x.device)
                cur_pad_mask = torch.cat([valid_mask, padding_mask], dim=0)
                
                cur_new_x = torch.cat((cur_x, torch.zeros((max_len - cur_x.shape[0], cur_x.shape[1]), dtype=cur_x.dtype, device=cur_x.device)), dim=0)
                cur_new_hidden_state = torch.cat((cur_hidden_state, 
                torch.zeros((max_len - cur_hidden_state.shape[0], cur_hidden_state.shape[1]), dtype=cur_hidden_state.dtype, device=cur_hidden_state.device)), dim=0)
                pad_mask.append(cur_pad_mask)
                new_denoised_x_align.append(cur_new_x)
                new_hidden_states_align.append(cur_new_hidden_state)

            new_denoised_x = torch.stack(new_denoised_x_align, dim=0)
            new_hidden_states = torch.stack(new_hidden_states_align, dim=0)
            pad_mask = torch.stack(pad_mask, dim=0)
        else:
            new_denoised_x = torch.stack(new_denoised_x, dim=0)
            new_hidden_states = torch.stack(new_hidden_states, dim=0)
            pad_mask = torch.full((new_denoised_x.shape[0], new_denoised_x.shape[1]), 
                                  True, dtype=torch.bool, device=new_denoised_x.device)
        
        with torch.amp.autocast('cuda', dtype=torch.float32):
            processed_graph_hidden = self.model.mm_inv_projector.ln_pre(new_hidden_states.to(torch.float32))
            gm_loss = self.model.mm_inv_projector(
                z=processed_graph_hidden.repeat(4, 1, 1).contiguous().float(),
                target=new_denoised_x.repeat(4, 1, 1).contiguous().float(),
                node_mask=pad_mask.repeat(4, 1).contiguous().bool(),
            )
        gm_loss = gm_loss.float().mean()
        return gm_loss


    def reconstruct_latent_similarity_loss(self, graph_data, hidden_states, boi_ids, eoi_ids):
        batch_similarity_loss = 0
        batch_size = len(graph_data)
        for cur_boi_id, cur_eoi_id, cur_graph_data, cur_hidden_state in zip(boi_ids, eoi_ids, graph_data, hidden_states):
            if cur_boi_id is None or cur_eoi_id is None:
                continue
            node_list = cur_graph_data.node_list
            agg_dict = cur_graph_data.agg_dict
            denoised_x = cur_graph_data.denoised_x
            cur_all_graph_hidden_states = cur_hidden_state[cur_boi_id: cur_eoi_id + 1]
            cur_graph_hidden_states = self.aggregate_node_hidden_states(node_list, agg_dict, 
                                                                        cur_all_graph_hidden_states, 
                                                                        self.config.agg_method)
            assert cur_graph_hidden_states.shape[0] == denoised_x.shape[0]
            with torch.amp.autocast('cuda', dtype=torch.float32):
                hidden_states = self.model.mm_inv_projector(cur_graph_hidden_states.to(torch.float32))
                u = F.normalize(hidden_states, p=2, dim=-1)
                v = F.normalize(denoised_x, p=2, dim=-1)
                cosine_similarity = (u * v).sum(dim=-1).clamp(min=-1.0, max=1.0)
                loss_per_token = 1.0 - cosine_similarity
                batch_similarity_loss += loss_per_token.sum().float() / (loss_per_token.shape[0] + 1e-8)
        batch_similarity_loss = batch_similarity_loss / batch_size
        return batch_similarity_loss


    def reconstruct_input_graph_loss(self, graph_data, hidden_states, boi_ids, eoi_ids):
        batch_feat_recon_loss, batch_topo_recon_loss = 0, 0
        batch_size = len(graph_data)
        
        for cur_boi_id, cur_eoi_id, cur_graph_data, cur_hidden_state in zip(boi_ids, eoi_ids, graph_data, hidden_states):
            if cur_boi_id is None or cur_eoi_id is None:
                continue
            edge_index = cur_graph_data.edge_index
            node_list = cur_graph_data.node_list
            agg_dict = cur_graph_data.agg_dict
            origin_graph_emb = cur_graph_data.x
            cur_all_graph_hidden_states = cur_hidden_state[cur_boi_id: cur_eoi_id + 1]
            cur_graph_hidden_states = self.aggregate_node_hidden_states(node_list, agg_dict, 
                                                                        cur_all_graph_hidden_states, 
                                                                        self.config.agg_method)
            assert cur_graph_hidden_states.shape[0] == origin_graph_emb.shape[0]
            with torch.amp.autocast('cuda', dtype=torch.float32):
                feat_recon_loss, topo_recon_loss = self.model.mm_inv_projector(cur_graph_hidden_states.to(torch.float32), 
                                                                               origin_graph_emb, edge_index, 
                                                                               self.config.topo_recon_ratio)
            batch_feat_recon_loss += feat_recon_loss.float()
            batch_topo_recon_loss += topo_recon_loss.float()
            
        batch_feat_recon_loss = batch_feat_recon_loss / batch_size
        batch_topo_recon_loss = batch_topo_recon_loss / batch_size
        return batch_feat_recon_loss, batch_topo_recon_loss
    
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph": kwargs.get("graph", None),
                "graph_emb": kwargs.get("graph_emb", None),
            }
        )
        return model_inputs

AutoConfig.register("reconglm", ReconglmConfig)
AutoModelForCausalLM.register(ReconglmConfig, ReconglmLlamaForCausalLM)
