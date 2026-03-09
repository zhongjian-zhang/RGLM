#!/bin/bash
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=2

mode="v1"  # use 'reconglm_llama_2' for llama and "v1" for others
dataset="cora"  # test dataset
task="nc"  # test task
emb="sbert"  # sbert, gcn_sbert, lgd_sbert

use_hop=2
sample_size=10
template="ND"  # or ND

model_path=${1:-}
output_path=${model_path}/${dataset}_${task}_${emb}_${use_hop}_${sample_size}_${template}.json
model_base=/local/zzj/llm/vicuna-7b-v1.5-16k/snapshots/c8df3ca4436a3bce5c4b5877e0117032081852b4  #meta-llama/Llama-2-7b-hf


python eval/eval_pretrain.py \
--model_path ${model_path} \
--model_base ${model_base} \
--conv_mode  ${mode} \
--dataset ${dataset} \
--pretrained_embedding_type ${emb} \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--answers_file ${output_path} \
--task ${task} \
--cache_dir ../../checkpoint \
--template ${template} \

python eval/eval_res.py \
--res_path ${output_path} \
--task ${task} \
--dataset ${dataset} \
