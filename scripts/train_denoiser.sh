#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
max_len=4096
sample_size=10

task=${2:-"nc"}
dataset=${3-"cora.3"}
bs=${4:-16}
emb=${5:-"sbert"}
MM_INV_PROJECTOR_TYPE="denoiser_git3x"
DENOISED_EMBEDDING_TYPE="lgd_sbert"
denoised_loss_weight=1.0

mm_projector_lr=2e-3
mm_inv_projector_lr=1e-4    # cora(1e-4)    pubmed(5e-4)    reddit(2e-4)    arxiv(5e-4)
learning_rate=1e-4          # cora(1e-4)    pubmed(5e-4)    reddit(2e-4)    arxiv(5e-4)

lora_r=8             # cora(8)     pubmed(16)    reddit(8)      arxiv(8)
lora_alpha=32        # cora(32)    pubmed(32)    reddit(16)     arxiv(16)
lora_dropout=0.1     # cora(0.1)   pubmed(0.05)  reddit(0.05)   arxiv(0.1)

use_hop=2
template="ND"
projector_type="linear"
prefix=reconglm_denoiser-vicuna-7b-${emb}-${use_hop}-${sample_size}-${projector_type}-projector-${DENOISED_EMBEDDING_TYPE}-${MM_INV_PROJECTOR_TYPE}-w_${denoised_loss_weight}
model_base=/local/zzj/llm/vicuna-7b-v1.5-16k/snapshots/c8df3ca4436a3bce5c4b5877e0117032081852b4
mode="v1"

echo "PREFIX:  ${prefix}"

wandb offline

python  train/train_mem.py \
--model_name_or_path ${model_base} \
--version ${mode} \
--cache_dir  /local/zzj/ReconGLM/checkpoints/reconglm_denoiser \
--pretrained_embedding_type ${emb} \
--mm_use_graph_start_end False \
--mm_use_graph_patch_token False \
--output_dir  ./checkpoints/reconglm_denoiser/${dataset}/${prefix}_${task} \
--num_train_epochs 1 \
--per_device_train_batch_size ${bs} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--model_max_length ${max_len} \
--gradient_checkpointing True \
--lazy_preprocess True \
--report_to wandb \
--use_hop ${use_hop} \
--sample_neighbor_size ${sample_size} \
--mm_projector_type ${projector_type} \
--use_task ${task} \
--use_dataset ${dataset} \
--template ${template} \
--mm_inv_projector_type ${MM_INV_PROJECTOR_TYPE} \
--denoised_embedding_type ${DENOISED_EMBEDDING_TYPE} \
--mm_projector_lr 2e-3 \
--mm_inv_projector_lr ${mm_inv_projector_lr} \
--learning_rate ${learning_rate} \
--tune_mm_mlp_adapter False \
--lora_enable True \
--bf16 True \
--tf32 True \
--denoised_loss_weight ${denoised_loss_weight} \
--agg_method mean \
--lora_r ${lora_r} \
--lora_alpha ${lora_alpha} \
--lora_dropout ${lora_dropout} \
