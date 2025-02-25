#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ep=40
model_name=SOR_Mamba
lamb=0.01
echo "Running with lamb=$lamb"
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs $ep\
  --lamb $lamb\
  --itr 1
