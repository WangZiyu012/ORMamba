#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ep=40
model_name=SOR_Mamba
lamb=0.001 #lr
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0001 \
  --train_epochs $ep\
  --lamb $lamb\
  --itr 1