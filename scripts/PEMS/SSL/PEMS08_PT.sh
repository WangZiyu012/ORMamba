#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
ep=40
model_name=SOR_Mamba
lamb=0.001 #lr
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_epochs $ep\
  --lamb $lamb\
  --itr 1 \
  --learning_rate 0.0001 \
  --use_norm 1
