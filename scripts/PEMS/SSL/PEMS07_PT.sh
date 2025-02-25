#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
ep=40
model_name=SOR_Mamba
lamb=0.001 #lr
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16\
  --learning_rate 0.0001 \
  --train_epochs $ep\
  --lamb $lamb\
  --itr 1 \
  --use_norm 0