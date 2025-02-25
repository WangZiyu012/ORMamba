#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
ep=40
model_name=SOR_Mamba
lamb=0.001
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 4 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 1024 \
  --learning_rate 0.0003 \
  --itr 1 \
  --train_epochs $ep\
  --lamb $lamb\
  --use_norm 0
