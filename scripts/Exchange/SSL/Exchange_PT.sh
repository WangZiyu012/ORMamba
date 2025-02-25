#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
ep=40
model_name=SOR_Mamba
lamb=0.01
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --batch_size 16\
  --learning_rate 0.0001 \
  --d_ff 128 \
  --train_epochs $ep\
  --lamb $lamb\
  --itr 1