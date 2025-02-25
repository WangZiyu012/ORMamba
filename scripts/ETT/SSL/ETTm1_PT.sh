#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
ep=40
model_name=SOR_Mamba
lamb=0.01
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --lamb $lamb\
  --train_epochs $ep\
  --d_ff 256 \
  --itr 1