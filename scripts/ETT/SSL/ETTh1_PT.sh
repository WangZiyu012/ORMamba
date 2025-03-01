#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
ep=40
model_name=SOR_Mamba

lamb=0.01
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_state 2\
  --d_ff 256 \
  --itr 1 \
  --train_epochs $ep\
  --lamb $lamb\
  --learning_rate 0.0001