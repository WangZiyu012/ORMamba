#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ep=40
model_name=SOR_Mamba
lamb=0.0001 # lr
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 16 \
  --batch_size 16 \
  --learning_rate 0.0001 \
  --train_epochs $ep\
  --lamb $lamb\
  --itr 1