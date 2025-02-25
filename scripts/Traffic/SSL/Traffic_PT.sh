#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ep=40
model_name=SOR_Mamba

lp=0
lamb=0.001 # lr
echo "Running with lamb=$lamb"
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --train_epochs $ep\
  --lamb $lamb\
  --learning_rate 0.0001 \
  --itr 1