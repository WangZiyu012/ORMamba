#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
ep=40
model_name=SOR_Mamba
lamb=0.001
echo "Running with lamb=$lamb"
python -u run_SSL_PT.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512\
  --learning_rate 0.00005 \
  --d_state 2 \
  --d_ff 512\
  --train_epochs $ep\
  --lamb $lamb\
  --itr 1
