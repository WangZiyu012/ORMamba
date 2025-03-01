#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ep=40
model_name=SOR_Mamba
s_data=ETTm2

lp=0
python -u run_SSL_TL.py \
  --is_training 1 --S_data $s_data\
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
  --lp $lp\
  --lamb $lamb\
  --learning_rate_alpha $lra\
  --learning_rate 0.00007

python -u run_SSL_TL.py \
  --is_training 1 --S_data $s_data\
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --lamb $lamb\
  --train_epochs $ep\
  --learning_rate_alpha $lra\
  --lp $lp\
  --learning_rate 0.00007

python -u run_SSL_TL.py \
  --is_training 1 --S_data $s_data\
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --lamb $lamb\
  --train_epochs $ep\
  --learning_rate_alpha $lra\
  --lp $lp\
  --learning_rate 0.00005

python -u run_SSL_TL.py \
  --is_training 1 --S_data $s_data\
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --lamb $lamb\
  --train_epochs $ep\
  --learning_rate_alpha $lra\
  --lp $lp\
  --learning_rate 0.00005