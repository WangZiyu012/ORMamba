#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
ep=40
model_name=SOR_Mamba

lp=0
lamb=0.01
python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --train_epochs $ep\
  --lp $lp\
  --lamb $lamb\
  --learning_rate_alpha 0.5\
  --itr 1

python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
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
  --learning_rate 0.00005 \
  --train_epochs $ep\
  --lp $lp\
  --lamb $lamb\
  --learning_rate_alpha 0.2\
  --d_ff 256 \
  --itr 1

python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
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
  --learning_rate 0.00003 \
  --train_epochs $ep\
  --lp $lp\
  --lamb $lamb\
  --learning_rate_alpha 2\
  --d_ff 256 \
  --itr 1

python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
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
  --learning_rate 0.00005 \
  --train_epochs $ep\
  --lp $lp\
  --lamb $lamb\
  --learning_rate_alpha 5\
  --d_ff 256 \
  --itr 1