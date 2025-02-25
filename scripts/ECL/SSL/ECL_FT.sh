#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
ep=40
model_name=SOR_Mamba

lp=0
lamb=0.0001
python -u run_SSL_FT.py \
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
  --learning_rate 0.001 \
  --lamb $lamb\
  --train_epochs $ep\
  --lp $lp\
  --learning_rate_alpha 0.2\
  --itr 1

python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_state 16 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --lamb $lamb\
  --train_epochs $ep\
  --lp $lp\
  --learning_rate_alpha 1\
  --itr 1

python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --d_state 16 \
  --learning_rate 0.0005 \
  --lamb $lamb\
  --train_epochs $ep\
  --lp $lp\
  --learning_rate_alpha 1\
  --itr 1

python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --lamb $lamb\
  --d_state 16 \
  --train_epochs $ep\
  --lp $lp\
  --learning_rate_alpha 1\
  --learning_rate 0.0005 \
  --itr 1