#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
ep=80
model_name=SOR_Mamba

lp=0
lamb=0.001 #lr

python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07_96_12 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 12 \
  --e_layers 2 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16\
  --learning_rate 0.0007 \
  --lamb $lamb\
  --train_epochs $ep\
  --learning_rate_alpha 0.2\
  --lp $lp\
  --itr 1 \
  --use_norm 0

python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07_96_24 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 2 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0007 \
  --lamb $lamb\
  --train_epochs $ep\
  --learning_rate_alpha 0.2\
  --lp $lp\
  --itr 1 \
  --use_norm 0

python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07_96_48 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 2 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --lamb $lamb\
  --train_epochs $ep\
  --learning_rate_alpha 2\
  --lp $lp\
  --itr 1 \
  --use_norm 0

  python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07_96_96 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16\
  --learning_rate 0.0005 \
  --lamb $lamb\
  --train_epochs $ep\
  --learning_rate_alpha 1\
  --lp $lp\
  --itr 1 \
  --use_norm 0