
#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
ep=40
model_name=SOR_Mamba
lamb=0.001
lp=0
python -u run_SSL_FT.py \
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
  --train_epochs $ep\
  --d_state 2 \
  --d_ff 512\
  --lamb $lamb\
  --learning_rate_alpha 2\
  --lp $lp\
  --itr 1


python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs $ep\
  --d_model 512\
  --d_state 2 \
  --d_ff 512\
  --lamb $lamb\
  --learning_rate_alpha 0.5\
  --lp $lp\
  --itr 1


python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs $ep\
  --d_model 512\
  --d_state 2 \
  --d_ff 512\
  --lamb $lamb\
  --learning_rate_alpha 0.5\
  --lp $lp\
  --itr 1


python -u run_SSL_FT.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --train_epochs $ep\
  --d_model 512\
  --d_state 2 \
  --d_ff 512\
  --lamb $lamb\
  --learning_rate_alpha 1\
  --lp $lp\
  --itr 1
