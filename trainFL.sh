#!/bin/bash

# シェルスクリプトのファイル名: run_experiments.sh

# 1 ~ 10 の範囲で num_clients を変化させて実行
for num_clients in {2..10}
do
  echo "Running experiment with num_clients=$num_clients"
  nohup python3 train_fl.py --model_type srtdn --num_epochs 100 --num_rounds 100 --num_clients $num_clients > output_num_clients_$num_clients.log 2>&1 &
done

echo "All experiments have been started."
