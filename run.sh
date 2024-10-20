#!/bin/bash

# プロセスのチェック
pid=$(pgrep -f "/home/yamamoto/workspace/research/edtmsr/venv/bin/python3 train.py")

if [ -z "$pid" ]
then
    # プロセスが見つからなかった場合に実行
    nohup /home/yamamoto/workspace/research/edtmsr/venv/bin/python3 train.py > result.txt 2>&1 &
    echo "train.py をバックグラウンドで実行しました。"
else
    echo "train.py は既に実行中です。"
    read -p "実行中のプロセスを中断しますか？ (y/n): " choice
    if [ "$choice" = "y" ]; then
        kill -9 $pid
        echo "実行中のプロセスを中断しました。"
        nohup /home/yamamoto/workspace/research/edtmsr/venv/bin/python3 train.py > result.txt 2>&1 &
        echo "train.py をバックグラウンドで再実行しました。"
    else
        echo "プロセスを中断せずに終了します。"
    fi
fi
