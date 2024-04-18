#!/bin/bash

if [ -z "$1" ]; then
    config="tiny atari"
else
    config="$1"
fi

if [ -z "$2" ]; then
    task="atari_pong"
else
    task="$2"
fi

python3 director/train.py \
	--logdir /logdir/$(date +%Y%m%d-%H%M%S) \
	--configs=$config \
	--task=$task
