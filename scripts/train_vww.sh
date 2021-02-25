#!/bin/sh
./train.py --epochs 500 --optimizer Adam --lr 0.0001 --compress schedule-cifar100-ressimplenet.yaml --model ai85mobilenetsv2 --dataset VisualWakeWord --device MAX78000 --batch-size 64 --print-freq 100 --confusion --validation-split .1 --use-bias "$@"
