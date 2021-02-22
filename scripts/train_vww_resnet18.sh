#!/bin/sh
./train.py --epochs 500 --optimizer Adam --lr 0.00064 --compress schedule-cifar100-ressimplenet.yaml --model ai85resnet18 --dataset VisualWakeWord --device MAX78000 --batch-size 64 --print-freq 100 --validation-split .1 "$@"
