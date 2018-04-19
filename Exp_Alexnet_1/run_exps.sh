#!/bin/bash
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=0 --lr=2e-4 --freeze_layers=1 --exp_name_suffix='freeze_1_extra';
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=0 --lr=2e-4 --freeze_layers=2 --exp_name_suffix='freeze_2_extra';
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=0 --lr=2e-4 --freeze_layers=3 --exp_name_suffix='freeze_3_extra'