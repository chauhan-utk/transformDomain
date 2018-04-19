#!/bin/bash
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=0 --lr=2e-4 --freeze_layers=0 --exp_name_suffix='freeze_no_conv';
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=0 --lr=2e-4 --freeze_layers=1 --exp_name_suffix='freeze_first_conv';
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=0 --lr=2e-4 --freeze_layers=2 --exp_name_suffix='freeze_0_1_conv';
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=0 --lr=2e-4 --freeze_layers=3 --exp_name_suffix='freeze_0_1_2_conv';
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=0 --lr=2e-4 --freeze_layers=4 --exp_name_suffix='freeze_0_1_2_3_conv'