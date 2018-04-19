#!/bin/bash
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=1 --lr=1e-4;
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=1 --lr=1e-4 --net_biasDcy=0. --btl_biasDcy=0. --exp_name_suffix='no_bias_dcy_ntbtl';
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=1 --lr=2e-4 --net_biasDcy=0. --btl_biasDcy=0. --exp_name_suffix='no_bias_dcy_ntbtl';
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=1 --lr=4e-4 --net_biasDcy=0. --btl_biasDcy=0. --exp_name_suffix='no_bias_dcy_ntbtl';
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=1 --lr=1e-6 --net_biasDcy=0. --btl_biasDcy=0. --exp_name_suffix='no_bias_dcy_ntbtl';
python alexnet_source_only.py --epochs=1000 --batch_size=64 --gpu=1 --lr=1e-6;