#!/bin/bash

python vgg16_train.py --epochs=1000 --gpu=1 --batch_size=32 --lr=2e-4 --net_wtDcy=0.0001 --net_biasDcy=0.0001 --btl_wtDcy=0.0001 --btl_biasDcy=0.0001 --exp_name_suffix='wtDcy_to_all';
 python vgg16_train.py --epochs=1000 --gpu=1 --batch_size=32 --lr=2e-4 --net_wtDcy=0.0001 --net_biasDcy=0. --btl_wtDcy=0.0001 --btl_biasDcy=0. --exp_name_suffix='wtDcy_nt_bias';
 python vgg16_train.py --epochs=1000 --gpu=1 --batch_size=32 --lr=2e-4 --net_wtDcy=0. --net_biasDcy=0. --btl_wtDcy=0. --btl_biasDcy=0. --exp_name_suffix='no_wtDcy';
 python vgg16_train.py --epochs=1000 --gpu=1 --batch_size=32 --lr=2e-5 --net_wtDcy=0.0001 --net_biasDcy=0.0001 --btl_wtDcy=0.0001 --btl_biasDcy=0.0001 --exp_name_suffix='wtDcy_to_all';
 python vgg16_train.py --epochs=1000 --gpu=1 --batch_size=32 --lr=2e-6 --net_wtDcy=0.0001 --net_biasDcy=0.0001 --btl_wtDcy=0.0001 --btl_biasDcy=0.0001 --exp_name_suffix='wtDcy_to_all';