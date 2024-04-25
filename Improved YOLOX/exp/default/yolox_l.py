#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        
        # Define yourself dataset path
        self.data_dir = "datasets/USD_COCO"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.test_ann = "test.json"
        # クラス数の変更
        self.num_classes=7
	## 評価間隔を変更（初期では10epochごとにしか評価が回らない）
        self.eval_interval = 1
