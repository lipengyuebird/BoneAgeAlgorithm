#!venv/Scripts/python
# -*- coding: utf-8 -*-
# @file     : config_loader.py
# @author   : lipengyu @CAICT
# @datetime : 2021/7/8 17:41 
# @software : PyCharm

import yaml
from pathlib import Path

# locate the config file and read
with open(Path(__file__).parent.parent / 'config.yml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
