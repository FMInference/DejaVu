# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/12/22 23:00:33
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import requests

from tqdm import tqdm
import requests
from filelock import FileLock

def download_with_progress_bar(save_path, url):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pbar = tqdm(total=int(r.headers['Content-Length']), unit_scale=True)
            for chunk in r.iter_content(chunk_size=32 * 1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

MODEL_ULRS = {
    'ice_text.model': 'https://cloud.tsinghua.edu.cn/f/2c73ea6d3e7f4aed82ec/?dl=1',
    'ice_image.pt': 'https://cloud.tsinghua.edu.cn/f/ae2cd37af814429d875d/?dl=1'
}

def auto_create(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock = FileLock(file_path + '.lock')
    with lock:
        if os.path.exists(file_path):
            return False 
        else:
            url = MODEL_ULRS[file_path.split('/')[-1]]
            print(f'Downloading tokenizer models {url} into {file_path} ...')
            download_with_progress_bar(file_path, url)
            return True