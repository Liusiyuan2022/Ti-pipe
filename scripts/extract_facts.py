from zhipuai import ZhipuAI
import json
import os
import conf
import base64
from PIL import Image
from tqdm import tqdm
import time
import re
from batchapi import upload_task, download_result
import argparse

TASK="extract"

# 自定义dump_jsonl 和 parse_filter_jsonl函数

# 将一张图片路径在其中处理
def dump_jsonl(i, img_path, f):
    return

# 解析返回的jsonl文件
def parse_filter_jsonl(input_path, result_path):
    return

if __name__ == "__main__":
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Upload or download tasks.")
    parser.add_argument("--action", type=str, required=True, choices=["upload", "download"],help="Action to perform: 'upload' to upload tasks, 'download' to download results.")
    args = parser.parse_args()
    if args.action == "upload":
        # data是全部页面图片
        # 找到index2img_filename.txt,每行提取出来,作为图片文件名的list
        namefile = os.path.join(conf.IMG_PAGE_DIR, 'index2img_filename.txt')
        if not os.path.exists(namefile):
            print(f"index2img_filename.txt not found in {conf.IMG_PAGE_DIR}.")
            quit()
        with open(namefile, 'r') as f:
            img_names = f.read().split('\n')
        
        # 每页作为一个请求
        img_paths = [os.path.join(conf.IMG_PAGE_DIR, img_name) for img_name in img_names]
        
        upload_task(img_paths, dump_jsonl, TASK)
    elif args.action == "download":
        download_result(parse_filter_jsonl, TASK)
