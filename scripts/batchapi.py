
from zhipuai import ZhipuAI
import json
import os
import conf
import base64
from PIL import Image
from tqdm import tqdm
import time
import re

client = ZhipuAI(api_key=conf.ZHIPU_API_KEY)

# data: 一个整理好的列表类型数据，每个元素对应构造一个请求
# dump_jsonl: 一个函数，负责将其中的一条数据按特定要求写入jsonl文件，具体的构造方法由这个函数定义
# 具体要求是：dump_jsonl(i, batch, f)，其中i是当前请求的索引，batch是当前请求的数据，f是打开的文件对象
# task_tag: 任务标签，用于区分不同的任务
def create_batch_jsonl(data, dump_jsonl, task_tag):
    # 创建批量 JSONL 文件
    file_paths = []
    m = -1
    file_path = None
    for i, batch in tqdm(enumerate(data), desc="Creating batch requests", total=len(data)):
        # 如果文件大小超过限制，则创建新的文件
        if file_path is None or i % conf.BATCH_REQ_LIMIT == 0 or os.path.getsize(file_path) >= conf.BATCH_BYTE_LIMIT:
            m += 1
            file_path = os.path.join(conf.BATCH_DIR, f"batch_{task_tag}_{m}.jsonl")
            file_paths.append(file_path)
            f = open(file_path, 'w')
         # 向该文件中写入一条请求
        dump_jsonl(i ,batch, f)
            
    return file_paths




def upload_batchfile(file_paths):
    ids = []
    for file_path in file_paths:
        print(f"Uploading {file_path}")
        upload_file = open(file_path, "rb")
        
        result = client.files.create(
            file=upload_file,
            purpose="batch"
        )
        print(f"Upload success! File ID: {result.id}")
        ids.append(result.id)
    return ids

def submit_batch_task(file_ids, task_tag):
    batch_ids = []
    for i , file_id in enumerate(file_ids):
        print(f"Submitting {task_tag} task for file {file_id}")
        create = client.batches.create(
            input_file_id=file_id,
            endpoint="/v4/chat/completions", 
            auto_delete_input_file=True,
            metadata={
                "description": f"{task_tag} Batch {i}"
            }
        )
        batch_ids.append(create.id)
    batch_id_path = os.path.join(conf.BATCH_DIR, f'batch_ids_{task_tag}.json')
    # # 记录file_ids为一个json文件，用于后续查询以及下载
    with open(batch_id_path, 'w') as f:
        json.dump(batch_ids, f)
    print(f"Submit Success! batch_ids were saved to {batch_id_path}")
        
def check_jobs(batch_ids):
    output_file_ids = []
    statuses = []
    for batch_id in batch_ids:
        batch_job = client.batches.retrieve(batch_id)
        output_file_ids.append(batch_job.output_file_id)
        statuses.append(batch_job.status)
    print("Batch job statuses:")
    for (id, status) in zip(batch_ids, statuses):
        print(f"Batch ID: {id}, Status: {status}")
    for i, status in enumerate(statuses):
        if status != "completed":
            print(f"The {i}th batch task is still in {status}, waiting for completion...")
            print("Donwload canceled.")
            return None
    print("All batch jobs are completed!")
    return output_file_ids

def download_output(output_file_ids, task_tag, parse_filter_jsonl):
    result_path = os.path.join(conf.DATASET_DIR, f'{task_tag}.jsonl')
    # 清空可能的旧内容
    with open(result_path, 'w') as f:
        f.write("")
    # 遍历每个输出文件 ID，下载并解析
    for i, output_file_id in enumerate(output_file_ids):
        # client.files.content返回 _legacy_response.HttpxBinaryResponseContent实例
        print(f"downloading output file {output_file_id}")
        content = client.files.content(output_file_id)
        # 使用write_to_file方法把返回结果写入文件
        output_file_path = os.path.join(conf.BATCH_DIR, f"batch_output_{task_tag}_{i}.jsonl")
        content.write_to_file(output_file_path)
        print(f"Download success! Content was saved to {output_file_path}")
        # 以append形式将这些文件逐个写入到result_path
        print(f"Parsing and filtering JSONL file: {output_file_path}")
        parse_filter_jsonl(output_file_path, result_path)
    print(f"All output files were parsed and filtered to {result_path}")

def upload_task(data, dump_jsonl, task_tag):
    # 创建批量 JSONL 文件
    file_paths = create_batch_jsonl(data, dump_jsonl, task_tag)
    if not file_paths:
        print("No files to upload.")
        return

    # 上传文件并获取文件 ID
    file_ids = upload_batchfile(file_paths)

    # 提交任务并获取批次 ID
    submit_batch_task(file_ids, task_tag)



def download_result(parse_filter_jsonl,task_tag):
    # 从保存的 JSON 文件中加载批次 ID
    batch_ids_path = os.path.join(conf.BATCH_DIR, f'batch_ids_{task_tag}.json')
    if not os.path.exists(batch_ids_path):
        print(f"Batch IDs file not found: {batch_ids_path}")
        return

    batch_ids = json.load(open(batch_ids_path))
    
    # 检查任务状态并获取输出文件 ID
    output_file_ids = check_jobs(batch_ids)
    
    if output_file_ids is None:
        print("Some tasks are still in progress. Exiting...")
        return
    # print(f"batch_ids: {batch_ids}, output_file_ids: {output_file_ids}")
    # 下载输出文件
    download_output(output_file_ids, task_tag, parse_filter_jsonl)

# 用于从返回体中提取特定信息
# 输入是一个返回的json对象
def get_resp_id(data):
    # 提取 custom_id
    return data["response"]["body"]["request_id"]
    
def get_resp_content(data):
    # 提取返回的内容,经过字符处理
    content = data["response"]["body"]["choices"][0]["message"]["content"]
    stripped_content = content.strip("`json").strip()
    content_json = json.loads(stripped_content)
    return content_json
