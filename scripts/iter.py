from zhipuai import ZhipuAI
import json
import os
import conf
import base64
from PIL import Image
from tqdm import tqdm
import time
import re
from batchapi import upload_task, download_result, get_resp_id, get_resp_content
import argparse
import random

TASK="iter_0"

# 创建task标签
# 找到下个任务的TASK号，同时返回最近的iter序号
def get_latest_iter():
    # 在文件夹中查找iter_*.jsonl文件
    iter_files = [f for f in os.listdir(conf.DATASET_DIR) if f.startswith('iter_') and f.endswith('.jsonl')]
    # 找到最新的iter号
    prev_idx = max([int(f.split('_')[1].split('.')[0]) for f in iter_files], default=0)
    TASK = f"iter_{prev_idx + 1}"
    print(f"prev is {prev_idx}th iter , now TASK: {TASK}")
    return os.path.join(conf.DATASET_DIR, f'iter_{prev_idx}.jsonl')

def pick_one_method():
    # 从字典中随机选择一个方法,
    return random.choice(METHOD_LIST)

PROMPT = """你是一个资深的模型数据工程师,你需要对已经出好的问答对进行优化重写。
[任务描述]：你需要对给定的prompt进行重写，使用更复杂的语言和结构来构造重写问答对以及相应的解析，以使得这些著名的AI系统（例如chatgpt和GPT4）更难处理。同时，重写的问答对必须合理，并且能够被人类理解和响应。
[核心方法]：{{}}
[输入]：问题question，答案answer，解析analysis，题型sub_type，以及所基于的知识块base_knowledge。
[限制条件]：1.你应该尽量避免使重写的prompt变得冗长，重写的prompt只能在原始prompt中添加10到20个单词。2.处理后的题型sub_type不能改变，仍然是对应题型 3. 可以使用知识块中的信息进行一定程度的丰富
以下是你需要处理的具体内容：
{{}}
[输出格式]：请严格按照json格式返回重写过的问答对，只包含题型sub_type,问题question，答案answer，解析analysis四部分
[输出示例]：
{
    "sub_type": "statement_judgment",
    "question": "在多盘摩擦离合器中，如果摩擦系数f增加三倍，，而其他条件不变，那么可传递的最大转矩T_max会增加九倍", 
    "answer": "错误", 
    "analysis": "根据公式$$T_{{max}} = fFR$$，可以看出最大转矩$T_{{max}}$与摩擦系数$f$成正比。因此，在其他条件（F和R）不变的情况下，摩擦系数$f$增加三倍，最大转矩$T_{{max}}$应该增加三倍，而不是九倍。"
}
"""

METHOD_LIST = [
    ("constraint", "添加一些约束条件"),
    ("deepen_issue", "如果问题包括特定的事例和场景，对这些事例和场景进行细化，使之略微复杂一些"),
    ("general_specific", "对比较宽泛的概念进行具体细化"),
    ("multiple_step", "如果解决这个问题需要几步的推理过程，再增加一点推理的步骤")
]


# 自定义dump_jsonl 和 parse_filter_jsonl函数

# 将一张图片路径在其中处理
def dump_jsonl(i, prev_data, f):
    
    source = prev_data["source"]
    
    # original_qa_info 不包括source这个字段
    original_qa_info = json.dumps({
        "sub_type": prev_data["sub_type"],
        "question": prev_data["question"],
        "answer": prev_data["answer"],
        "analysis": prev_data["analysis"],
        "base_knowledge": prev_data["base_knowledge"]
    }, ensure_ascii=False)
    
    (method, desc) = pick_one_method()
    
    # 生成新的prompt
    content = PROMPT.format(desc, original_qa_info)
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
        
                
    f.write(json.dumps({
        "custom_id": f"request-{i}-<{method}>-<{source}>",
        "method": "POST",
        "url": "/v4/chat/completions",
        "body": {
            "model": "glm-4-plus",
            "messages": messages,
            "response_format":{
                'type': 'json_object'
            },
            "max_tokens": 2048,
        }
    }, ensure_ascii=False) + '\n')
    
    return

# 解析返回的jsonl文件到result_path,以append的方式
def parse_filter_jsonl(input_path, result_path):
    i = 1
    tot = 0
    low_conf = 0
    err_num = 0
    print(f"Parsing and Filtering JSONL file: {input_path}")  
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # 提取 custom_id
            request_id = get_resp_id(data)
            # 形如ssource<EEdesign.pdf_0.png>,提取
            pattern = r"request-(\d+)-<([^>]*)>-<([^>]*)>"
            match = re.search(pattern, request_id)
            if match:
                source = match.group(3)
            else:
                print(f"Failed to parse request_id at line: {i}, request_id: {request_id}")
                continue
            # 提取返回的内容,经过字符处理
            try:
                content = get_resp_content(data)
                
                sub_type = content.get("sub_type", "N/A")
                question = content.get("question", "N/A")
                answer   = content.get("answer", "N/A")
                analysis = content.get("analysis", "N/A")
                with open(result_path, 'a') as out_f:
                    out_f.write(json.dumps({
                        "source": source,
                        # "task": task,
                        "sub_type": sub_type,
                        "question": question,
                        "answer": answer,
                        "analysis": analysis
                    }, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError:
                print(f"Failed to parse result content as JSON at line: {i}")
                err_num += 1
            except KeyError as e:
                print(f"KeyError: {e} at line: {i}")
                err_num += 1
            tot += 1
            i += 1
    print(f"Total lines: {tot}, Low confidence lines: {low_conf}, Errors: {err_num}")
    print(f"Filtered results of {input_path} saved to {result_path}")
    
    return

if __name__ == "__main__":
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Upload or download tasks.")
    parser.add_argument("--action", type=str, required=True, choices=["upload", "download"],help="Action to perform: 'upload' to upload tasks, 'download' to download results.")
    args = parser.parse_args()
    if args.action == "upload":
        # data
        fact_path = os.path.join(conf.DATASET_DIR, 'extract.jsonl')
        # 每行是一个json对象，组织成一个字典，键值为source,值为facts
        src_facts = {}
        with open(fact_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                src_facts[data["source"]] = data["facts"]
        
        prev_iter_path = get_latest_iter()
        # 每行是一个json对象，组织成一个列表
        # 包括问答对信息和对应的facts部分
        prev_data = []
        with open(prev_iter_path, 'w') as prev_f:
            for line in tqdm(prev_f, desc="Processing data to next iter"):
                data = json.loads(line)
                sub_type = data["sub_type"]
                question = data["question"]
                answer = data["answer"]
                analysis = data["analysis"]
                source = data["source"]
                base_knowledge = src_facts[source]
                prev_data.append({
                    "source": source,
                    "sub_type": sub_type,
                    "question": question,
                    "answer": answer,
                    "analysis": analysis,
                    "base_knowledge": base_knowledge,
                })
        
        upload_task(prev_data, dump_jsonl, TASK)
    elif args.action == "download":
        get_latest_iter()
        
        download_result(parse_filter_jsonl, TASK)
