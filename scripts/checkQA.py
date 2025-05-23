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

TASK="checkQA"



PROMPT = """你是一个资深的考试出题人,你需要对已经出好的题目进行审核。
所传入的内容是一个出好的题目，包含四个字段“sub_type”、“question”、“answer”、“analysis”，分别表示题目类型、题目内容、答案和解析。
你的任务是判断题目的质量，给出一个分数，范围是0-10分，0分表示题目质量极差，10分表示题目质量极好。
以下是判断标准：
1. 问题虽然基于教材所出，但考生考场上不知道教材内容，也不知道是哪本书，所以，问题必须围绕知识点，而不能是对于某张图片，某张表格或者某本书信息的提问。如，"表3提里面可以看到什么"、"图2中可以看到什么"、"12.2节介绍了什么"等问题都是不合格的。
2. 问题格式需要完整，如果是选择题，却没有选项，或者是填空题却没有填空内容这种情况,都是不合格的。
3. 问题需要是有价值的。知识领域的知识点，而不能是某本书的编者，出版社等信息的问题。
以下是相关问题：{}
请作出你的判断。
请返回json格式,包括你的理由和最终给分，例如:
```json
{{
    "reason": "该问题和答案聚焦于知识点，且问题格式完整，解析也比较完善，符合出题标准。",
    "score": 9
    
}}
```
或者
```json
{{
    "reason": "该问题问的是编者和出版社等信息，这并不适合作为考试问题。是不合格的。",
    "score": 0
}}
```"""

def quality_sort(qa_path, quality_path, qualified_path, unqualified_path):
    # qa_path是原始的jsonl文件，quality_path是质量评分的jsonl文件
    # quality的id是testpath的行号-1
    quality_data = []
    with open(quality_path, 'r') as f :
        for line in f:
            data = json.loads(line)
            # 提取 "id" 和 "score" 字段
            id = data["id"]
            score = data["score"]
            reason = data["reason"]
            quality_data.append({
                "id": id,
                "score": score,
            })
    # 根据id排序,将排好的score拿出来
    quality_data.sort(key=lambda x: int(x["id"]))
    scores = [data["score"] for data in quality_data]
    qnum = 0
    unum = 0
    # 读取qa_path的内容,根据对应的得分分类到两个文件
    with open(qa_path, 'r') as f, open(qualified_path, 'w') as qf, open(unqualified_path, 'w') as uf:
        for i, line in enumerate(f):
            # score = scores[i]
            score = scores[i]
            if score >= conf.QA_QUALITY_THRESHOLD:
                qf.write(line)
                qnum += 1
            else:
                uf.write(line)
                unum += 1
    print(f"Sorted and filtered QA data based on quality scores.")
    print(f"Qualified QAs was saved to {qualified_path}, total: {qnum}")
    print(f"Unqualified QAs was saved to {unqualified_path}, total: {unum}")


# 自定义dump_jsonl 和 parse_filter_jsonl函数

# 将一张图片路径在其中处理
def dump_jsonl(i, qa, f):
    content = PROMPT.format(json.dumps(qa, ensure_ascii=False))
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # 序号即为编号
    f.write(json.dumps({
        "custom_id": f"request-{i}",
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
    err_num = 0
    print(f"Parsing and Filtering JSONL file: {input_path}")  
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # 提取 custom_id
            request_id = get_resp_id(data)
            # 形如ssource<EEdesign.pdf_0.png>,提取
            pattern = r"request-(\d+)"
            match = re.search(pattern, request_id)
            if match:
                id = match.group(1)
            else:
                print(f"Failed to parse request_id at line: {i}, request_id: {request_id}")
                continue
            # 提取返回的内容,经过字符处理
            try:
                content = get_resp_content(data)
                reason = content["reason"]
                score = content["score"]
                
                with open(result_path, 'a') as out_f:
                    out_f.write(json.dumps({
                        "id": id,
                        "reason": reason,
                        "score": score
                    }, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"Failed to parse result content as JSON at line: {i}")
                err_num += 1
            except KeyError as e:
                print(f"KeyError: {e} at line: {i}")
                err_num += 1
            tot += 1
            i += 1
    print(f"Total lines: {tot}, Errors: {err_num}")
    print(f"Filtered results of {input_path} saved to {result_path}")
    
    return

if __name__ == "__main__":
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Upload or download tasks.")
    parser.add_argument("--action", type=str, required=True, choices=["upload", "download"],help="Action to perform: 'upload' to upload tasks, 'download' to download results.")
    args = parser.parse_args()
    if args.action == "upload":
        # data
        qa_path = os.path.join(conf.DATASET_DIR, 'genQA.jsonl')
        qas = []
        # 每行是一个json对象，组织成一个列表
        with open(qa_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                qas.append({
                    "sub_type": data["sub_type"],
                    "question": data["question"],
                    "answer": data["answer"],
                    "analysis": data["analysis"]
                })
        upload_task(qas, dump_jsonl, TASK)
    elif args.action == "download":
        done = download_result(parse_filter_jsonl, TASK)
        if not done:
            print("download result failed")
            quit()
        # 进行质量分选
        qualified_path = os.path.join(conf.DATASET_DIR, f'qualifiedQA.jsonl')
        unqualified_path = os.path.join(conf.DATASET_DIR, f'unqualifiedQA.jsonl')
        qa_path = os.path.join(conf.DATASET_DIR, f'genQA.jsonl')
        score_path = os.path.join(conf.DATASET_DIR, f'{TASK}.jsonl')
        
        # 将合格的文件复制一份作为iter_0.jsonl
        iter_0_path = os.path.join(conf.DATASET_DIR, f'iter_0.jsonl')
        try :
            quality_sort(qa_path, score_path, qualified_path, unqualified_path)
            # 除了task之外的字段，从qualifyed_path中读取到iter_0_path
            with open(qualified_path, 'r') as qf, open(iter_0_path, 'w') as iter_f:
                for line in qf:
                    data = json.loads(line)
                    iter_f.write(json.dumps({
                        "source": data["source"],
                        "sub_type": data["sub_type"],
                        "question": data["question"],
                        "answer": data["answer"],
                        "analysis": data["analysis"],
                    }, ensure_ascii=False) + '\n')
            
        except FileNotFoundError as e:
            pass
