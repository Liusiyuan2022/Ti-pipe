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

TASK="genQA"



PROMPT_R = """你是一个资深的考试出题人,面对以下的若干考点知识,请对其巧妙组织出若干是非判断和简答题,和计算题(如果存在相关知识点),并给出答案和解析。请注意以下几点：
难度要求：难度较高的Reasoning类型,虽然答案在考点知识中,但学生不能直接看出答案,而是需要进一步的推理才能做答。
答案解析：给出答案并且给出解析过程,说明怎么推理做出的答案。
问题个数：1-3个问题,数量取决于考点知识可细分的程度。
限制条件：a.问题内容需要聚焦于知识本身而非其参考的表格，图片等。不要问出"表xx中可以看出什么?"，"相关信息可以在表几找到?"这种具体指代某个表格或图片的问题。b.不要直接对某一章节的索引问问题，例如"第几章的内容是什么？","第四章和第三章是否独立"等。c.公式类信息需要给出latex格式的公式代码。
考点知识：{}
输出格式：
请返回json格式,例如:
{{
    "questions":[
        {{
            "task": "Reasoning",
            "sub_type": "statement_judgment", 
            "question": "在多盘摩擦离合器中，如果摩擦系数$f$增加，而其他条件不变，那么可传递的最大转矩$T_max$会成比例增加", 
            "answer": "正确", 
            "analysis": "根据公式$$T_{{max}} = fFR$$，可以看出最大转矩$T_{{max}}$与摩擦系数$f$成正比。因此，在其他条件（F和R）不变的情况下，摩擦系数$f$增加，最大转矩$T_{{max}}$也会成比例增加。"
        }},
        {{
            "task": "Reasoning",
            "sub_type": "calculation", 
            "question": "假设某多盘摩擦离合器的摩擦系数f为0.5，摩擦力的合力F为1000 N，作用在平均半径R为0.2 m的圆周上，计算该离合器可传递的最大转矩T_max。", 
            "answer": "100 N·m", 
            "analysis": "根据公式$$T_{{max}} = fFR$$，代入已知数值：$$T_{{max}} = 0.5 \\times 1000 \, \\text{{N}} \\times 0.2 \, \\text{{m}} = 100 \, \\text{{N·m}}$$。因此，该离合器可传递的最大转矩为100 N·m。"
        }},
        {{
            "task": "Reasoning", 
            "sub_type": "short_answer", 
            "question": "为什么薄壁轴瓦在装配时不再修刮轴瓦内圆表面？", "answer": "因为薄壁轴瓦的刚性小，其形状完全取决于轴承座的形状，因此需要精密加工以确保配合精度。", 
            "analysis": "根据考点知识，薄壁轴瓦由于刚性小，受力后形状完全取决于轴承座的形状。因此，为了保证轴瓦和轴承座的精确配合，装配时不再修刮轴瓦内圆表面，而是依赖精密加工来保证其形状和尺寸的准确性。"
        }}
    ]
}}
```"""

PROMPT_U =  """你是一个资深的考试出题人,面对一下的若干考点知识,请对其组织出若干选择,填空和简答题,并给出答案和解析。请注意以下几点：
任务类型：选择题multiple_choice、填空题fill_in_the_blank、简答题short_Answer
难度要求：难度较低的基础题Understanding,学生可以自信地直接从考点知识中找到答案。
答案解析：给出答案并且给出简单的解析过程。
问题个数：1-5个问题,数量取决于考点知识的信息价值和可细分的程度。
限制条件：a.问题内容需要聚焦于知识本身而非其参考的表格，图片等。不要问出"表xx中可以看出什么?"，"相关信息可以在表几找到?"这种具体指代某个表格或图片的问题。b.不要直接对某一章节的索引问问题，例如"第几章的内容是什么？","第四章和第三章是否独立"等。c.公式类信息需要给出latex格式的公式代码。
考点知识：{}
格式要求：
请返回json格式,例如:
{{
    "questions":[
        {{
            "task": "Understanding",
            "sub_type" : "multiple_choice",
            "question": "下列哪种设备中使用了弹簧来测量力的大小？ A. 温度计 B. 测力器 C. 电流表 D. 水表", 
            "answer": "B", 
            "analysis": "测力器和弹簧秤中使用弹簧来测量力的大小。其他选项所列设备不使用弹簧进行力的测量。"
        }},
        {{  
            "task": "Understanding",
            "sub_type": "fill_in_the_blank",
            "question": "弹簧可以用于__________和__________，如汽车、火车车厢下的减振弹簧。", "answer": "减振、缓冲",
            "analysis": "弹簧可以用于减振和缓冲，常见的应用如汽车、火车车厢下的减振弹簧，以及各种缓冲器用的弹簧。"
        }},
        {{
            "task": "Understanding", 
            "sub_type": "short_answer", 
            "question": "提高带速对带传动有哪些影响？", 
            "answer": "提高带速可以降低带传动的有效拉力，减少带的根数或V带的横截面积，从而减少带传动的尺寸；但也会提高V带的离心应力，增加单位时间内带的循环次数，不利于提高带传动的疲劳强度和寿命。", 
            "analysis": "根据考点知识，提高带速有利有弊，一方面可以降低有效拉力，减少尺寸；另一方面会增加离心应力，不利于疲劳强度和寿命。因此，答案应包括这两方面的内容。"
        }}
    ]
}}
```"""



# 自定义dump_jsonl 和 parse_filter_jsonl函数

# 将一张图片路径在其中处理
def dump_jsonl(i, fact_src, f):
    
    fact_chunk = fact_src["facts"]
    source = fact_src["source"]
    
    for prompt in [PROMPT_R, PROMPT_U]:
        content = prompt.format(fact_chunk)
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        task_class = "R" if prompt == PROMPT_R else "U"
                
        f.write(json.dumps({
            "custom_id": f"request-{i}-<{task_class}>-<{source}>",
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
                for item in content["questions"]:
                    with open(result_path, 'a') as out_f:
                        out_f.write(json.dumps({
                            "source": source,
                            "task": item["task"],
                            "sub_type": item["sub_type"],
                            "question": item["question"],
                            "answer": item["answer"],
                            "analysis": item["analysis"]
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
        # 每行是一个json对象，组织成一个列表
        with open(fact_path, 'r') as f:
            fact_srcs = [json.loads(line) for line in f]
        upload_task(fact_srcs, dump_jsonl, TASK)
    elif args.action == "download":
        download_result(parse_filter_jsonl, TASK)
