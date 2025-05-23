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

TASK="extract"



PROMPT = """
你是一个资深的出题人，即将出一份试卷。
任务描述: 请所给文档中选出若干你认为可以作为出题点的事实，作为考试范围以便后续出题。
限制条件: 从图片或者有关文字中选择你认为可以作为出题点的事实论点+相关数据，例子(如有)，从文档中#提取#出1-5条事实性信息，并且对该页面的信息价值进行评估
信息要求：
1.对于文字类信息：不要自己生成额外信息，严格遵照文档抽取,如果文档中对事实论点有相关例子或数据，请把例子和数据也提取出来。每个fact必须信息完整 
2.对于图片类信息：用语言详细描述图片的有关内容，不能只说“图片xx所示”，而应该用语言描述图片的内容。
3.对于表格类信息：不要说“信息如表xx所示”，而应该以xml格式给出表格的解析。
4.对于公式类信息：必须给出latex格式的公式代码。
打分格式：给出一个1-10的分数confidence，1表示这些信息价值很低，10表示信息价值很高。
打分要求：如果1.内容本身是与知识没有太大关联的，例如编者信息，教材名称等，或2.内容并不完整，例如说“表中xx”但是并没有表中具体信息，则被认为低价值。如果内容完整，且较好地符合了信息要求中的标准，则被认为高价值。
回复格式json格式严格遵循以下字段，例如：
{
   "facts":[
     "胶接接头的典型结构包括板件接头、圆柱形接头、锥形及盲孔接头和角接头。",
     "胶接接头的受力状况有拉伸、剪切、剥离与拉离等。", "胶接接头的抗剪切及抗拉伸能力强，而抗拉离及抗剥离能力弱。",
     "计算径向压力 $p$ 的公式：
    $$
    p > \frac{\sqrt{F^2 + \left( \frac{2T}{d} \right)^2}}{z d f}
    $$
    其中各符号意义如下：
    * $F$：轴向载荷
    * $T$：扭矩
    * $d$：配合面的公称直径
    * $z$：配合面的数量
    * $f$：摩擦系数",
     "<title="摩擦系数f值">
        <header>
            <column>压入法</column>
            <column>无润滑时</column>
            <column>有润滑时</column>
            <column>连接零件材料</column>
            <column>结合方式、润滑</column>
            <column>f</column>
        </header>
        <row>
            <pressFit>钢-铸钢</pressFit>
            <dry>0.11</dry>
            <lubricated>0.08</lubricated>
            <material>钢-钢</material>
            <method>油压扩孔，压力油为矿物油</method>
            <friction>0.125</friction>
        </row>
        <row>
            <pressFit>钢-结构钢</pressFit>
            <dry>0.10</dry>
            <lubricated>0.07</lubricated>
            <material>钢-钢</material>
            <method>油压扩孔，压力油为甘油，结合面清洁干净</method>
            <friction>0.18</friction>
        </row>
        <row>
            <pressFit>钢-优质结构钢</pressFit>
            <dry>0.11</dry>
            <lubricated>0.08</lubricated>
            <material>钢-钢</material>
            <method>在电炉中加热包容件至300°C</method>
            <friction>0.14</friction>
        </row>
        <row>
            <pressFit>钢-青铜</pressFit>
            <dry>0.15-0.20</dry>
            <lubricated>0.03-0.06</lubricated>
            <material>钢-青铜</material>
            <method>在电炉中加热包容件至300°C以后，结合面脱脂</method>
            <friction>0.2</friction>
        </row>
        <row>
            <pressFit>钢-铸铁</pressFit>
            <dry>0.12-0.15</dry>
            <lubricated>0.05-0.10</lubricated>
            <material>钢-铸铁</material>
            <method>油压扩孔，压力油为矿物油</method>
            <friction>0.1</friction>
        </row>
        <row>
            <pressFit>铸铁-铸铁</pressFit>
            <dry>0.15-0.25</dry>
            <lubricated>0.15-0.10</lubricated>
            <material>钢-铝铜合金</material>
            <method>无润滑</method>
            <friction>0.10-0.15</friction>
        </row>
        </table>
        "
        ]
  "confidence": 9
}
"""

# 自定义dump_jsonl 和 parse_filter_jsonl函数

# 将一张图片路径在其中处理
def dump_jsonl(i, img_path, f):
    content = []
    content.append({
        "type": "text",
        "text": PROMPT
    })
    with open(img_path, 'rb') as img_f:
        img_data = img_f.read()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        content.append({
            "type": "image_url",
            "image_url": {"url": img_base64}
        })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # 根据图片来源设置custom_id
    request_id = "source"
    img_name = os.path.basename(img_path)
    request_id += "<" + img_name + ">"
    f.write(json.dumps({
        "custom_id": request_id,
        "method": "POST",
        "url": "/v4/chat/completions",
        "body": {
            "model": "glm-4v-plus",
            "messages": messages,
            "response_format":{'type': 'json_object'},
            "max_tokens": 2048
        }
    }, ensure_ascii=False) + '\n')
    
    return

# 解析返回的jsonl文件到result_path,以append的方式
def parse_filter_jsonl(input_path, result_path):
    i = 1
    tot = 0
    low_conf = 0
    err_num = 0
    fact_srcs = []
    print(f"Parsing and Filtering JSONL file: {input_path}")  
    with open(input_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # 提取 custom_id
            request_id = get_resp_id(data)
            pattern = r"source<([^>]*)>"
            match = re.search(pattern, request_id)
            if match:
                source = match.group(1)
            else:
                print(f"Failed to parse request_id at line: {i}, request_id: {request_id}")
                continue
            # 提取返回的内容,经过字符处理
            try:
                content = get_resp_content(data)
                confidence = content["confidence"]
                facts = content["facts"]
                if confidence < conf.FACT_THRESHOLD:
                    low_conf += 1
                    continue
                
                with open(result_path, 'a') as result_f:
                    result_f.write(json.dumps({
                        "source": source,
                        "facts": facts
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
