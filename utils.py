from statistics import mean
from torch.utils.data import Dataset
from collections import OrderedDict
import xml.etree.ElementTree as ET
import openai
import os
import multiprocessing
import json
import numpy as np
import random
import torch
import torchtext
import re
import random
import time
import datetime
import pandas as pd
from openai import OpenAI
import json

from http import HTTPStatus

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass

def decoder_for_gpt3(args, input, max_length):
    
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    # time.sleep(1)
    time.sleep(args.api_time_interval)
    
    if(args.llm_service == "aliyun"):
        engine = args.model
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key = args.api_key,
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        if(engine in ["deepseek-r1", "deepseek-r1-distill-qwen-1.5b", "deepseek-r1-distill-qwen-7b", "deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-qwen-32b"]): #是推理模型
            completion = client.chat.completions.create(
                model = engine,
                messages = [
                    {'role': 'user', 'content': input}
                ],
                stream=True
            )
            answer_content = ""
            is_answering = False
            for chunk in completion:
                if not chunk.choices:
                    print("\nUsage:")
                    print(chunk.usage)
                else:
                    delta = chunk.choices[0].delta
                    if(delta.content != "" and is_answering == False):
                        is_answering = True
                print(delta.content, end='', flush=True)
                answer_content += delta.content
            return answer_content

            
        else: #不是推理模型
            completion = client.chat.completions.create(
                model = engine,
                messages = [
                    {'role': 'user', 'content': input}
                ]
            )
            response = completion.model_dump_json()
            response = json.loads(response)
            content = response['choices'][0]['message']['content']
            clean_content = content.replace('```json', '').replace('```', '').strip()
            clean_content = clean_content.replace('\\n', '\n').replace('\\"', '"')
            tmp_res = eval(clean_content)
            return tmp_res


    ##elif ...
    else:
        ValueError("llm_service is not properly defined ...")

    
    
class Decoder():
    def __init__(self, args):
        print_now()
 
    def decode(self, args, input, max_length, i, k, api_key):
        response = decoder_for_gpt3(args, input, max_length, i, k, api_key)
        return response