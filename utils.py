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
import dashscope

from http import HTTPStatus

dict_num2alphabet = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G' 
}

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
            # clean_content = content.replace('```json', '').replace('```', '').strip()
            # clean_content = clean_content.replace('\\n', '\n').replace('\\"', '"')
            # print(content)
            tmp_res = content
            return tmp_res


    ##elif ...
    else:
        ValueError("llm_service is not properly defined ...")

    
    
class Decoder():
    def __init__(self, args):
        print_now()
 
    def decode(self, args, input, max_length):
        response = decoder_for_gpt3(args, input, max_length)
        return response

def data_reader(args):

    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset in ["logiqa_mrc"]:
      with open(args.dataset_path) as f:
        line = f.readline()
        while line:
            tmp_dict = eval(line)
            choices = ""
            for i in range(0, len(tmp_dict['options'])):
                choices += "("
                choice += dict_num2alphabet[i]
                choices += ") "
                choices += tmp_dict[i]
                choices += " "
                
            questions.append(tmp_dict['text'] + "\n" + tmp_dict['question'] + "\n" + choices)
            answers.append(dict_num2alphabet[tmp_dict["answer"]])
            line = f.readline()
    elif args.dataset in ["reclor"]:
        with open(args.dataset_path) as f:
            data_ls = eval(f.read())
            for i in data_ls:
                choices = ""
                for j in range(0, len(i['answers'])):
                    choices += "("
                    choices += dict_num2alphabet[j]
                    choices += ") "
                    choices += i['answers'][j]
                    choices += " "
                questions.append(i['context'] + "\n" + i['question'] + '\n' + choices)
                answers.append(dict_num2alphabet[i["label"]])
    else:
        raise ValueError("dataset is not properly defined ...")

    # if args.dataset == "aqua":
    #   with open(args.dataset_path) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #       json_res = decoder.raw_decode(line)[0]
    #       choice = "(" + "(".join(json_res["options"])
    #       choice = choice.replace("(", " (").replace(")", ") ")
    #       choice = "Answer Choices:" + choice
    #       questions.append(json_res["question"].strip() + " " + choice)
    #       answers.append(json_res["correct"])
  
    # elif args.dataset == "gsm8k":
    #   with open(args.dataset_path) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #       json_res = decoder.raw_decode(line)[0]
    #       questions.append(json_res["question"].strip())
    #       answers.append(json_res["answer"].split("#### ")[-1])

    # # elif args.dataset == "commonsensqa":
    # elif args.dataset in ["commonsensqa", "logiqa", "reclor"]:
    #   with open(args.dataset_path) as f:
    #     lines = f.readlines()
    #     for line in lines:
    #       json_res = decoder.raw_decode(line)[0]
    #       choice = "Answer Choices:"
    #       for c in json_res["question"]["choices"]:
    #           choice += " ("
    #           choice += c["label"]
    #           choice += ") "
    #           choice += c["text"]
    #       questions.append(json_res["question"]["stem"].strip() + " " + choice)
    #       answers.append(json_res["answerKey"])

    # elif args.dataset in ("addsub", "multiarith", "singleeq"):
    #   with open(args.dataset_path) as f:
    #     json_data = json.load(f)
    #     for line in json_data:
    #       q = line["sQuestion"].strip()
    #       a = str(line["lSolutions"][0])
    #       if a[-2:] == ".0":
    #           a = a[:-2]
    #       questions.append(q)
    #       answers.append(a)
        
    # elif args.dataset == "strategyqa":
    #   with open(args.dataset_path) as f:
    #     json_data = json.load(f)["examples"]
    #     for line in json_data:
    #       q = line["input"].strip()
    #       a = int(line["target_scores"]["Yes"])
    #       if a == 1:
    #           a = "yes"
    #       else:
    #           a = "no"
    #       questions.append(q)
    #       answers.append(a)
        
    # elif args.dataset == "svamp":
    #   with open(args.dataset_path) as f:
    #     json_data = json.load(f)
    #     for line in json_data:
    #         q = line["Body"].strip() + " " + line["Question"].strip()
    #         a = str(line["Answer"])
    #         if a[-2:] == ".0":
    #             a = a[:-2]
    #         questions.append(q)
    #         answers.append(a)
            
    # elif args.dataset in ("bigbench_date", "object_tracking"):
    #   with open(args.dataset_path) as f:
    #     json_data = json.load(f)
    #     json_data = json_data["examples"]
    #     if args.dataset == "bigbench_date":
    #         choice_index = ['A','B','C','D','E','F']
    #     elif args.dataset in ("object_tracking"):
    #         choice_index = ['A','B','C']
    #     else:
    #         raise ValueError("dataset is not properly defined ...")
    #     for line in json_data:
    #       q = line["input"].strip()
    #       if args.dataset == "bigbench_date":
    #           choice = "Answer Choices:"
    #           # Randomly shuffle the answer choice dictionary because the original answer is always A ...
    #           choice_dic = shuffleDict(line["target_scores"])
    #       elif args.dataset == "object_tracking":
    #           choice = "\nWhich choice is true ? Answer Choices:"
    #           choice_dic = line["target_scores"]
    #       else:
    #           raise ValueError("dataset is not properly defined ...")
    #       for i, key_value in enumerate(choice_dic.items()):
    #           key, value = key_value
    #           choice += " ("
    #           choice += choice_index[i]
    #           choice += ") "
    #           choice += key
    #           if value == 1:
    #               a = choice_index[i]
    #               #a = key
    #       q = q + " " + choice
    #       questions.append(q)
    #       answers.append(a)            
          
    # elif args.dataset in ("coin_flip", "last_letters"):
    #   with open(args.dataset_path) as f:
    #     json_data = json.load(f)
    #     json_data = json_data["examples"]
    #     for line in json_data:
    #       q = line["question"]
    #       a = line["answer"]
    #       questions.append(q)
    #       answers.append(a)
        
    # else:
    #     raise ValueError("dataset is not properly defined ...")
    
    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)
    
    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))
    
    return questions, answers
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output
def setup_data_loader(args):

    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(worker_seed)
    
    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))
    
    dataset = MyDataset(args)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                  shuffle=True,
                  batch_size=args.minibatch_size,
                  drop_last=False,
                  num_workers=dataloader_num_workers,
                #   worker_init_fn=seed_worker,
                  generator=g,
                  pin_memory=True)

    return dataloader


def answer_cleansing(args, pred):
    print("pred_before : " + pred)
    if args.method in ("few_shot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False 
        pred = preds[-1]

    if args.dataset in ("reclor", "logiqa_mrc"):
        pred = re.findall(r'A|B|C|D|E', pred)
    else:
        raise ValueError("dataset is not properly defined in answer cleansing ...")


    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")

    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    
    print("pred_after : " + pred)
    
    return pred