import argparse
import logging
import torch
import random
import time
import os
from utils import *

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    # print("OPENAI_API_KEY:")
    # print(os.getenv("OPENAI_API_KEY"))
    
    # Initialize decoder class (load model and tokenizer) ...

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Eval")
    parser.add_argument(
        "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="logiqa", choices=["logiqa", "reclor"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--llm_service", type=str, default="aliyun", choices=["aliyun", "openai", "anthorpic", "local"], help="mandatory argument. the source of llm_service decides how to request LLMs."
    )

    parser.add_argument(
        "--api_key", type=str, default="api_key1", choices=["api_key1", "api_key2", "api_key3"], help="api key used for experiment"
    )

    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="qwen2.5-72b-instruct", choices=["qwen2.5-72b-instruct"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot", choices=["zero_shot"], help="method"
    )

    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="prompt_no"
    )

    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )

if __name__ == "__main__":
    main()