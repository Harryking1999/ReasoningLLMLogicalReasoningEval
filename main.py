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

    decoder = Decoder(args)

    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    
    total = 0
    correct_list = []
    with open(os.path.join(args.log_dir, args.api_log_file_name), 'w') as fout:
        for i, data in enumerate(dataloader):
            print('*************************')
            print("{}st data".format(i+1))
            print(data)

            # Prepare question template ...
            x, y = data
            x = "Q: " + x[0] + "\n" + "A:"
            y = y[0].strip()

            if args.method == "zero_shot":
                x = x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                x = x + " " + args.cot_trigger
            else:
                raise ValueError("method is not properly defined ...")

            
            # Answer prediction by generating text ...
            max_length = args.max_length
            z = None
            flag_while = 1
            while flag_while:
                try:
                    z = decoder.decode(args, x, max_length)
                    flag_while = 0
                    print("try success")
                except Exception as e:
                    print(e)
                    print("retry")

            # Answer extraction for zero-shot-cot ...
            if args.method == "zero_shot":
                z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                max_length = args.max_length_direct

                flag_while_2 = True
                pred = None
                while flag_while_2:
                    try:
                        pred = decoder.decode(args, z2, max_length)
                        flag_while_2 = False
                    except Exception as e:
                        print(e)
                        print("retry 2")
                # pred = decoder.decode(args, z2, max_length, i, 2)
                print(z2 + pred)
                fout.write(z2 + pred + '\n')
            else:
                pred = z
                print(x + pred)
                fout.write(x + pred + '\n')

            # Clensing of predicted answer ...
            pred = answer_cleansing(args, pred)

            # Choose the most frequent answer from the list ...
            print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')
            fout.write("pred : {}".format(pred) + '\n')
            fout.write("GT : " + y + '\n')
            fout.write('*************************' + '\n')

            # Checking answer ...
            correct = (np.array([pred]) == np.array([y])).sum().item()
            correct_list.append(correct)
            total += 1 #np.array([y]).size(0)
            
            # if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            #     break

            break
        # Calculate accuracy ...
        accuracy = (sum(correct_list) * 1.0 / total) * 100
        print("accuracy : {}".format(accuracy))
        fout.write("accuracy : {}".format(accuracy) + '\n')

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
        "--dataset", type=str, default="reclor", choices=["logiqa_mrc", "reclor"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--llm_service", type=str, default="aliyun", choices=["aliyun", "openai", "anthorpic", "local_vllm"], help="mandatory argument. the source of llm_service decides how to request LLMs."
    )

    parser.add_argument(
        "--api_key", type=str, default="", help="api key used for experiment"
    )

    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="qwen2.5-1.5b-instruct", choices=["qwen2.5-72b-instruct", "qwen2.5-1.5b-instruct"], help="model used for decoding. Note that 'gpt3' are the smallest models."
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

    parser.add_argument(
        "--max_length", type=int, default=128, help="maximum length of output tokens by model for reasoning extraction"
    )

    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )

    args = parser.parse_args()

    if args.dataset == "reclor":
        args.dataset_path = "./datasets/reclor/val.json"
        args.direct_answer_trigger = "\nTherefore, among A through D, the answer is "
    elif args.dataset == "logiqa_mrc":
        args.dataset_path = "./datasets/LogiQA2.0-main/logiqa/DATA/test.txt"
        args.direct_answer_trigger = "\nTherefore, among A through D, the answer is "
    else:
        raise ValueError("dataset is not properly defined ...")
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    args.direct_answer_trigger_for_fewshot = "The answer is"
    # if (args.dataset in ["multiarith", "gsm8k", "svamp", "singleeq", "addsub"]):
    #     args.direct_answer_trigger_for_fewshot = "The answer is "
    # elif (args.dataset in ["commonsensqa", "aqua"]):
    #     args.direct_answer_trigger_for_fewshot = "\nTherefore, among A through E, the answer is "
    # elif (args.dataset in ["strategyqa"]):
    #     args.direct_answer_trigger_for_fewshot = "\nTherefore, the answer (Yes or No) is"
    # elif (args.dataset in ["logiqa","reclor"]):
    #     args.direct_answer_trigger_for_fewshot = "\nTherefore, among A through D, the answer is "
    # elif (args.dataset in ['bigbench_date']):
    #     args.direct_answer_trigger_for_fewshot = "\nTherefore, among A through F, the answer is "

    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."

    return args

if __name__ == "__main__":
    main()