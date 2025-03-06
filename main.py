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
    decoder = Decoder(args)
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()
    
    if args.method == "few_shot":
        demo = create_demo_text(args, cot_flag=False)
    elif args.method in ["few_shot_cot", "few_shot_cot_chaos"]:
        demo = create_demo_text(args, cot_flag=True)
    else:
        pass
    
    total = 0
    correct_list = []
    with open(os.path.join(args.log_dir, args.api_log_file_name), 'w') as fout:        
        for i, data in enumerate(dataloader):
            print('*************************')
            print("{}st data".format(i+1))
            fout.write('*************************' + '\n')
            fout.write("{}st data".format(i+1) + '\n')
                    
            # Prepare question template ...
            x, y = data
            x = "Q: " + x[0] + "\n" + "A:"
            y = y[0].strip()
            
            if args.method == "zero_shot":
                x = x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                x = x + " " + args.cot_trigger
            elif args.method == "few_shot":
                x = demo + x
            elif args.method in ["few_shot_cot", "few_shot_cot_chaos"]:
                x = demo + x
            else:
                raise ValueError("method is not properly defined ...")
            
            # Answer prediction by generating text ...
            max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
            z = None
            flag_while = 1
            while flag_while:
                try:
                    z = decoder.decode(args, x, max_length, i, 1, args.api_key)
                    flag_while = 0
                    print("fzz:try success")
                except Exception as e:
                    print(e)
                    print("fzz:retry")

            # print(z)
            # return

            # while flag_while:
            #     try:
            #         z = decoder.decode(args, x, max_length, i, 1, args.api_key)
            #         flag_while = 0
            #         print("fzz:try success")
            #     except Exception as e:
            #         print(e)
            #         print("fzz:retry")

            # Answer extraction for zero-shot-cot ...
            if args.method == "zero_shot_cot":
                z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                max_length = args.max_length_direct

                flag_while_2 = True
                pred = None
                while flag_while_2:
                    try:
                        pred = decoder.decode(args, z2, max_length, i, 2, args.api_key)
                        flag_while_2 = False
                    except Exception as e:
                        print(e)
                        print("fzz:retry 2")
                # pred = decoder.decode(args, z2, max_length, i, 2)
                print(z2 + pred)
                fout.write(z2 + pred + '\n')
            elif args.method in ['few_shot_cot', "few_shot_cot_chaos"]:
                z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                max_length = args.max_length_direct

                flag_while_2 = True
                pred = None
                while flag_while_2:
                    try:
                        pred = decoder.decode(args, z2, max_length, i, 2, args.api_key)
                        flag_while_2 = False
                    except Exception as e:
                        print(e)
                        print("fzz:retry 2")
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
            
            if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
                break
                #raise ValueError("Stop !!")
    
        # Calculate accuracy ...
        accuracy = (sum(correct_list) * 1.0 / total) * 100
        print("accuracy : {}".format(accuracy))
        fout.write("accuracy : {}".format(accuracy) + '\n')
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters", "logiqa", "reclor"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--api_key", type=str, default="api_key1", choices=["api_key1", "api_key2", "api_key3"], help="api key used for experiment"
    )
    
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--model", type=str, default="gpt3", choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "gpt3.5", 'chatglm3-6b', 'chatglm-6b-v2', 'baichuan2-7b-chat-v1', 'baichuan-7b-v1', 'qwen1.5-110b-chat', 'qwen1.5-72b-chat', 'qwen1.5-32b-chat', 'qwen1.5-14b-chat', 'qwen1.5-7b-chat', 'qwen1.5-1.8b-chat', 'qwen1.5-0.5b-chat'], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    
    parser.add_argument(
        "--method", type=str, default="zero_shot_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_cot_chaos"], help="method"
    )
    parser.add_argument(
        "--cot_trigger_no", type=int, default=1, help="A trigger sentence that elicits a model to execute chain of thought"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=128, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is "
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is "
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is "
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is "
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is "
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is "
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is "
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is "
    elif args.dataset == "logiqa":
        args.dataset_path = "./dataset/LogiQA/task.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through D, the answer is "
    elif args.dataset == "reclor":
        args.dataset_path = "./dataset/ReClor/task.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through D, the answer is "
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    
    if (args.dataset in ["multiarith", "gsm8k", "svamp", "singleeq", "addsub"]):
        args.direct_answer_trigger_for_fewshot = "The answer is "
    elif (args.dataset in ["commonsensqa", "aqua"]):
        args.direct_answer_trigger_for_fewshot = "\nTherefore, among A through E, the answer is "
    elif (args.dataset in ["strategyqa"]):
        args.direct_answer_trigger_for_fewshot = "\nTherefore, the answer (Yes or No) is"
    elif (args.dataset in ["logiqa","reclor"]):
        args.direct_answer_trigger_for_fewshot = "\nTherefore, among A through D, the answer is "
    elif (args.dataset in ['bigbench_date']):
        args.direct_answer_trigger_for_fewshot = "\nTherefore, among A through F, the answer is "
    
    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    elif args.cot_trigger_no == 15:
        args.cot_trigger = "Please skip as much as possible."
    elif args.cot_trigger_no == 16:
        args.cot_trigger = "Please don't think step by step."
    elif args.cot_trigger_no == 17:
        args.cot_trigger = "Please think as much as possible."
    elif args.cot_trigger_no == 18:
        args.cot_trigger = "Please speak as much as possible."
    elif args.cot_trigger_no == 19:
        args.cot_trigger = "Please don't think step by step."
    elif args.cot_trigger_no == 20:
        args.cot_trigger = "Let's skip as much as possible."
    elif args.cot_trigger_no == 21:
        args.cot_trigger = "Let's don't think step by step."
    elif args.cot_trigger_no == 22:
        args.cot_trigger = "Let's think as much as possible."
    elif args.cot_trigger_no == 23:
        args.cot_trigger = "Let's speak as much as possible."
    elif args.cot_trigger_no == 24:
        args.cot_trigger = "Let's quickly conclude the answer without showing step-by-step reasoning."
    elif args.cot_trigger_no == 25:
        args.cot_trigger = "We have only a few seconds to answer. What's our immediate response?"
    elif args.cot_trigger_no == 26:
        args.cot_trigger = "If we had to guess the answer without thinking too deeply, what would it be?"
    elif args.cot_trigger_no == 27:
        args.cot_trigger = "What's the final answer or outcome? Let's avoid detailing the process."
    elif args.cot_trigger_no == 28:
        args.cot_trigger = "Imagine we're making a quick decision based on limited information. What's our answer to this question?"
    elif args.cot_trigger_no == 29:
        args.cot_trigger = "Based on our first impression, what seems to be the right answer?"
    elif args.cot_trigger_no == 30:
        args.cot_trigger = "Think abstractly – what's an unconventional answer to this problem?"
    elif args.cot_trigger_no == 31:
        args.cot_trigger = "What metaphor or analogy comes to mind when we think about this problem's solution?"
    elif args.cot_trigger_no == 32:
        args.cot_trigger = "Rapidly evaluate and use the most effective reasoning shortcut to answer the question."
    elif args.cot_trigger_no == 33:
        args.cot_trigger = "Think outside the box and quickly identify an innovative shortcut to solve this problem."
    elif args.cot_trigger_no == 34:
        args.cot_trigger = "Let's quickly conclude the answer with shortcut reasoning."
    else:
        raise ValueError("cot_trigger_no is not properly defined ...")
    
    return args

if __name__ == "__main__":
    main()