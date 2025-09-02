import argparse
import logging

from webshop.models import gpt_usage
from frozenlake.lats import lats_search
from frozenlake.task import FrozenLakeTask


def run(args):
    task = FrozenLakeTask()
    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    rewards = []
    for i in range(args.task_start_index, args.task_end_index):
        state, value, reward, em = lats_search(args, task, f'fixed_{i}', args.iterations, True)
        rewards.append(reward)
        print(i + 1, sum(rewards) / len(rewards))

    print('usage_so_far', gpt_usage(args.backend))


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'llama2', 'qwen/qwen-2.5-vl-7b-instruct'], default='gpt-3.5-turbo')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=1)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])
    args.add_argument('--n_generate_sample', type=int, default=1)
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=30)
    args.add_argument('--log', type=str)
    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)

