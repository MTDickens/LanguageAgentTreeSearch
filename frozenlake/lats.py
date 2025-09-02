import copy
import numpy as np
from functools import partial
from webshop.models import gpt
import logging
import random

from frozenlake.env import FrozenLakeTextEnv


env = FrozenLakeTextEnv()


class Node:
    def __init__(self, state, question, env_state=None, parent=None):
        self.state = {'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.em = 0
        self.env_state = env_state

    def uct(self):
        if self.visits == 0 and self.value >= 0:
            return float('inf')
        elif self.visits == 0 and self.value < 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)


def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y, [], [])
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, cache_value=False):
    values = []
    local_value_cache = {}
    for y in ys:
        if y in local_value_cache:
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y, [])
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


def lats_search(args, task, idx, iterations=30, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)

    logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

    obs, info = env.reset()
    x = obs
    if to_print:
        print(idx, x)
    root = Node(state=None, question=x)
    root.env_state = env.clone_state()

    for i in range(iterations):
        node = select_node(root)
        if node is None:
            break
        if node.is_terminal and node.reward == 1:
            return node.state, node.value, node.reward, node.em
        expand_node(node, args, task, idx)
        while node.is_terminal:
            node = select_node(root)
            if node is None:
                break
            expand_node(node, args, task, idx)
        if node is None:
            break
        evaluate_node(node, args, task, idx)
        terminal_node = rollout(max(node.children, key=lambda child: child.value), args, task, idx, max_depth=30)
        if terminal_node.reward == 1:
            return terminal_node.state, terminal_node.value, terminal_node.reward, terminal_node.em
        backpropagate(terminal_node, terminal_node.reward)

    best_child = max(collect_all_nodes(root), key=lambda n: n.reward)
    return best_child.state, best_child.value, best_child.reward, best_child.em


def select_node(node):
    while node and node.children:
        terminal_children = [child for child in node.children if child.is_terminal]
        node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
        if node_with_reward_1:
            return node_with_reward_1
        node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
        while node and node.is_terminal and node.reward != 1:
            node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
    return node


def expand_node(node, args, task, idx):
    n = args.n_generate_sample
    if node.depth >= 30:
        node.is_terminal = True
        return
    new_nodes = generate_new_states(node, args, task, idx, n)
    node.children.extend(new_nodes)


def rollout(node, args, task, idx, max_depth=30):
    depth = node.depth
    n = 5
    while not node.is_terminal and depth < max_depth:
        new_states = []
        values = []
        while len(new_states) == 0:
            new_states = generate_new_states(node, args, task, idx, n)
        for state in new_states:
            if state.is_terminal:
                return state
        child_prompts = [generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
        while len(values) == 0:
            values = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
        max_value_index = values.index(max(values))
        node = new_states[max_value_index]
        depth += 1
    return node


def generate_new_states(node, args, task, idx, n):
    prompt = generate_prompt(node)
    sampled_actions = get_samples(task, prompt, "\nAction: ", n, prompt_sample=args.prompt_sample, stop="Observation")
    unique_states = {}
    for action in sampled_actions:
        local = copy.deepcopy(node.env_state)
        # we reuse env without full clone support; step directly
        new_state = node.state.copy()
        action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)
        if action_line:
            obs, r, done, info = env.step(action_line)
            new_state['action'] = action_line
            new_state['observation'] = obs
            new_node = Node(state=new_state, question=node.question, env_state=local, parent=node)
            if r > 0 or done:
                new_node.is_terminal = True
            new_node.reward = r
            new_node.value = r
            unique_states[action_line] = new_node
    return list(unique_states.values())


def evaluate_node(node, args, task, idx):
    child_prompts = [generate_prompt(child) for child in node.children if not child.is_terminal]
    votes = get_values(task, node.question, child_prompts, args.n_evaluate_sample)
    votes = votes + [0] * (len(node.children) - len(votes))
    max_vote = max(votes) if votes else 1
    if max_vote == 0:
        max_vote = 1
    for i, child in enumerate(node.children):
        child.value = votes[i] / max_vote
    return sum(votes) / len(votes) if votes else 0


def collect_all_nodes(node):
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_all_nodes(child))
    return nodes


def backpropagate(node, value):
    while node:
        node.visits += 1
        node.value = (node.value * (node.visits - 1) + value) / node.visits
        node = node.parent


def generate_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state['action']:
            new_segment.append(f"Action: {node.state['action']}")
        if node.state['observation'] and node.depth != 0:
            new_segment.append(f"Observation: {node.state['observation']}")
        trajectory.append('\n'.join(new_segment))
        node = node.parent
    return question + '\n\n' + '\n'.join(reversed(trajectory))

