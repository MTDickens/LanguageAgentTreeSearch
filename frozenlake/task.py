import re
from frozenlake.base import Task
from frozenlake.prompts import standard_prompt
from webshop.models import gpt
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


def get_token_length(text):
    return len(tokenizer.encode(text))


class FrozenLakeTask(Task):
    def __init__(self):
        super().__init__()
        self.steps = 30
        self.stops = ['\nObservation:\n', None]
        self.value_cache = {}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '', reflection_mapping_list=[]):
        return FrozenLakeTask.standard_prompt_wrap(x, y)

    @staticmethod
    def value_prompt_wrap(x: str, y: str, z: list = [], reflections: list = []) -> str:
        return FrozenLakeTask.standard_prompt_wrap(x, y)

    @staticmethod
    def value_outputs_unwrap(evaluate_prompt: str):
        return 0
