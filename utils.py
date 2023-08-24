import os
import json
import time
import torch
import random
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, T5ForConditionalGeneration

pad_tokens = {
    'gpt2': '<|endoftext|>',
    'gpt2-xl': '<|endoftext|>',
    'EleutherAI/gpt-j-6b': '<|endoftext|>',
    'togethercomputer/GPT-JT-6B-v1': '<|endoftext|>',
    't5-small': '<pad>',
    't5-large': '<pad>',
    'google/flan-t5-small': '<pad>',
    'google/flan-t5-large': '<pad>',
    'chavinlo/alpaca-native': '<unk>',
    'chavinlo/alpaca-13b': '<unk>'
}

pad_ids = {
    'gpt2': 50256,
    'gpt2-xl': 50256,
    'EleutherAI/gpt-j-6b': 50256,
    'togethercomputer/GPT-JT-6B-v1': 50256,
    't5-small': 0,
    't5-large': 0,
    'google/flan-t5-small': 0,
    'google/flan-t5-large': 0,
    'chavinlo/alpaca-native': 0,
    'chavinlo/alpaca-13b': 0
}

eos_ids = {
    'gpt2': 50256,
    'gpt2-xl': 50256,
    'EleutherAI/gpt-j-6b': 50256,
    'togethercomputer/GPT-JT-6B-v1': 50256,
    't5-small': 1,
    't5-large': 1,
    'google/flan-t5-small': 1,
    'google/flan-t5-large': 1,
    'chavinlo/alpaca-native': 2,
    'chavinlo/alpaca-13b': 2

}

prompt_types = {
    'open': '{input} The correct answer is',
    'closed': '{input} The possible answers are {choices}, but the correct answer is',
    'closed-adv': 'QUESTION: {input}\nOPTIONS: {choices}\nANSWER:'
}

def prepare_data_bigbench(data, prompt_type, number_of_shots, name):
    if name in ['vitaminc_fact_verification', 'rhyming']:
        prompt_types['open'] = '{input}\nThe correct answer is'
        prompt_types['closed'] = '{input}\nThe possible answers are {choices}, but the correct answer is'
    if name in ['common_morpheme', 'phrase_relatedness']:
        prompt_types['open'] = '{input}. The correct answer is'
        prompt_types['closed'] = '{input}. The possible answers are {choices}, but the correct answer is'
    if name in ['modified_arithmetic']:
        prompt_types['open'] = '{input} ?\nThe correct answer is'
        prompt_types['closed'] = '{input} ?\nThe possible answers are {choices}, but the correct answer is'
        prompt_types['closed-adv'] = 'QUESTION: {input} ?\nOPTIONS: {choices}\nANSWER:'
    if name in ['codenames']:
        prompt_types['closed'] = '{input} The correct answer is'
    inputs_targets = []
    for item in data:
        input = item['input'].rstrip(' ').replace(' ,', ',').replace(' .', '.')
        if name == 'tracking_shuffled_objects':
            input = input + '?'
        if name == 'codenames':
            options = {item['target']: 1}
            target = item['target']
        else:
            options = item['target_scores']
            target = '\"{}\"'.format(list(options.keys())[list(options.values()).index(1)].rstrip('.'))
        if prompt_type in ['closed']:
            input = prompt_types[prompt_type].format(
                input=input,
                choices=', '.join(['\"{}\"'.format(choice.rstrip('.')) for j, choice in enumerate(options.keys())])
            )
        elif prompt_type in ['closed-adv']:
            target = '({})'.format({0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k'}[list(options.values()).index(1)])
            input = prompt_types[prompt_type].format(
                input=input,
                choices=', '.join(['({}) \"{}\"'.format({0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k'}[j], choice) for j, choice in enumerate(options.keys())])
            )
        elif prompt_type in ['open']:
            input = prompt_types[prompt_type].format(input=input)
        if number_of_shots > 0:
            with open(f'prompts/{name}/{prompt_type}_{number_of_shots}.txt') as file:
                prefix = file.read()
            input = prefix + input
        longest_sequence = list(options.keys())[0]
        for option in list(options.keys())[1:]:
            if len(option) > len(longest_sequence):
                longest_sequence = option
        inputs_targets.append((input, target, '\t'.join(list(options.keys())), longest_sequence))
    return inputs_targets

def prepare_loader(inputs_targets, tokenizer, pad_token, batch_size=4, shuffle=True, device='cuda'):
    inputs = [item[0] for item in inputs_targets]
    tokenizer.add_special_tokens({'pad_token': pad_token})
    inputs = tokenizer(inputs, padding=True, return_tensors='pt').to(device)
    input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    return DataLoader([(input_ids[i], attention_mask[i], inputs_targets[i][0], inputs_targets[i][1], inputs_targets[i][2], inputs_targets[i][3]) for i in range(len(inputs_targets))], batch_size=batch_size, shuffle=shuffle)

def predict(input, model, tokenizer, max_new_tokens, pad_token_id, eos_token_id, top_k=20, temperature=0.9, do_sample=True, device='cuda'):
    input_ids = tokenizer(input, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new_tokens, top_k=top_k, pad_token_id=pad_token_id, eos_token_id=eos_token_id, do_sample=do_sample, temperature=temperature)
    return tokenizer.batch_decode(output)[0]

def batch_predict(input_ids, attention_mask, model, tokenizer, max_new_tokens, pad_token_id, eos_token_id, top_k=20, temperature=0.9, do_sample=True, device='cuda'):
    with torch.no_grad():
        output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, top_k=top_k, pad_token_id=pad_token_id, eos_token_id=eos_token_id, do_sample=do_sample, temperature=temperature)
    return tokenizer.batch_decode(output)

def log_and_save(results, config):
    with open('config_reference.txt', 'a') as file:
        file.write(('\t'.join(['{' + i + '}' for i in config.keys()]) + '\n').format(**config))
        file.close()

    with open('results/results-{}.json'.format(config['run_id']), 'w') as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

