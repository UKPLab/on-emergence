import re
from utils import *

model_list = ['gpt2']

datasets = ['analytic_entailment/task.json']

p_ts = ['closed', 'closed-adv', 'open']
seeds = [2266, 105, 86379]
device = 'cuda'

for pretrained in model_list:
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side='left')
    if pretrained in ['gpt-2']:
        model = GPT2LMHeadModel.from_pretrained(pretrained).to(device)
    elif pretrained in ['t5-small', 'google/flan-t5-small', 't5-large', 'google/flan-t5-large']:
        model = T5ForConditionalGeneration.from_pretrained(pretrained).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float16).to(device)

    for dataset in datasets:
        if dataset == 'causal_judgment/task.json':
            n_o_s = [0, 2]
        elif dataset == 'strange_stories/task.json':
            n_o_s = [0, 4]
        else:
            n_o_s = [0, 5]
        for prompt_type in p_ts:
            for number_of_shots in n_o_s:
                for seed in seeds:
                    config = {
                        'run_id': time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                        'filename': dataset,
                        'number_of_data': None, ## check here
                        'model': pretrained,
                        'prompt_type': prompt_type,
                        'number_of_shots': number_of_shots,
                        'temperature': 0.01,
                        'max_new_tokens': 'adaptive', ## check here
                        'batch_size': 16, ## check here
                        'pad_token': pad_tokens[pretrained],
                        'pad_token_id': pad_ids[pretrained],
                        'eos_token_id': eos_ids[pretrained],
                        'seed': seed,
                        'device': torch.cuda.get_device_name(torch.cuda.current_device())
                    }

                    torch.cuda.manual_seed_all(config['seed'])
                    
                    name = config['filename'].replace('/task.json', '')
                    
                    try:
                        with open(os.path.join('data/', config['filename'])) as file:
                            data = json.loads(file.read())['examples']
                    except Exception as e:
                        with open(os.path.join('data/', config['filename'])) as file:
                            data = json.loads(file.read())
                    
                    inputs_targets = prepare_data_bigbench(data, config['prompt_type'], config['number_of_shots'], name)
                    config['number_of_data'] = len(inputs_targets)
                    loader = prepare_loader(inputs_targets, tokenizer, config['pad_token'], batch_size=config['batch_size'], shuffle=False, device=device)

                    results = defaultdict(list)
                    with tqdm(total=len(loader)) as t:
                        for input_ids, attention_mask, raw_inputs, targets, options, longest_sequence in loader:
                            max_new_tokens = config['max_new_tokens']
                            if config['max_new_tokens'] == 'adaptive':
                                max_new_tokens = tokenizer(list(longest_sequence), padding=True, return_tensors='pt').input_ids.shape[1] + 8
                            outputs = batch_predict(
                                input_ids,
                                attention_mask,
                                model,
                                tokenizer,
                                max_new_tokens,
                                pad_token_id=config['pad_token_id'],
                                eos_token_id=config['eos_token_id'],
                                temperature=config['temperature'],
                                device=device
                            )
                            for i in range(len(outputs)):
                                results['input'].append(raw_inputs[i])
                                results['target'].append(targets[i])
                                results['options'].append(options[i])
                                results['prediction'].append(outputs[i].replace(raw_inputs[i], '').lstrip('</s>'))
                            t.update(1)
                    log_and_save(results, config)
                    time.sleep(2)

    del model
    torch.cuda.empty_cache()
    time.sleep(2)