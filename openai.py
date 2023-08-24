import openai
from utils import *

openai.organization = ''
openai.api_key = ''

datasets = ['analytic_entailment/task.json']

p_ts = ['closed', 'closed-adv', 'open']
tokenizer = AutoTokenizer.from_pretrained('gpt2', padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

for dataset in datasets:
    if dataset == 'causal_judgment/task.json':
        n_o_s = [0, 2]
    elif dataset == 'strange_stories/task.json':
        n_o_s = [0, 4]
    else:
        n_o_s = [0, 5]
    for prompt_type in p_ts:
        for number_of_shots in n_o_s:
            config = {
                'run_id': time.strftime('%Y%m%d_%H%M%S', time.localtime()),
                'filename': dataset,
                'number_of_data': None, ## check here
                'model': 'davinci',
                'prompt_type': prompt_type,
                'number_of_shots': number_of_shots,
                'temperature': 0,
                'max_new_tokens': 'adaptive', ## check here
                'batch_size': None, ## check here
                'pad_token': None,
                'pad_token_id': None,
                'eos_token_id': None,
                'seed': None,
                'device': None
            }

            name = config['filename'].replace('/task.json', '')

            try:
                with open(os.path.join('data/', config['filename'])) as file:
                    data = json.loads(file.read())['examples']
            except Exception as e:
                with open(os.path.join('data/', config['filename'])) as file:
                    data = json.loads(file.read())
            
            inputs_targets = prepare_data_bigbench(data, config['prompt_type'], config['number_of_shots'], name)
            config['number_of_data'] = len(inputs_targets)

            results = defaultdict(list)
            with tqdm(total=config['number_of_data']) as t:
                for j in range(config['number_of_data']):
                    input, target, options, longest_sequence = inputs_targets[j]
                    max_new_tokens = config['max_new_tokens']
                    if config['max_new_tokens'] == 'adaptive':
                        max_new_tokens = tokenizer(list(longest_sequence), padding=True, return_tensors='pt').input_ids.shape[1] + 8
                    try:
                        response = openai.Completion.create(
                            prompt=input,
                            model=config['model'],
                            max_tokens=max_new_tokens,
                            temperature=config['temperature']
                        )
                        results['prediction'].append(response.choices[0].message['content'].replace(input, ''))
                        results['input'].append(input)
                        results['target'].append(target)
                        results['options'].append(options)
                    except Exception as e:
                        time.sleep(6)
                        response = openai.Completion.create(
                            prompt=input,
                            model=config['model'],
                            max_tokens=max_new_tokens,
                            temperature=config['temperature']
                        )
                        results['prediction'].append(response.choices[0].message['content'].replace(input, ''))
                        results['input'].append(input)
                        results['target'].append(target)
                        results['options'].append(options)
                    t.update(1)
            log_and_save(results, config)
