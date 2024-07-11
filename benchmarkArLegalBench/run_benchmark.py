import json
import os
import hashlib
import logging
import random
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
import json
import os
import random
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check and create necessary directories
os.makedirs('experiment_results', exist_ok=True)


def config2llm(model_config):
    from llama_index.core import PromptTemplate
    import importlib

    # Extracting the class path and parameters from the JSON
    # Splitting the class path to import the module and get the class
    if '.' not in model_config['class']:
        raise ValueError('Class path should be module_name.class_name')

    from copy import deepcopy

    params = deepcopy(model_config['params'])
    if 'query_wrapper_prompt' in model_config['params']:
        params['query_wrapper_prompt'] = PromptTemplate(model_config['params']['query_wrapper_prompt'])

    def messages_to_prompt(messages):
        sep = model_config['params']['messages_to_prompt']['separator']
        footer = model_config['params']['messages_to_prompt']['footer']

        return sep.join([model_config['params']['messages_to_prompt'][x.role].format(query_str=x) for x in messages]) + footer

    if 'messages_to_prompt' in model_config['params']:
        params['messages_to_prompt'] = messages_to_prompt

    if 'completion_to_prompt' in model_config['params']:
        params['completion_to_prompt'] = lambda x: model_config['params']['query_wrapper_prompt'].format(query_str=x)

    module_name, class_name = model_config['class'].rsplit('.', 1)

    # Importing the module
    module = importlib.import_module(module_name)

    # Getting the class from the module
    Class = getattr(module, class_name)

    # Creating an instance of the class with the parameters
    return Class(**params)


def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise Exception(f'Dataset file does not exist: {file_path}')
    if os.path.getsize(file_path) == 0:
        raise Exception(f'Dataset file is empty: {file_path}')

    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise Exception(f'Error decoding JSON from file {file_path}: {str(e)}')

        processed_data = []
        for item in data:
            entry = {
                'Question': item.get('Question_translation', ''),
                'answer': item.get('answer_Translation', ''),
                'contract': item.get('contract_Translation', item.get('text_Translation', '')),
            }
            processed_data.append(entry)

        return processed_data


def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def generate_experiment_id(model_name, technique_name, prompt):
    """
    generate a filename based on the model, technique, and prompt
    let's say we have model_name=GPT4 and technique_name=technique1 and prompt="prompt1....."
    only the prompt is hashed and we want just part of the hash
    """
    # Hash the prompt
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:6]
    return f'{model_name}_{technique_name}_{prompt_hash}'


def run_benchmarks():
    # Load configuration
    llm_config = yaml.safe_load(open('../llm_config.yaml', 'r', encoding='utf8'))
    llms = {model_name: config2llm(model_config) for model_name, model_config in llm_config['models'].items()}

    import glob
    from pathlib import Path

    dataset_files = [
        './tasks/consumer_contract/test/',
        './tasks/contract_qa/test/',
        './tasks/privacy_policy_entailment/test/',
        './tasks/privacy_policy_qa/test/',
    ]
    dataset_files = [f for path in dataset_files for f in glob.glob(path + '*.json')]
    task_names = [Path(f).parent.parent.stem for f in dataset_files]
    datasets = {Path(f).parent.parent.stem: load_dataset(f) for f in dataset_files}

    for task_name in datasets:
        # logger.info(f'Loaded {len(dataset)} entries for task {task_name}')
        if llm_config['ArLegalBench'].get('sample_size') is not None:
            if llm_config['ArLegalBench'].get('sample_size') < len(datasets[task_name]):
                # logger.info(f'Sampling {llm_config["ArLegalBench"].get("sample_size")} entries for task {task_name}')
                datasets[task_name] = random.sample(datasets[task_name], llm_config['ArLegalBench'].get('sample_size'))

    # benchmarkArLegalBench/prompts/consumer_contract/Fewshots.txt
    task2techniques = {task_name: {Path(f).parent.stem: f for f in glob.glob(f'./prompts/{task_name}/*.txt')} for task_name in task_names}
    # list of datasetfiles, datasets, tasknames, techniques:

    # dataset_file = llm_config['ArLegalBench']['dataset_file']

    def process_run(task_name, dataset, llm_name, llm, techniques):
        # Shuffle the dataset
        random.shuffle(dataset)
        for technique_name, prompt_file in tqdm(techniques.items(), desc='Techniques', leave=False):
            prompt_template = load_prompt(prompt_file)
            experiment_id = generate_experiment_id(llm_name, technique_name, prompt_template)
            experiment_dir = os.path.join('experiment_results', experiment_id)

            experiment_results = []

            all_true_labels = []
            all_predictions = []

            for dataset_entry in tqdm(dataset, desc='Dataset Entries', leave=False):
                # Save results and metadata in JSON
                result_file_path = os.path.join(experiment_dir, f'{experiment_id}_result.json')
                metadata_path = os.path.join(experiment_dir, f'{experiment_id}_metadata.json')
                if os.path.exists(result_file_path) and os.path.exists(metadata_path):
                    logger.info(f'Experiment {experiment_id} already exists. Skipping...')
                    continue

                # Replace placeholders with actual context and Question
                prompt = prompt_template.replace('{contract}', dataset_entry['contract']).replace('{Question}', dataset_entry['Question'])

                # Log the full prompt for debugging
                # logger.info(f'Full prompt: {prompt}')

                # Call the model's prediction method
                response = llm.complete(prompt)

                # Extract the text from the response object
                if hasattr(response, 'text'):
                    prediction = response.text.strip()
                elif hasattr(response, 'choices') and len(response.choices) > 0:
                    prediction = response.choices[0].text.strip()
                else:
                    raise ValueError('Unexpected response format from model.complete')

                logger.info(f'Prediction: {prediction}')

                true_label = dataset_entry['answer']

                all_predictions.append(prediction)
                all_true_labels.append(true_label)

                experiment_results.append(
                    {
                        'Question': dataset_entry['Question'],
                        'contract': dataset_entry['contract'],
                        'true_label': true_label,
                        'predictions': prediction,
                    }
                )

            # Calculate aggregated metrics
            f1 = f1_score(all_true_labels, all_predictions, average='macro')
            precision = precision_score(all_true_labels, all_predictions, average='macro')
            recall = recall_score(all_true_labels, all_predictions, average='macro')
            accuracy = accuracy_score(all_true_labels, all_predictions)
            balanced_accuracy = balanced_accuracy_score(all_true_labels, all_predictions)

            metadata = {
                'model': llm_name,
                'technique': technique_name,
                'prompt': prompt_template,
                'dataset': task_name,
                'F1 score': f1,
                'Precision': precision,
                'Recall': recall,
                'Accuracy': accuracy,
                'Balanced Accuracy': balanced_accuracy,
                'Number of contracts': len(dataset),
            }
            os.makedirs(experiment_dir, exist_ok=True)

            with open(result_file_path, 'w', encoding='utf-8') as file:
                json.dump(experiment_results, file, ensure_ascii=False, indent=4)

            with open(metadata_path, 'w', encoding='utf-8') as file:
                json.dump(metadata, file, ensure_ascii=False, indent=4)

            logger.info(f'Experiment {experiment_id} results saved.')

    experiments = [
        (task_name, datasets[task_name], llm_name, llm, techniques)
        for task_name, techniques in task2techniques.items()
        for llm_name, llm in llms.items()
    ]
    from multiprocess.pool import ThreadPool

    # for task_name, dataset, llm_name, llm, techniques in tqdm(experiments, desc='Models', total=len(llms), leave=False):
    #     process_run(task_name, dataset, llm_name, llm, techniques)

    list(tqdm(ThreadPool().imap(lambda x: process_run(*x), experiments), desc='Models', total=len(experiments), leave=False))


if __name__ == '__main__':
    run_benchmarks()
