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
    import importlib

    # Extracting the class path and parameters from the JSON
    # Splitting the class path to import the module and get the class
    if '.' not in model_config['class']:
        raise ValueError('Class path should be module_name.class_name')

    module_name, class_name = model_config['class'].rsplit('.', 1)

    # Importing the module
    module = importlib.import_module(module_name)

    # Getting the class from the module
    Class = getattr(module, class_name)

    # Creating an instance of the class with the parameters
    return Class(**model_config['params'])


def load_datasets(file_path, sample_size=8):
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

        # Get a sample of the data
        if len(processed_data) > sample_size:
            processed_data = random.sample(processed_data, sample_size)

        return processed_data


def load_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def generate_experiment_id(model_name, technique_name, prompt):
    content = f'{model_name}_{technique_name}_{prompt}'
    return hashlib.md5(content.encode()).hexdigest()


def run_benchmark():
    # Load configuration
    llm_config = yaml.safe_load(open('../llm_config.yaml', 'r', encoding='utf8'))
    llms = {model_name: config2llm(model_config) for model_name, model_config in llm_config['models'].items()}

    techniques = llm_config['arLegalBench']['techniques']
    dataset_file = llm_config['arLegalBench']['dataset_file']
    datasets = load_datasets(dataset_file)

    # Shuffle the dataset
    random.shuffle(datasets)

    for model_name, model in tqdm(llms.items(), desc='Models', total=len(llms), leave=False):
        for technique_name, prompt_file in tqdm(techniques.items(), desc='Techniques', leave=False):
            prompt_template = load_prompt(prompt_file)
            experiment_id = generate_experiment_id(model_name, technique_name, prompt_template)
            experiment_dir = os.path.join('experiment_results', experiment_id)
            os.makedirs(experiment_dir, exist_ok=True)

            experiment_results = []
            metadata = {
                'model': model_name,
                'technique': technique_name,
                'prompt': prompt_template,
                'dataset': dataset_file,
                'F1 score': 0,
                'Precision': 0,
                'Recall': 0,
                'Accuracy': 0,
                'Balanced Accuracy': 0,
                'Number of contracts': len(datasets),
            }

            all_true_labels = []
            all_predictions = []

            for dataset_entry in tqdm(datasets, desc='Dataset Entries', leave=False):
                # Replace placeholders with actual context and Question
                prompt = prompt_template.replace('{contract}', dataset_entry['contract']).replace('{Question}', dataset_entry['Question'])

                # Log the full prompt for debugging
                logger.info(f'Full prompt: {prompt}')

                # Call the model's prediction method
                response = model.complete(prompt)

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

            metadata['F1 score'] = f1
            metadata['Precision'] = precision
            metadata['Recall'] = recall
            metadata['Accuracy'] = accuracy
            metadata['Balanced Accuracy'] = balanced_accuracy

            # Save results and metadata in JSON
            result_file_path = os.path.join(experiment_dir, f'{experiment_id}_result.json')
            metadata_path = os.path.join(experiment_dir, f'{experiment_id}_metadata.json')

            with open(result_file_path, 'w', encoding='utf-8') as file:
                json.dump(experiment_results, file, ensure_ascii=False, indent=4)

            with open(metadata_path, 'w', encoding='utf-8') as file:
                json.dump(metadata, file, ensure_ascii=False, indent=4)

            logger.info(f'Experiment {experiment_id} results saved.')


if __name__ == '__main__':
    run_benchmark()
