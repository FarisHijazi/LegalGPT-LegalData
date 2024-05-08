import os
import time
from multiprocessing.pool import ThreadPool

import pandas as pd
from tqdm.auto import tqdm


def remove_nul_chars_from_string(s):
    """Remove NUL characters from a single string."""
    return s.replace('\x00', '')


def remove_nul_chars_from_run_data(run_data):
    """Iterate over all fields of RunData to remove NUL characters."""
    for run in run_data:
        run.reference_question = remove_nul_chars_from_string(run.reference_question)
        run.reference_answer = remove_nul_chars_from_string(run.reference_answer)
        run.llm_answer = remove_nul_chars_from_string(run.llm_answer)
        run.llm_context = [remove_nul_chars_from_string(context) for context in run.llm_context]


def make_get_llama_response(query_engine):
    def get_llama_response(prompt):
        # print(prompt)
        response = query_engine.query(prompt)
        context = []
        for x in response.source_nodes:
            # Initialize context string with the text of the node
            node_context = x.text
            # Check if 'window' metadata exists and append it to the context
            if 'window' in x.metadata:
                node_context += '\n\nWindow Context:\n' + x.metadata['window']
            context.append(node_context)
        return {'llm_answer': response.response, 'llm_context_list': context}

    return get_llama_response


def chunked_iterable(iterable, size):
    """Yield successive size chunks from iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


import tonic_validate
from llama_index.core.query_engine import TransformQueryEngine
from tonic_validate import ValidateApi, ValidateScorer


def run_experiment(
    experiment_name: str,
    query_engine: TransformQueryEngine,
    scorer: ValidateScorer,
    benchmark: tonic_validate.classes.benchmark.Benchmark,
    validate_api: ValidateApi,
    tonic_validate_project_key: str = None,
    runs=5,
    experiment_index=None,
):
    # List to store results dictionaries
    def process_run(i):
        get_llama_response_func = make_get_llama_response(query_engine)
        assert len(benchmark.items) > 0, 'Benchmark is empty. Please provide a non-empty benchmark.'

        retries = 5
        for retry in range(retries):
            try:
                run = scorer.score(benchmark, get_llama_response_func, callback_parallelism=3, scoring_parallelism=3)  # TODO: len(benchmark)
                break
            except Exception as e:
                print(f'Error: in {experiment_name} Run {i+1}:', e)
                # print('llm_answer:', run.run_data.llm_answer)
                print(f'Retrying {retry+1}/{retries}...')
                time.sleep(1)
        else:
            # return None
            raise Exception(f'Failed to run {experiment_name} Run {i+1} after {retries} retries.')

        print(f'{experiment_name} Run {i+1} Overall Scores:', run.overall_scores)
        remove_nul_chars_from_run_data(run.run_data)

        # Add this run's results to the list
        if tonic_validate_project_key:
            validate_api.upload_run(tonic_validate_project_key, run=run, run_metadata={'approach': experiment_name, 'run_number': i + 1})
        else:
            print(f'Skipping upload for {experiment_name} Run {i+1}.')

        overall_scores = {'retrieval_precision': None, 'answer_similarity': None}
        overall_scores.update(run.overall_scores)
        return {'Run': i + 1, 'Experiment': experiment_name, 'OverallScores': overall_scores}

    # Use ThreadPool to process runs in parallel
    with ThreadPool(processes=runs) as pool:
        results_list = list(
            tqdm(
                pool.imap(process_run, range(runs)),
                f'running {runs}x "{experiment_name}"',
                total=runs,
                unit='runs',
                leave=False,
                position=experiment_index,
            )
        )
    # results_list = list(tqdm(map(process_run, range(runs)), f'running {runs}x "{experiment_name}"', total=runs, unit='runs', leave=False, position=experiment_index))

    # count Nones
    none_count = sum([1 for result in results_list if result is None])
    if none_count:
        print(f'WARNING: {none_count}/{len(results_list)} runs failed to generate results.')
    # filter Nones
    results_list = [result for result in results_list if result is not None]
    assert len(results_list) > 0, 'No results were generated. Please check the experiment setup and try again.'

    #
    # Create a DataFrame from the list of results dictionaries
    results_df = pd.DataFrame(results_list)

    # Return the DataFrame containing all the results
    return results_df


def filter_large_nodes(nodes, max_length=8000):
    """
    Filters out nodes with 'window' or 'text' length greater than max_length.
    Needed bcs sometimes the sentences are too long due to tables or refereneces in data.
    It creates one giga long non-sensical sentence. Before filtering please do analysis
    so that you dont throw out anything important.

    Args:
    - nodes (list): List of node objects.
    - max_length (int): Maximum allowed length for 'window' and 'text'.

    Returns:
    - list: Filtered list of nodes.
    """
    filtered_nodes = []
    for node in nodes:
        text_length = len(node.text)
        window_length = len(node.metadata.get('window', ''))

        if text_length <= max_length and window_length <= max_length:
            filtered_nodes.append(node)
    return filtered_nodes


import os

from llama_index.embeddings.openai import OpenAIEmbedding


def get_embed_model():
    # embed_model = TextEmbeddingsInference(
    #     model_name="BAAI/bge-large-en-v1.5",
    #     base_url="http://localhost:5004"
    # )
    embed_model = OpenAIEmbedding(model='text-embedding-3-large')
    return embed_model


def get_llm():
    from llama_index.llms.azure_openai import AzureOpenAI

    llm = AzureOpenAI(
        # model="gpt-4-32k",
        engine=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    )
    llm.complete('hi')
    # llm = MistralAI(api_key=os.getenv("MISTRALAI_API_KEY")
    # llm = OpenAI('gpt-4', api_key=os.getenv("OPENAI_API_KEY"))
    # llm = OpenAI('gpt-3.5-turbo', api_key=os.getenv('OPENAI_API_KEY'))
    # llm = Anthropic("claude-3-opus-20240229", api_key=os.getenv("ANTHROPIC_API_KEY"))

    return llm
