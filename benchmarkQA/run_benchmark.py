import argparse
import datetime
import json
import logging
import os
import pprint
import re
import time
from multiprocessing.pool import ThreadPool

import dotenv
import llama_index.core.instrumentation as instrument
import openai
import pandas as pd
import tonic_validate
import yaml
from dotenv import load_dotenv
from easydict import EasyDict
from tqdm.auto import tqdm
import random

import utils

dispatcher = instrument.get_dispatcher(__name__)

openai.log = 'warning'

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)


### SETUP
# --------------------------------------------------------------------------------------------------------------
# Load the config file (.env vars)
load_dotenv()

# Set the OpenAI API key for authentication.
openai.api_key = os.getenv('OPENAI_API_KEY')
# tonic_validate_api_key = os.getenv('TONIC_VALIDATE_API_KEY')
# tonic_validate_project_key = os.getenv('TONIC_VALIDATE_PROJECT_KEY')
# tonic_validate_benchmark_key = os.getenv("TONIC_VALIDATE_BENCHMARK_KEY")
cohere_api_key = os.getenv('COHERE_API_KEY')
# validate_api = ValidateApi(tonic_validate_api_key)

parser = argparse.ArgumentParser(description='Description of your program', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--collection_name', type=str, default='ai_arxiv', help='Vectore DB collection name')
parser.add_argument(
    '--benchmark', '-b', type=str, default='../data/experiments/NajizFAQ/Najiz_QA_with_context_v2.benchmark.json', help='Path to the benchmark file'
)
parser.add_argument('--interactive', '-i', action='store_true', help='Enable interactive mode')
parser.add_argument('--outpath', '-o', type=str, default='outputs/{exp_name}/{datetime_stamp}', help='Output path')
parser.add_argument('--config_path', '-c', type=str, default='resources/defaults_final.yaml', help='Path to the config file')
parser.add_argument('--runs_per_exp', '-r', type=int, default=10, help='Number of runs per experiment')
parser.add_argument('--benchmark_limit', '-l', type=int, default=None, help='Limit of benchmark questions')
parser.add_argument('--question_parallelism', '-p', type=int, default=5, help='Concurrency')
parser.add_argument('--model_parallelism', type=int, default=None, help='Concurrency')
parser.add_argument('--llm_cache', action='store_true', help='Use LLM cache')
args = parser.parse_args()

datetime_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# get base filename without extension
exp_name = os.path.splitext(os.path.basename(args.config_path))[0]
args.outpath = args.outpath.format(
    exp_name=exp_name,
    datetime_stamp=datetime_stamp,
)

# Format the current date and time as YYYY-MM-DD-HH-MM-SS

print('===== config =====')
pprint.pprint(vars(args))
print('=================')

###############################################
#
###############################################


dotenv.load_dotenv()


config = EasyDict(yaml.safe_load(open(args.config_path)))
config


benchmark_df = pd.DataFrame(json.loads(open(args.benchmark, 'r').read()))

# use args.limit
if args.benchmark_limit is not None:
    benchmark_df = benchmark_df.sample(n=args.benchmark_limit)

tonic_benchmark = tonic_validate.classes.benchmark.Benchmark(list(benchmark_df['questions']), list(benchmark_df['ground_truths']), 'ARAGOG')

llms = utils.get_llms(use_cache=args.llm_cache, benchmark=tonic_benchmark)
# del llms['claude-3-opus-20240229']

# benchmark = json.loads(open('eval_questions/benchmark.json', 'r').read())


judge_llms = [
    llms['gpt4-0125-preview'],
    # llms['claude-3-opus-20240229'],
]


config['llms'] = list(sorted(list(llms.keys())))

args.config = vars(config)


def run_question(llm, question, context=None):
    if context is None:
        return llm.complete(config.TEXT_QA_NORAG_TEMPLATE.format(query_str=question))
    else:
        return llm.complete(config.TEXT_QA_TEMPLATE.format(query_str=question, context_str=context))


def benchmark_llm(i_llm_name_llm, with_context=True):
    i, (llm_name, llm) = i_llm_name_llm

    def evaluate_question(question_i_llm_question_ground_truth_context):
        question_i, (question, ground_truth, context) = question_i_llm_question_ground_truth_context
        if not with_context:
            context = None

        # print('question', question)
        # print('ground_truth', ground_truth)
        llm_answer = run_question(llm, question, context)
        # print('llm_answer', llm_answer)

        retries = 5
        for retry in range(retries):
            try:
                # randomly pick each time
                judge_llm = random.choice(judge_llms)
                judge_response = judge_llm.complete(
                    config.ANSWER_SIMILARITY_TEMPLATE
                    +
                    # f"\n\nContext\n{context}" +
                    f'\n\nQuestion: "{question}"'
                    + f'\n\nReference answer: "{ground_truth}"'
                    + f'\n\nReference new answer:\n{llm_answer}'
                )
                # print('judge_response', judge_response)
                judge_score = int(re.search(r'([\d\.]+)', str(judge_response)).group(1))
                break
            except Exception as e:
                print('retry', retry, e)
                judge_score = 0
                time.sleep(8)

        return {
            'question': question,
            'ground_truth': ground_truth,
            'question_i': question_i,
            'llm': llm_name,
            'context': context,
            'llm_answer': str(llm_answer),
            'answer_similarity': judge_score,
        }

    llm_results = list(
        tqdm(
            ThreadPool(args.question_parallelism).imap(
                evaluate_question,
                enumerate(zip(benchmark_df['questions'], benchmark_df['ground_truths'], benchmark_df['context'])),
            ),
            desc=f'benchmarking {llm_name}',
            total=len(benchmark_df),
            position=i,
            leave=False,
        )
    )
    return llm_results


results_list = list(ThreadPool(args.model_parallelism).imap(lambda x: benchmark_llm(x, with_context=True), list(enumerate(llms.items()))))
results_list += list(ThreadPool(args.model_parallelism).imap(lambda x: benchmark_llm(x, with_context=False), list(enumerate(llms.items()))))

os.makedirs(args.outpath, exist_ok=True)

# flatten
for llm_name, results in zip(llms.keys(), results_list):
    for i, result in enumerate(results):
        result['llm'] = llm_name
        result['question_i'] = i
        # result['Experiment'] = llm_name + ' | ' + ("w Context" if result['context'] else "w/o Context")

results = [item for sublist in results_list for item in sublist]

# join where left key is question_i and right key is index and make sure to duplicate missing values
results_df = pd.DataFrame(results).join(benchmark_df, on='question_i', rsuffix='_r', how='outer')

results_df['Experiment'] = results_df['llm'] + ' | ' + results_df['context'].apply(lambda x: 'with Context' if x else 'w/o Context')
df_outpath = os.path.join(args.outpath, 'results.csv')
results_df.to_csv(df_outpath, index=False)
print('results be saved here', df_outpath)
results_df
# # Append the results of this experiment to the master DataFrame
# experiments_results_df = pd.concat(experiment_results_dfs, ignore_index=True)
# # Assuming experiments_results_df is your DataFrame
# experiments_results_df['RetrievalPrecision'] = experiments_results_df['OverallScores'].apply(lambda x: x.get('retrieval_precision', None))

# os.makedirs(args.outpath, exist_ok=True)
# with open(os.path.join(args.outpath, 'args.yaml'), 'w', encoding='utf8') as f:
#     yaml.dump(vars(args), f, default_flow_style=False, allow_unicode=True)
