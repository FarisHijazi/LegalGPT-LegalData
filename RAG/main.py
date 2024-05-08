import datetime
import json
import logging
import os
from multiprocessing.pool import ThreadPool

import chromadb
import dotenv
import openai
import pandas as pd
import tonic_validate
import yaml
from dotenv import load_dotenv
from easydict import EasyDict
from llama_index.core import PromptTemplate, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.vector_stores.chroma import ChromaVectorStore
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tonic_validate import ValidateApi, ValidateScorer
from tonic_validate.metrics import AnswerSimilarityMetric, RetrievalPrecisionMetric
from tqdm.auto import tqdm

import utils
from utils import run_experiment

openai.log = 'warning'

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)


### SETUP --------------------------------------------------------------------------------------------------------------
# Load the config file (.env vars)
load_dotenv()

# Set the OpenAI API key for authentication.
openai.api_key = os.getenv('OPENAI_API_KEY')
tonic_validate_api_key = os.getenv('TONIC_VALIDATE_API_KEY')
tonic_validate_project_key = os.getenv('TONIC_VALIDATE_PROJECT_KEY')
# tonic_validate_benchmark_key = os.getenv("TONIC_VALIDATE_BENCHMARK_KEY")
cohere_api_key = os.getenv('COHERE_API_KEY')
validate_api = ValidateApi(tonic_validate_api_key)

# Service context
# llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)

# parser = argparse.ArgumentParser(description='Description of your program')
# parser.add_argument('--collection_name', dest='collection_name', type=str, default='ai_arxiv', help='Name of the collection')
# parser.add_argument('--benchmark_path', dest='benchmark_path', type=str, default='eval_questions/benchmark.json', help='Path to the benchmark file')
# parser.add_argument('--interactive', dest='interactive', action='store_true', help='Enable interactive mode')
# parser.add_argument('--outpath', dest='outpath', type=str, default='output/', help='Output path')
# parser.add_argument('--config_path', dest='config_path', type=str, default='resources/defaults_en.yaml', help='Path to the config file')
# parser.add_argument('--runs_per_exp', dest='runs_per_exp', type=int, default=10, help='Number of runs per experiment')
# parser.add_argument('--benchmark_limit', dest='benchmark_limit', type=int, default=20, help='Limit of benchmark questions')

# args = parser.parse_args()

# args_collection_name = args.collection_name
# args_benchmark_path = args.benchmark_path
# args_interactive = args.interactive
# args_outpath = args.outpath
# args_config_path = args.config_path
# args_runs_per_exp = args.runs_per_exp
# args_benchmark_limit = args.benchmark_limit
datetime_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

args_collection_name = 'ai_arxiv'
args_benchmark_path = 'eval_questions/benchmark.json'
args_interactive = False
args_config_path = 'resources/defaults_en.yaml'
args_runs_per_exp = 10
args_benchmark_limit = None


# get base filename without extension
exp_name = os.path.splitext(os.path.basename(args_config_path))[0]
args_outpath = f'output/{exp_name}/{datetime_stamp}'

# Format the current date and time as YYYY-MM-DD-HH-MM-SS

print('===== config =====')
print(f'{args_collection_name=}')
print(f'{args_benchmark_path=}')
print(f'{args_interactive=}')
print(f'{args_outpath=}')
print(f'{args_config_path=}')
print(f'{args_runs_per_exp=}')
print(f'{args_benchmark_limit=}')
print('=================')

###############################################
#
###############################################


dotenv.load_dotenv()

os.makedirs(args_outpath, exist_ok=True)

config = EasyDict(yaml.safe_load(open(args_config_path)))
config


llm = utils.get_llm()
embed_model = utils.get_embed_model()


Settings.embed_model = embed_model
Settings.llm = llm

chroma_client = chromadb.PersistentClient(path='./chroma_db')
# Traditional VDB
chroma_collection = chroma_client.get_collection(f'{args_collection_name}_full')
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
# Sentence window VDB
chroma_collection_sentence_window = chroma_client.get_collection(f'{args_collection_name}_sentence_window')
vector_store_sentence_window = ChromaVectorStore(chroma_collection=chroma_collection_sentence_window)
index_sentence_window = VectorStoreIndex.from_vector_store(vector_store=vector_store_sentence_window)
# Document summary VDB

# storage_context = StorageContext.from_defaults(persist_dir="Obelix")
storage_context = StorageContext.from_defaults(persist_dir=f'{args_collection_name}_doc_summary')
doc_summary_index = load_index_from_storage(llm=llm, storage_context=storage_context, embed_model=embed_model)

text_qa_template = PromptTemplate(config.TEXT_QA_TEMPLATE)


# Tonic Validate setup #TODO: move this into the run_experiments function, put all the tonic API stuff in one place
# benchmark = validate_api.get_benchmark(tonic_validate_benchmark_key)

# benchmark_json = json.loads(open('eval_questions/benchmark.json', 'r').read())
benchmark_json = json.loads(open(args_benchmark_path, 'r').read())

benchmark = tonic_validate.classes.benchmark.Benchmark(
    benchmark_json['questions'][:args_benchmark_limit], benchmark_json['ground_truths'][:args_benchmark_limit], 'ARAGOG'
)
assert len(benchmark.items) > 0, 'Benchmark is empty. Please provide a non-empty benchmark.'


RetrievalPrecisionMetric.prompt = config.RETRIEVAL_PRECISION_TEMPLATE
AnswerSimilarityMetric.prompt = config.ANSWER_SIMILARITY_TEMPLATE


scorer = ValidateScorer(metrics=[RetrievalPrecisionMetric(), AnswerSimilarityMetric()], model_evaluator='gpt-4')

### Define query engines -------------------------------------------------------------------------------------------------

# norag
index_norag = VectorStoreIndex(nodes=[])
query_engine_llm_norag = index_norag.as_query_engine(llm=llm, text_qa_template=PromptTemplate(config.TEXT_QA_NORAG_TEMPLATE))

# Naive RAG
query_engine_naive = index.as_query_engine(llm=llm, text_qa_template=text_qa_template, similarity_top_k=3, embed_model=embed_model)

# Cohere Rerank
cohere_rerank = CohereRerank(api_key=cohere_api_key, top_n=3)  # Ensure top_n matches k in naive RAG for comparability
query_engine_rerank = index.as_query_engine(
    similarity_top_k=10, text_qa_template=text_qa_template, node_postprocessors=[cohere_rerank], llm=llm, embed_model=embed_model
)

# HyDE
hyde = HyDEQueryTransform(include_original=True)
query_engine_hyde = TransformQueryEngine(query_engine_naive, hyde)

# HyDE + Cohere Rerank
query_engine_hyde_rerank = TransformQueryEngine(query_engine_rerank, hyde)

# Maximal Marginal Relevance (MMR)
query_engine_mmr = index.as_query_engine(vector_store_query_mode='mmr', similarity_top_k=3, embed_model=embed_model, llm=llm)

# Multi Query
vector_retriever = index.as_retriever(similarity_top_k=3)
retriever_multi_query = QueryFusionRetriever(
    [vector_retriever], similarity_top_k=3, num_queries=5, llm=llm, mode='reciprocal_rerank', use_async=False, verbose=True
)
query_engine_multi_query = RetrieverQueryEngine.from_args(retriever_multi_query, verbose=True, embed_model=embed_model, llm=llm)

# Multi Query + Cohere rerank + simple fusion
retriever_multi_query_rerank = QueryFusionRetriever(
    [vector_retriever], similarity_top_k=10, llm=llm, num_queries=5, mode='simple', use_async=False, verbose=True
)
query_engine_multi_query_rerank = RetrieverQueryEngine.from_args(
    retriever_multi_query_rerank, verbose=True, node_postprocessors=[cohere_rerank], embed_model=embed_model, llm=llm
)

## LLM Rerank
llm_rerank = LLMRerank(choice_batch_size=10, top_n=3)
query_engine_llm_rerank = index.as_query_engine(
    similarity_top_k=10, text_qa_template=text_qa_template, node_postprocessors=[llm_rerank], embed_model=embed_model, llm=llm
)
# HyDE + LLM Rerank
query_engine_hyde_llm_rerank = TransformQueryEngine(query_engine_llm_rerank, hyde)

# Sentence window retrieval
query_engine_sentence_window = index_sentence_window.as_query_engine(
    text_qa_template=text_qa_template, similarity_top_k=3, embed_model=embed_model, llm=llm
)

# Sentence window retrieval + Cohere rerank
query_engine_sentence_window_rerank = index_sentence_window.as_query_engine(
    similarity_top_k=10, text_qa_template=text_qa_template, node_postprocessors=[cohere_rerank], embed_model=embed_model, llm=llm
)

# Sentence window retrieval + LLM Rerank
query_engine_sentence_window_llm_rerank = index_sentence_window.as_query_engine(
    similarity_top_k=10, text_qa_template=text_qa_template, node_postprocessors=[llm_rerank], embed_model=embed_model, llm=llm
)

# Sentence window retrieval + HyDE
query_engine_sentence_window_hyde = TransformQueryEngine(query_engine_sentence_window, hyde)

# Sentence window retrieval + HyDE + Cohere Rerank
query_engine_sentence_window_hyde_rerank = TransformQueryEngine(query_engine_sentence_window_rerank, hyde)

# Sentence window retrieval + HyDE + LLM Rerank
query_engine_sentence_window_hyde_llm_rerank = TransformQueryEngine(query_engine_sentence_window_llm_rerank, hyde)

# Document summary index + Cohere Rerank
query_engine_doc_summary_rerank = doc_summary_index.as_query_engine(
    similarity_top_k=5, text_qa_template=text_qa_template, node_postprocessors=[cohere_rerank], llm=llm
)

# Document summary index + HyDE + Cohere Rerank
query_engine_hyde_doc_summary_rerank = TransformQueryEngine(query_engine_doc_summary_rerank, hyde)

## Run experiments -------------------------------------------------------------------------------------------------------
# Dictionary of experiments, now referencing the predefined query engine objects
experiments = {
    # TODO: add regular non-RAG LLM
    'Raw LLM': query_engine_llm_norag,
    # 'Classic VDB + Naive RAG': query_engine_naive,
    # "Classic VDB + Cohere Rerank": query_engine_rerank,
    # "Classic VDB + LLM Rerank": query_engine_llm_rerank,
    # "Classic VDB + HyDE": query_engine_hyde,
    # "Classic VDB + HyDE + Cohere Rerank": query_engine_hyde_rerank,
    # 'Classic VDB + HyDE + LLM Rerank': query_engine_hyde_llm_rerank,
    # "Classic VDB + Maximal Marginal Relevance (MMR)": query_engine_mmr,
    # "Classic VDB + Multi Query + Reciprocal": query_engine_multi_query,
    # "Classic VDB + Multi Query + Cohere rerank": query_engine_multi_query_rerank,
    'Sentence window retrieval': query_engine_sentence_window,
    # "Sentence window retrieval + Cohere rerank": query_engine_sentence_window_rerank,
    # "Sentence window retrieval + LLM Rerank": query_engine_sentence_window_llm_rerank,
    # "Sentence window retrieval + HyDE": query_engine_sentence_window_hyde,
    # "Sentence window retrieval + HyDE + Cohere Rerank": query_engine_sentence_window_hyde_rerank,
    # "Sentence window retrieval + HyDE + LLM Rerank": query_engine_sentence_window_hyde_llm_rerank,
    # "Document summary index + Cohere Rerank": query_engine_doc_summary_rerank,
    # "Document summary index + HyDE + Cohere Rerank": query_engine_hyde_doc_summary_rerank
}


if args_interactive:
    show_sources = True
    # qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    qa = query_engine_sentence_window_hyde_llm_rerank
    # args_interactive questions and answers
    while True:
        query = input('\nEnter a query: ')
        if query == 'exit':
            break
        # Get the answer from the chain
        response = qa.query(query)

        # print(response.source_nodes[0].get_content())

        # Print the result
        print('\n\n> Question:')
        print(query)
        print('\n> Answer:')
        print(str(response))

        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print('----------------------------------SOURCE DOCUMENTS---------------------------')
            for i, node in enumerate(response.source_nodes, 1):
                # print("\n> " + str(node) + ":")
                # if "page" in node.metadata:
                #     print("\tPage: " + str(node.metadata["page"]))
                # print node.page_content indented
                import textwrap

                print(str(i) + '\t' + textwrap.indent(node.get_content().replace('\n', '\n\t'), '\t'))
            print('----------------------------------SOURCE DOCUMENTS---------------------------')


# Initialize an empty DataFrame to collect results from all experiments
all_experiments_results_df = pd.DataFrame(columns=['Run', 'Experiment', 'OverallScores'])


def run_experiment_wrapper(args):
    i, (experiment_name, query_engine) = args
    return run_experiment(
        experiment_name,
        query_engine,
        scorer,
        benchmark,
        validate_api,
        tonic_validate_project_key=tonic_validate_project_key,
        runs=args_runs_per_exp,
        experiment_index=i,
    )  # Adjust the number of runs as needed


# Use ThreadPool.map to run the experiments in parallel
experiment_results_dfs = ThreadPool(processes=len(experiments)).map(
    run_experiment_wrapper, tqdm(list(enumerate(experiments.items())), desc='Running Experiments')
)
# experiment_results_dfs = list(map(run_experiment_wrapper, tqdm(list(enumerate(experiments.items())), desc="Running Experiments")))

# Append the results of this experiment to the master DataFrame
all_experiments_results_df = pd.concat(experiment_results_dfs, ignore_index=True)
# Assuming all_experiments_results_df is your DataFrame
all_experiments_results_df['RetrievalPrecision'] = all_experiments_results_df['OverallScores'].apply(lambda x: x.get('retrieval_precision', None))

print('done running experiments, saving intermediate results in all_experiments_results_df.csv')
all_experiments_results_df.to_csv(os.path.join(args_outpath, 'all_experiments_results.csv'), index=False)
print('saved to', os.path.join(args_outpath, 'all_experiments_results.csv'))

if len(experiments) < 3:
    print("Can't run statistics for less than 3 experiments")
    exit(0)

# Check for normality and homogeneity of variances
from scipy.stats import shapiro

# Test for normality for each group
for exp in experiments.keys():
    stat, p = shapiro(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'])
    print(f'{exp} - Normality test: Statistics={stat}, p={p}')
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


from scipy.stats import levene

# Test for equal variances
stat, p = levene(*(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'] for exp in experiments.keys()))
print(f'Levene’s test: Statistics={stat}, p={p}')
if p > alpha:
    print('Equal variances across groups (fail to reject H0)')
else:
    print('Unequal variances across groups (reject H0)')

import scipy

# ANOVA
f_value, p_value = scipy.stats.f_oneway(
    *(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'] for exp in experiments.keys())
)
print(f'ANOVA F-Value: {f_value}, P-Value: {p_value}')

# If ANOVA assumptions are not met, use Kruskal-Wallis
# h_stat, p_value_kw = stats.kruskal(*(all_experiments_results_df[all_experiments_results_df['Experiment'] == exp]['RetrievalPrecision'] for exp in experiments.keys()))
# print(f"Kruskal-Wallis H-Stat: {h_stat}, P-Value: {p_value_kw}")

# Extract just the relevant columns for easier handling
data = all_experiments_results_df[['Experiment', 'RetrievalPrecision']]

# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(endog=data['RetrievalPrecision'], groups=data['Experiment'], alpha=0.05)
print(tukey_result)

# You can also plot the results to visually inspect the differences
import matplotlib.pyplot as plt

tukey_result.plot_simultaneous()
plt.show()
# save output to path
plt.savefig(os.path.join(args_outpath, 'stats.png'))
