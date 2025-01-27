import pickle
from api_models import get_api_keys, set_llm_and_embed
from constants import EmbeddingModelsMap, LLMsMap
from parameters import parse_args
from req2nodes import get_requirements_nodes
from indexing.utils import create_semantically_similar_nodes
import os

"""
"""

from requests.exceptions import HTTPError
import time
import random

# Function for exponential backoff retry mechanism
def exponential_backoff_request(request_func, max_retries=5, backoff_factor=1):
    """
    Retry function that uses exponential backoff.
    """
    retries = 0
    while retries < max_retries:
        try:
            return request_func()
        except HTTPError as e:
            if e.response.status_code == 429:
                wait_time = backoff_factor * (2 ** retries)  # Exponential backoff
                print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise e
    print("Max retries reached, giving up.")
    return None

# Add delay between requests to avoid hitting rate limits
def rate_limited_request(request_func, delay=5):
    """
    Wrap requests to apply a delay between them.
    """
    time.sleep(delay)  # Delay between requests to avoid hitting rate limits
    return request_func()

# Function to cache the results and avoid redundant requests
def cache_request(request_func, cache_file):
    """
    Cache the results to avoid repeated requests.
    """
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    else:
        print("Requesting data...")
        result = request_func()
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result


"""
"""

def generate_similar_nodes(args, dataset_name):
    base_dir = args.base_dir
    dataset_dir = f'{base_dir}/{dataset_name}'

    """
    """

    if not os.path.exists('similar_requirements'):
        os.makedirs('similar_requirements')
    
    """
    """

    # req_nodes = get_requirements_nodes(
    #     dataset_dir,
    #     all_req_files_path=args.all_req_filenames,
    # )
    # req_nodes = create_semantically_similar_nodes(
    #     req_nodes,
    #     args.num_similar_nodes
    # )

    # Cache the result of getting requirement nodes to reduce redundant requests
    req_nodes_cache_file = f"similar_requirements/{dataset_name}_req_nodes.pkl"
    req_nodes = cache_request(
        lambda: get_requirements_nodes(dataset_dir, all_req_files_path=args.all_req_filenames),
        req_nodes_cache_file
    )

    # Cache the result of creating semantically similar nodes
    similar_nodes_cache_file = f"similar_requirements/{dataset_name}_similar_nodes.pkl"
    req_nodes = cache_request(
        lambda: create_semantically_similar_nodes(req_nodes, args.num_similar_nodes),
        similar_nodes_cache_file
    )


    with open(f'similar_requirements/{dataset_name}.pkl', 'wb') as f:
        pickle.dump(req_nodes, f)


def main():
    configs = [
        ('iTrust', 'bge_large'),
        ('smos', 'bge_m3'),
        ('eANCI', 'bge_m3'),
        ('eTour', 'bge_large')
    ]
    args = parse_args()

    for i, (dataset_name, embed_model) in enumerate(configs):
        api_key = get_api_keys(llm_type=args.llm_type, idx=i)
        args.embed_model = embed_model
        llm_name = LLMsMap[args.llm]
        embed_model_name = EmbeddingModelsMap[embed_model]
        print(f"Using LLM: {llm_name}")
        print(f"Using Embedding Model: {embed_model_name}")

        set_llm_and_embed(
            llm_type=args.llm_type,
            llm_name=llm_name,
            embed_model_name=embed_model_name,
            api_key=api_key
        )

        generate_similar_nodes(args, dataset_name)

if __name__ == '__main__':
    main()