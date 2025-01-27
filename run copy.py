# """
# Summarization of the Script:

# The provided Python script orchestrates the creation of various indexes and retrieval mechanisms for datasets. It integrates tools like vector stores, keyword tables, and knowledge graphs to process and query information from documents and codebases. These indices and query engines are used for tasks such as code retrieval, natural language queries, and requirement matching.

# This script is tailored for projects involving large-scale information retrieval, such as source code analysis or requirement traceability.

# Main Functions and Workflow
# Indexing and Retrieval Setup:

# get_vector_index: Creates a vector index for documents using a vector store for similarity-based retrieval.
# get_kg_links_index: Creates a knowledge graph (KG) links index by combining document embeddings with a graph database.
# get_kw_table_index: Creates a keyword table index for quick retrieval of content based on keywords.
# get_kg_index: Builds a standalone knowledge graph index.
# Code Graph Nodes Indexing:

# get_code_nodes: Generates graph nodes from source code and indexes these nodes, adding method call relationships if specified.
# Dataset Execution:

# execute_dataset: The central function that coordinates dataset-specific indexing, including:
# Parsing documents into nodes.
# Building vector, keyword, and knowledge graph indices.
# Storing and evaluating the results.
# Evaluation:

# Includes commented-out functionalities to evaluate correctness and retrieval performance by comparing system responses with expected results.
# Configuration Handling:

# main: Configures datasets and embeddings, sets up API keys, and triggers the dataset processing pipeline.
# Information Required by the Script
# Configurations and Parameters:

# Dataset names (e.g., iTrust, smos, etc.).
# Embedding model names (e.g., bge_large, bge_m3).
# LLM type, API keys, and other configurations.
# Directories and File Paths:

# Base directory of the dataset (args.base_dir).
# Chroma DB storage directory (args.chroma_db_dir).
# Path to the call graph file (args.call_graph_file).
# Path to solution files (args.solutions_file).
# Data Inputs:

# Documents or source code files for creating indices.
# Requirements files for evaluation tasks.
# Optional: Method call links for additional relationships in the code graph.
# External Libraries and Tools:

# BM25 retriever for document similarity retrieval.
# LlamaIndex for creating vector and graph-based indices.
# Parser utility for extracting document nodes.
# API Keys:

# Required for LLMs and embedding models, dynamically set per dataset.
# Optional Evaluation Metrics:

# Queries for retrieval evaluation.
# Ground truth data for response evaluation.
# Primary Use Cases
# Information Retrieval: Enabling search and querying on structured/unstructured data.
# Source Code Analysis: Indexing and querying codebases for software engineering tasks.
# Requirement Traceability: Linking requirements to related code/documents using retrieval models.
# Evaluation and Benchmarking: Testing the effectiveness of different retrieval methods.

# """


# """
# What We Are Giving:

# Dataset Names
# Embedding Model Names
# Base Directory Path
# Chroma DB Directory Path
# Call Graph File Path
# Solutions File Path
# API Keys
# Documents or Source Code Files
# Requirements Files (Optional)
# Configuration Parameters (e.g., LLM type, number of threads, etc.)

# What We Are Getting:

# Indexed Nodes
# Vector Index
# Knowledge Graph Links Index
# Keyword Table Index
# Knowledge Graph Index
# Code Graph Nodes
# Query Engines
# Correctness Evaluation Results
# Retrieval Evaluation Results
# Results Stored in Files (e.g., .pkl or .json)
# """


# import pickle
# from typing import List
# import json
# import os


# from bm25_retriever import BM25Retriever
# from llama_index.core.schema import TextNode, MetadataMode
# from llama_index.core.indices import VectorStoreIndex
# from evaluation import (
#     evaluate_response,
#     evaluate_retrieval, 
#     get_questions_from_nodes
# )
# from req2nodes import get_requirements_nodes
# from querying import custom_query_engine
# from api_models import get_api_keys, set_llm_and_embed
# from indexing.utils import get_parser
# from indexing.indices import (
#     create_keyword_table_index, 
#     create_knowledge_graph_index, 
#     get_graph_store
# )


# from querying import get_fusion_qe
# from code2graph import get_code_graph_nodes
# from indexing.code_index import add_method_call_links_to_docs, create_code_graph_nodes_index
# from indexing.utils import get_vector_storage_context
# from indexing.indices import create_vector_index
# from llama_index.core.schema import Document
# from constants import (
#     LLMsMap,
#     EmbeddingModelsMap,
# )
# from llama_index.core.indices import (
#     KnowledgeGraphIndex, 
# )

# from parameters import parse_args


# def get_vector_index(docs, indices_path, dataset_name):
#     print(f"Creating Vector Index...")
#     storage_context, _ = get_vector_storage_context(
#         chroma_db_path=f'{indices_path}/VI',
#         collection_name=f'{dataset_name}',
#     )

#     indexed_nodes, vector_index = create_vector_index(
#         nodes=docs, 
#         storage_context=storage_context, 
#         num_threads=8,
#         save_path=f"{indices_path}/{dataset_name}",
#         show_progress=True
#     )
#     print(f"Vector Index created.")
#     return indexed_nodes, vector_index


# def get_kg_links_index(
#         docs, 
#         graph_nodes, 
#         dataset_name,
#         host_ip='localhost',
#         include_embeddings=False,
#     ):
#     graph_storage_context = get_graph_store(
#         db_name=f'nl2code{dataset_name}',
#         host_ip=host_ip,
#     )
#     index = create_knowledge_graph_index(
#         docs, 
#         graph_nodes, 
#         graph_storage_context,
#         include_embeddings=include_embeddings
#     )
#     return index


# def get_kw_table_index(
#         docs, 
#         indices_path, 
#         dataset_name,
#         use_vector_store=False,
#     ) -> VectorStoreIndex:
#     print(f"Creating Keyword Table Index...")
#     storage_context, vector_store = get_vector_storage_context(
#         chroma_db_path=f'{indices_path}/KW',
#         collection_name=f'{dataset_name}',
#     )
#     if use_vector_store:
#         vector_index = VectorStoreIndex.from_vector_store(
#             vector_store=vector_store,
#         )
#     else:
#         _, vector_index = create_keyword_table_index(
#             docs=docs, 
#             storage_context=storage_context, 
#             num_keywords=20,
#             num_threads=8,
#             save_path=f'{indices_path}/{dataset_name}',
#             show_progress=True
#         )
    
#     print(f"Keyword Table Index created.")
#     return vector_index


# def get_kg_index(
#         docs: List[TextNode],
#         dataset_name,
#         host_ip='localhost',
#     ) -> KnowledgeGraphIndex:

#     graph_storage_index = get_graph_store(
#         db_name=f'nl2code{dataset_name}',
#         host_ip=host_ip,
#     )
    
#     docs = [Document(text=doc.get_content(MetadataMode.LLM)) for doc in docs]

#     kg_index = KnowledgeGraphIndex.from_documents(
#         documents=docs,
#         storage_context=graph_storage_index,
#     )
#     return kg_index


# def get_code_nodes(args, dataset_name):
#     os.makedirs('results', exist_ok=True)
#     args.dataset_name = dataset_name

#     base_dir = args.base_dir
#     dataset_name = args.dataset_name
#     dataset_dir = f'{base_dir}/{dataset_name}'
#     indices_path = f"{args.chroma_db_dir}/{dataset_name}"
#     os.makedirs(indices_path, exist_ok=True)

#     call_graph_file = f'{dataset_name.lower()}_{args.call_graph_file}'


#     graph_nodes = get_code_graph_nodes(
#         base_dir=dataset_dir,
#         all_code_files_path=args.all_code_files_path,
#         callgraph_file_name=call_graph_file,
#     )

#     print(f"Indexing code nodes...")
#     code_nodes = create_code_graph_nodes_index(
#         graph_nodes=graph_nodes,
#         add_method_calls=args.add_method_calls
#     )
#     print(f"Number of code nodes: {len(code_nodes)}")
#     print(f"Adding method call links to docs...")
#     return code_nodes


# def execute_dataset(args, dataset_name):

#     os.makedirs('results', exist_ok=True)
#     args.dataset_name = dataset_name

#     base_dir = args.base_dir
#     dataset_name = args.dataset_name
#     dataset_dir = f'{base_dir}/{dataset_name}'
#     indices_path = f"{args.chroma_db_dir}/{dataset_name}"
#     os.makedirs(indices_path, exist_ok=True)
    
#     solutions_file = args.solutions_file
#     solutions_file_path = f'{dataset_dir}/{dataset_name.lower()}_{solutions_file}'

#     code_nodes = get_code_nodes(args, dataset_name)
#     # add_method_call_links_to_docs(code_nodes, rel_type='parent')

#     parser = get_parser()
#     print(f"Getting nodes from documents...")
#     docs = parser.get_nodes_from_documents(code_nodes)
#     print(f"Number of nodes: {len(docs)}")
#     indexed_nodes, vector_index = get_vector_index(docs, indices_path, dataset_name)

#     with open(f'indices/{dataset_name}{"_mc" if args.add_method_calls else ""}.pkl', 'wb') as f:
#         pickle.dump(indexed_nodes, f)

#     # fusion_qe = get_fusion_qe(
#     #     retrievers=[bm25_retriever],
#     #     similarity_top_k=args.similarity_top_k,
#     #     num_queries=args.num_queries,
#     # )

#     # print(f"Creating KG Links Index...")
#     # kg_links_index = get_kg_links_index(
#     #     docs, 
#     #     graph_nodes,
#     #     dataset_name,
#     #     args.host_ip,
#     #     include_embeddings=args.include_embeddings
#     # )
#     # print(f"KG Links Index created.")

#     print(f"Creating KG Index...")
#     # kg_index = get_kg_index(docs, dataset_name, args.host_ip)
#     # kw_index = get_kw_table_index(docs, indices_path, dataset_name)
#     print(f"Indices created.")

#     # query_engines = {
#     #     'vector_index': vector_index.as_query_engine(response_mode='refine', similarity_top_k=10),
#         # 'fusion_qe_index': fusion_qe,
#         # 'kg_links_index': kg_links_index.as_query_engine(),
#         # 'kg_index': kg_index.as_query_engine(),
#         # 'kw_index': kw_index.as_query_engine(),
#         # 'vector_kg_links': custom_query_engine(
#         #     vector_index.as_retriever(),
#             # kg_links_index.as_retriever(),
#         # ),
#         # 'vector_kw_table': custom_query_engine(
#         #     vector_index.as_retriever(),
#         #     kw_index.as_retriever(),
#         # ),
#         # 'vector_kg': custom_query_engine(
#         #     vector_index.as_retriever(),
#         #     kg_index.as_retriever(),
#         # ),
#     # }


#     # req_nodes = get_requirements_nodes(
#     #     dataset_dir,
#     #     all_req_files_path=args.all_req_filenames,
#     # )

#     # correctness_results = evaluate_response(
#     #     req_nodes=req_nodes,
#     #     query_engines=query_engines,
#     #     solutions_file=solutions_file_path,
#     #     dataset_name=dataset_name
#     # )
    
#     # questions = get_questions_from_nodes(code_nodes)
#     # print(f"Number of questions: {len(questions)}")
#     # eval_results = evaluate_retrieval(
#     #     questions, query_engines, dataset_name
#     # )

#     # results = {
#     #     'correctness': correctness_results,
#     #     # 'evaluation': eval_results,
#     # }
#     # with open(f'results/{dataset_name}_results.json', 'w') as f:
#     #     json.dump(results, f, indent=4)


# def main():
#     configs = [
#         ('iTrust', 'bge_large'),
#         #('smos', 'bge_m3'),
#         #('eANCI', 'bge_m3'),
#         #('eTour', 'bge_large')
#     ]
#     args = parse_args()
#     dataset_name = 'iTrust'
#     embed_model = 'bge_large'
#     api_key = get_api_keys(llm_type=args.llm_type)


#     for i, (dataset_name, embed_model) in enumerate(configs):
#         api_key = get_api_keys(llm_type=args.llm_type, idx=i)
        
#         llm_name = LLMsMap[args.llm]
#         embed_model_name = EmbeddingModelsMap[embed_model]
#         print(f"Using LLM: {llm_name}")
#         print(f"Using Embedding Model: {embed_model_name}")

#         set_llm_and_embed(
#             llm_type=args.llm_type,
#             llm_name=llm_name,
#             embed_model_name=embed_model_name,
#             api_key=api_key
#         )

#         execute_dataset(args, dataset_name)
        
        


# if __name__ == '__main__':
#     main()