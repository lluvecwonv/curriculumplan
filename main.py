import argparse
from db.db_search import DatabaseHandler
from dataset.dataset import goalDataset, goalDatasetjson

from  aov import  build_prereq_postreq, visualize_and_sort_department_graphs
import os
import json

from open_ai import initialize_openai_client, query_expansion, selected_edge_by_llm
from dataset.dataset import class_depart_Dataset, goalDatasetjson
from dense_retriver import DenseRetriever, classRetriever
from reranker import ReRanker

import json
from utils import update_json_with_index , read_json
import random
from collections import OrderedDict
import numpy as np
import logging

# Save logs to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='result.log',  # Save logs to a file
    filemode='w'  # 'w': overwrite, 'a': append
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Fix seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

set_seed()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--query_prompt_path", type=str, required=True)
    parser.add_argument("--query_path", type=str, default="/root/Ai_mentor/curriclum/data/val_expand_random.json")
    parser.add_argument("--department_path", type=str, default="/root/Ai_mentor/curriclum/data/depart_info.json")
    parser.add_argument("--save_path", type=str, default="/root/Ai_mentor/curriclum/result")
    parser.add_argument('--prerequisite_path', type=str, default="/root/Ai_mentor/curriclum/prerequisities/prerequisities/result_hyun_j")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--goal_index_path', type=str, default=None)
    parser.add_argument('--class_index_path', type=str, default=None)
    parser.add_argument('--db_config_path', type=str, default="/root/Ai_mentor/curriclum/data/db.json")
    parser.add_argument('--selected_class_path', type=str, default="/root/Ai_mentor/curriclum/result/selected_class.json")

    # Hyperparameters
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--reranker", action="store_true", default=False)
    parser.add_argument('--query_exp', action="store_true", default=False)
    parser.add_argument('--department_y', action="store_true", default=False)
    parser.add_argument('--required_dept_count', type=int, default=30)
    return parser.parse_args()


def recursive_top1_selection(client, db_handler, query_embedding, query, selected_dept_list,  
                             class_retriever, graph_path, gt_department, 
                             already_selected_classes=None, graph_visited_ids=None, depth=0):

    # Initialize already_selected_classes if None or invalid
    if already_selected_classes is None or not isinstance(already_selected_classes, list):
        already_selected_classes = []
        
    if graph_visited_ids is None or not isinstance(graph_visited_ids, set):
        graph_visited_ids = set()
    
    # Track visited nodes
    visited_ids = {c.get("class_id") for c in already_selected_classes}
    graph_visited_ids.update(visited_ids)

    logger.info(f"Current number of visited nodes: {len(graph_visited_ids)}")

    # Retrieve candidate courses
    candidate_dict = class_retriever.retrieve_by_department(query_embedding, selected_dept_list, top_k=1, visited_class_ids=visited_ids)
    logger.info(f"Candidate dict retrieved: {candidate_dict}")

    candidate_list = []
    for dept_name, dept_results in candidate_dict.items():
        if isinstance(dept_results, list) and dept_results:
            candidate_list.extend(dept_results)

    # If no candidates found
    if not candidate_list:
        logger.info("No candidates found. Ending search.")
        G, visited_ids  = build_prereq_postreq(class_retriever, already_selected_classes, db_handler, query_embedding, logger=logger, existing_visited_ids=graph_visited_ids)
        graph_visited_ids.update(visited_ids)
        logger.info(f"Final number of visited nodes: {len(visited_ids)}")
        return G

    # Sort by score descending
    candidate_list = sorted(candidate_list, key=lambda x: x["score"], reverse=True)

    # Check if top score is high enough
    highest_score = candidate_list[0]["score"]
    if highest_score < 0.43:
        logger.info(f"Top score {highest_score} is below threshold. Ending search.")
        G, visited_ids = build_prereq_postreq(class_retriever, already_selected_classes, db_handler, query_embedding, logger=logger, existing_visited_ids=graph_visited_ids)
        graph_visited_ids.update(visited_ids)
        logger.info(f"Final number of visited nodes: {len(visited_ids)}")
        return G

    # Select candidates from diverse departments
    selected_candidates = []
    seen_departments = set()

    for candidate in candidate_list:
        if candidate["department_name"] not in seen_departments:
            selected_candidates.append(candidate)
            seen_departments.add(candidate["department_name"])
        if len(selected_candidates) >= 2:
            break

    # Add selected candidates
    for candidate in selected_candidates:
        logger.info(f"candidate: {candidate}")
        candidate_id = candidate.get("class_id")

        if candidate_id in visited_ids:
            logger.info(f"Candidate {candidate_id} already selected. Skipping.")
            continue

        already_selected_classes.append(candidate)
        logger.info(f"Added candidate: {candidate_id} (total: {len(already_selected_classes)})")
        logger.info(f'already_selected_classes: {already_selected_classes}')

    # Update graph
    G, graph_visited_ids = build_prereq_postreq(class_retriever, already_selected_classes, db_handler, query_embedding, logger=logger, existing_visited_ids=graph_visited_ids)
    graph_visited_ids.update(visited_ids)
    
    # Recursive call
    return recursive_top1_selection(
        client, db_handler, query_embedding, query, 
        selected_dept_list, class_retriever, graph_path, gt_department,
        already_selected_classes, graph_visited_ids, depth+1
    )


def main():
    args = get_args()
    db_config = read_json(args.db_config_path)
    
    # Initialize OpenAI client
    openai_api_key = args.openai_api_key  
    client = initialize_openai_client(openai_api_key)

    db_handler = DatabaseHandler(
        host=db_config["host"],
        port=db_config["port"],
        user=db_config["user"],
        password=db_config["password"],
        database=db_config["database"],
        charset=db_config["charset"]
    )
    db_handler.connect()
    print("Fetching department data...")

    # Department selection and embedding
    if args.goal_index_path is None:
        depart_dataset = goalDatasetjson(json.load(open(args.department_path)))
        retriever = DenseRetriever(client, args, depart_dataset)
        retriever.doc_embedding()
    else: 
        retriever = DenseRetriever(client, args)
        retriever.doc_embedding()

    # Class information embedding
    if args.class_index_path is None:
        class_info = db_handler.fetch_classes_info_by_department()
        print(f'First class_info: {class_info[0]}')
        class_dataset = class_depart_Dataset(class_info)
        class_retriever = classRetriever(client, args, class_dataset)
        class_retriever.doc_embedding()
    else:
        class_retriever = classRetriever(client, args)
        class_retriever.doc_embedding()

    query_path = read_json(args.query_path)

    query_list = []
    expanded_query = []
    gt_list = []
    department_list = []
    gt_len = []
    ordered_query_path = OrderedDict(sorted(query_path.items(), key=lambda x: int(x[0])))

    for idx, content in ordered_query_path.items(): 
        major_name = content['major_name']
        query_count = len(content['query_list'])
        department = content['departments']
        required_dept_count = content['ground_truth_len']

        gt_list.extend([major_name] * query_count)
        department_list.extend([department] * query_count)
        gt_len.extend([required_dept_count] * query_count)

        for query in content['query_list']:
            query_list.append(query['query'])
            expanded_query.append(query['expanded_query'])

    for idx, (query, expaned_query, gt_department, selected_depart_lists, required_dept_count) in enumerate(zip(query_list, expanded_query, gt_list, department_list, gt_len)): 
        print(f"Query: {query}")
        logging.info(f'Query query_{idx}: {query}')
        
        graph_path = os.path.join(args.save_path, f"recommendations_query_exp_siamilar_top{args.top_k}")
        query_info = query
        if args.query_exp:
            query_info = expaned_query
            graph_path = os.path.join(args.save_path, f"recommendations_query_exp_similar_top{args.top_k}")

        query_embbeding = retriever.query_embedding(query_info)

        if args.department_y and args.query_exp:
            selected_depart_list = selected_depart_lists
            graph_path = os.path.join(args.save_path, f"recommendations_query_exp_similar_top{args.top_k}_dept_y")
            graph_path2 = os.path.join(args.save_path, f"recommendations_query_exp_similar_top{args.top_k}_dept_y_edge")
        elif args.department_y:
            selected_depart_list = selected_depart_lists
            graph_path = os.path.join(args.save_path, f"recommendations_query_no_similar_top{args.top_k}_dept_y_")   
            graph_path2 = os.path.join(args.save_path, f"recommendations_query_no_similar_top{args.top_k}_dept_y_edge")   
        elif args.query_exp:
            selected_depart_list = retriever.retrieve(query_embbeding)
            print(f'len(selected_depart_list): {len(selected_depart_list)}')
            graph_path = os.path.join(args.save_path, f"recommendations_query_exp_similar_top{args.top_k}_dept_N")
            graph_path2 = os.path.join(args.save_path, f"recommendations_query_exp_similar_top{args.top_k}_dept_N_edge")
            graph_path3 = os.path.join(args.save_path, f"recommendations_query_exp_similar_top{args.top_k}_dept_N_refine")
        else:
            selected_depart_list = retriever.retrieve(query_embbeding)
            print(f'len(selected_depart_list): {len(selected_depart_list)}')
            graph_path = os.path.join(args.save_path, f"recommendations_query_no_similar_top{args.top_k}_dept_N")
            graph_path2 = os.path.join(args.save_path, f"recommendations_query_no_similar_top{args.top_k}_dept_N_edge")
        
        required_dept_count = 50
        G = recursive_top1_selection(
            client,
            db_handler,
            query_embbeding,
            query,
            selected_depart_list,
            class_retriever,
            graph_path,
            gt_department
        )

        all_results_json1 = visualize_and_sort_department_graphs(G, graph_path, idx, gt_department)
        G_edge = selected_edge_by_llm(client, query, G)
        all_results_json2 = visualize_and_sort_department_graphs(G_edge, graph_path2, idx, gt_department)
        # all_results_json3 = refine_visualize_graph(G, graph_path3, idx, gt_department)

    db_handler.close()

if __name__ == "__main__":
    main()


