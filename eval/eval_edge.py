import json
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

set_seed()

def load_json(file_path):
    """Load a JSON file and return it as a Python object."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_predicted_courses(pred_json_path):
    """
    Load predicted courses from JSON and extract course_name and department.
    """
    data = load_json(pred_json_path)

    print(f"Debug: Loaded JSON type={type(data)}, value={list(data.keys())}")

    # If JSON is a dict, concatenate all department course lists
    if isinstance(data, dict):
        all_courses = []
        for department, info in data.items():
            if "nodes" in info and isinstance(info["nodes"], list):
                all_courses.extend(info["nodes"])
        data = all_courses

    if not isinstance(data, list):
        raise ValueError(f"Invalid JSON format: Expected list, got {type(data)}")

    ground_truth_names = set()
    ground_truth_departments = set()

    for course in data:
        if not isinstance(course, dict):
            print(f"Warning: Unexpected data format {course}")
            continue

        course_names = course.get("course_name")
        department_names = course.get("department")

        # Add course names
        if isinstance(course_names, list):
            ground_truth_names.update(course_names)
        elif isinstance(course_names, str):
            ground_truth_names.add(course_names)

        # Add department names
        if isinstance(department_names, list):
            ground_truth_departments.update(department_names)
        elif isinstance(department_names, str):
            ground_truth_departments.add(department_names)

    return ground_truth_names, ground_truth_departments

def get_ground_truth_courses(gt_json_path):
    """
    Load ground truth courses and extract course_name and department sets.
    """
    ground_truth_data = load_json(gt_json_path)
    node = ground_truth_data.get("node", [])

    ground_truth_names = {course["course_name"] for course in node}
    ground_truth_departments = {course["department"] for course in node}

    return ground_truth_names, ground_truth_departments

def get_predicted_edges(pred_json_path):
    """
    Extract predicted edges from JSON and return as a set of (from, to) tuples.
    """
    predicted_data = load_json(pred_json_path)
    print(f"Debug: Loaded JSON type={type(predicted_data)}, keys={list(predicted_data.keys())}")

    predicted_edges = set()

    if isinstance(predicted_data, dict):
        for department, info in predicted_data.items():
            if "edges" in info and isinstance(info["edges"], list):
                for edge in info["edges"]:
                    if isinstance(edge, dict) and "from" in edge and "to" in edge:
                        predicted_edges.add((edge["from"], edge["to"]))
                    else:
                        print(f"Warning: Unexpected edge format: {edge}")

    return predicted_edges

def get_ground_truth_edges(gt_json_path):
    """
    Extract ground truth edges as a set of (from, to) tuples.
    """
    ground_truth_data = load_json(gt_json_path)
    ground_truth_edges = ground_truth_data.get("edge", [])
    return {(edge["from"], edge["to"]) for edge in ground_truth_edges}

def calculate_metrics(predicted, ground_truth):
    """
    Calculate precision, recall, F1-score, and counts of TP, FP, FN.
    """
    all_items = list(predicted | ground_truth)
    y_true = np.array([1 if item in ground_truth else 0 for item in all_items])
    y_pred = np.array([1 if item in predicted else 0 for item in all_items])

    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    precision = precision_score(y_true, y_pred) if (TP + FP) > 0 else 0
    recall = recall_score(y_true, y_pred) if (TP + FN) > 0 else 0
    f1 = f1_score(y_true, y_pred) if (precision + recall) > 0 else 0

    return precision, recall, f1, TP, FP, FN

def evaluate(pred_json, gt_json):
    """
    Run evaluation for predicted curriculum against ground truth.
    Includes:
      - course_name evaluation
      - department evaluation
      - prerequisite relation (edge) evaluation

    Returns:
        Dictionary containing metrics for nodes and edges.
    """
    # Evaluate node-level predictions
    predicted_names, predicted_departments = get_predicted_courses(pred_json)
    ground_truth_names, ground_truth_departments = get_ground_truth_courses(gt_json)

    name_precision, name_recall, name_f1, name_TP, name_FP, name_FN = calculate_metrics(predicted_names, ground_truth_names)
    dept_precision, dept_recall, dept_f1, dept_TP, dept_FP, dept_FN = calculate_metrics(predicted_departments, ground_truth_departments)

    # Evaluate edge-level predictions
    predicted_edges = get_predicted_edges(pred_json)
    ground_truth_edges = get_ground_truth_edges(gt_json)

    edge_precision, edge_recall, edge_f1, edge_TP, edge_FP, edge_FN = calculate_metrics(predicted_edges, ground_truth_edges)

    return {
        "node_evaluation": {
            "course_name": {
                "precision": name_precision,
                "recall": name_recall,
                "f1_score": name_f1,
                "TP": name_TP,
                "FP": name_FP,
                "FN": name_FN
            },
            "department": {
                "precision": dept_precision,
                "recall": dept_recall,
                "f1_score": dept_f1,
                "TP": dept_TP,
                "FP": dept_FP,
                "FN": dept_FN
            }
        },
        "edge_evaluation": {
            "prerequisite_relation": {
                "precision": edge_precision,
                "recall": edge_recall,
                "f1_score": edge_f1,
                "TP": edge_TP,
                "FP": edge_FP,
                "FN": edge_FN
            }
        }
    }
