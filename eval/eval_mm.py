import json
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Fix seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# Load JSON file
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Extract predicted course names and departments from JSON
def get_predicted_courses(pred_json_path):
    data = load_json(pred_json_path)

    predicted_names = set()
    predicted_departments = set()

    for course in data:
        course_names = course["course_name"]
        department_names = course["department"]

        if isinstance(course_names, list):
            predicted_names.add(frozenset(course_names))
        else:
            predicted_names.add(course_names)

        if isinstance(department_names, list):
            predicted_departments.add(frozenset(department_names))
        else:
            predicted_departments.add(department_names)

    # Remove frozensets that overlap with single departments
    final_departments = set()
    single_departments = {dept for dept in predicted_departments if isinstance(dept, str)}

    for dept in predicted_departments:
        if isinstance(dept, frozenset):
            if not dept.intersection(single_departments):
                final_departments.add(dept)
        else:
            final_departments.add(dept)

    return predicted_names, final_departments

# Extract ground truth course names and departments from JSON
def get_ground_truth_courses(gt_json_path):
    data = load_json(gt_json_path)

    ground_truth_names = set()
    ground_truth_departments = set()

    for course in data:
        course_names = course["course_name"]
        department_names = course["department"]

        if isinstance(course_names, list):
            ground_truth_names.add(frozenset(course_names))
        else:
            ground_truth_names.add(course_names)

        if isinstance(department_names, list):
            ground_truth_departments.add(frozenset(department_names))
        else:
            ground_truth_departments.add(department_names)

    # Remove frozensets that overlap with single departments
    final_departments = set()
    single_departments = {dept for dept in ground_truth_departments if isinstance(dept, str)}

    for dept in ground_truth_departments:
        if isinstance(dept, frozenset):
            if not dept.intersection(single_departments):
                final_departments.add(dept)
        else:
            final_departments.add(dept)

    return ground_truth_names, final_departments

# Calculate precision, recall, and F1-score for course name evaluation
def calculate_metrics_for_names(predicted_names, ground_truth_names):
    true_positives = predicted_names & ground_truth_names
    false_positives = predicted_names - ground_truth_names
    false_negatives = ground_truth_names - predicted_names

    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if predicted_names else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if ground_truth_names else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

# Calculate precision, recall, and F1-score for department evaluation
def calculate_metrics_for_departments(predicted_departments, ground_truth_departments):
    true_positives = predicted_departments & ground_truth_departments
    false_positives = predicted_departments - ground_truth_departments
    false_negatives = ground_truth_departments - predicted_departments

    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if predicted_departments else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if ground_truth_departments else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

# Main evaluation function
def evaluate(pred_json, gt_json):
    """
    Run evaluation on predicted and ground truth JSON files.
    Returns:
        - Course name precision, recall, F1
        - Department precision, recall, F1
    """
    predicted_names, predicted_departments = get_predicted_courses(pred_json)
    print(f"Predicted course names: {predicted_names}")
    print(f"Predicted departments: {predicted_departments}\n")

    ground_truth_names, ground_truth_departments = get_ground_truth_courses(gt_json)
    print(f"Ground truth course names: {ground_truth_names}")
    print(f"Ground truth departments: {ground_truth_departments}\n")
    print("=" * 50)

    name_precision, name_recall, name_f1 = calculate_metrics_for_names(predicted_names, ground_truth_names)
    dept_precision, dept_recall, dept_f1 = calculate_metrics_for_departments(predicted_departments, ground_truth_departments)

    return name_precision, name_recall, name_f1, dept_precision, dept_recall, dept_f1
