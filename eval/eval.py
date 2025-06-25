import json
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 🔹 Seed 고정 (일관된 결과 보장)
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# 🔹 JSON 파일 로드 함수
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 🔹 추론된 커리큘럼(JSON에서 name & department_name 값 가져옴)
def get_predicted_courses(pred_json_path):
        ground_truth_data = load_json(pred_json_path)
        
        ground_truth_names = set()
        ground_truth_departments = set()

        for course in ground_truth_data:
            course_names = course["course_name"]
            department_names = course["department"]  # 학과 정보 추가

            # 🔹 과목명이 리스트인 경우 처리 (둘 중 하나라도 맞으면 정답)
            if isinstance(course_names, list):  
                ground_truth_names.add(frozenset(course_names))  # 리스트를 frozenset으로 변환하여 저장
            else:
                ground_truth_names.add(course_names)  # 단일 과목 추가
            
            # 🔹 학과명이 리스트인 경우 처리 (둘 중 하나라도 맞으면 정답)
            if isinstance(department_names, list):  
                ground_truth_departments.add(frozenset(department_names))  # 리스트를 frozenset으로 변환하여 저장
            else:
                ground_truth_departments.add(department_names)  # 단일 학과 추가
                
        # 🔥 단일 학과와 겹치는 frozenset 제거
        final_departments = set()
        single_departments = {dept for dept in ground_truth_departments if isinstance(dept, str)}

        for dept in ground_truth_departments:
            if isinstance(dept, frozenset):  # 학과 리스트 형태인 경우
                # 🔹 frozenset 내부 학과가 이미 단일 학과에 포함되면 제외
                if not dept.intersection(single_departments):  
                    final_departments.add(dept)
            else:
                final_departments.add(dept)

        return ground_truth_names, final_departments

# 🔹 정답 커리큘럼(JSON에서 course_name & department 값 가져옴)
def get_ground_truth_courses(gt_json_path):
    ground_truth_data = load_json(gt_json_path)
    
    ground_truth_names = set()
    ground_truth_departments = set()

    for course in ground_truth_data:
        course_names = course["course_name"]
        department_names = course["department"]  # 학과 정보 추가

        # 🔹 과목명이 리스트인 경우 처리 (둘 중 하나라도 맞으면 정답)
        if isinstance(course_names, list):  
            ground_truth_names.add(frozenset(course_names))  # 리스트를 frozenset으로 변환하여 저장
        else:
            ground_truth_names.add(course_names)  # 단일 과목 추가
        
        # 🔹 학과명이 리스트인 경우 처리 (둘 중 하나라도 맞으면 정답)
        if isinstance(department_names, list):  
            ground_truth_departments.add(frozenset(department_names))  # 리스트를 frozenset으로 변환하여 저장
        else:
            ground_truth_departments.add(department_names)  # 단일 학과 추가
            
    # 🔥 단일 학과와 겹치는 frozenset 제거
    final_departments = set()
    single_departments = {dept for dept in ground_truth_departments if isinstance(dept, str)}

    for dept in ground_truth_departments:
        if isinstance(dept, frozenset):  # 학과 리스트 형태인 경우
            # 🔹 frozenset 내부 학과가 이미 단일 학과에 포함되면 제외
            if not dept.intersection(single_departments):  
                final_departments.add(dept)
        else:
            final_departments.add(dept)

    return ground_truth_names, final_departments

# 🔹 Precision, Recall, F1-score 계산 함수 (과목명 평가)
def calculate_metrics_for_names(predicted_names, ground_truth_names):
    true_positives = predicted_names & ground_truth_names  # 교집합
    false_positives = predicted_names - ground_truth_names  # 예측했지만 정답이 아님
    false_negatives = ground_truth_names - predicted_names  # 정답이지만 예측하지 못함

    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if predicted_names else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if ground_truth_names else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

# 🔹 Precision, Recall, F1-score 계산 함수 (학과 평가)
def calculate_metrics_for_departments(predicted_departments, ground_truth_departments):
    true_positives = predicted_departments & ground_truth_departments  # 교집합
    false_positives = predicted_departments - ground_truth_departments  # 예측했지만 정답이 아님
    false_negatives = ground_truth_departments - predicted_departments  # 정답이지만 예측하지 못함

    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if predicted_departments else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if ground_truth_departments else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

# 🔹 평가 실행 함수
def evaluate(pred_json, gt_json):
    """
    평가 실행 함수 (과목명 & 학과명 개별 평가)
    - return: (과목명 precision, recall, f1), (학과 precision, recall, f1)
    """
    predicted_names, predicted_departments = get_predicted_courses(pred_json)
    print(f"predictd_courses ::: {predicted_names}")
    print(f"predicted_departments ::: {predicted_departments}\n\n")
    
    ground_truth_names, ground_truth_departments = get_ground_truth_courses(gt_json)
    print(f"ground_truth_courses :: {ground_truth_names}")
    print(f"ground_truth_departments :: {ground_truth_departments}\n\n")
    print("="*50)

    # 🔹 과목명 평가
    name_precision, name_recall, name_f1 = calculate_metrics_for_names(predicted_names, ground_truth_names)

    # 🔹 학과 평가
    dept_precision, dept_recall, dept_f1 = calculate_metrics_for_departments(predicted_departments, ground_truth_departments)

    return name_precision, name_recall, name_f1, dept_precision, dept_recall, dept_f1


