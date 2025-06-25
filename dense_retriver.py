import os 
from typing import List, Dict 
from transformers import DataCollatorWithPadding, DefaultDataCollator
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import openai
import numpy as np
import pickle
from typing import Union
import faiss
from dataset.data_collector import collect_goal, collect_class
from rank_bm25 import BM25Okapi
import tiktoken
import json

class DenseRetriever:
    def __init__(self, client, args, dataset=None):
        self.client = client
        self.dataset = dataset
        self.args = args
        self.index = None
        self.lookup_index = None


    def split_text_to_chunks(self, text, max_tokens, model="text-embedding-3-large"):
    
        encoder = tiktoken.encoding_for_model(model)
        tokens = encoder.encode(text)  

        chunks = []
        current_chunk = []
        current_length = 0

        for token in tokens:
            if current_length + 1 > max_tokens:
                chunks.append(encoder.decode(current_chunk))
                current_chunk = [token]  
                current_length = 1
            else:
                current_chunk.append(token)
                current_length += 1

        # 마지막 청크 추가
        if current_chunk:
            chunks.append(encoder.decode(current_chunk))

        return chunks


    def get_gpt_embedding(self, text):

        max_tokens = 8000  
        chunks = self.split_text_to_chunks(text, max_tokens) 
        chunk_embeddings = []


        for chunk in chunks:
            response = self.client.embeddings.create(input=chunk, model='text-embedding-3-large')
            chunk_embeddings.append(response.data[0].embedding)

        # Combine embeddings of chunks by averaging
        return np.mean(chunk_embeddings, axis=0)


    def doc_embedding(self):
        
        if self.args.goal_index_path is not None:
            # Load from saved file
            save_file = os.path.join(self.args.save_path, "goal_Dataset.pkl")
            with open(save_file, "rb") as f:
                data = pickle.load(f)
                embeddings = data["embeddings"]
                embeddings_array = np.array(embeddings, dtype=np.float32)
                self.lookup_index = data["lookup_index"]
    

        os.makedirs(self.args.save_path, exist_ok=True)

        if self.dataset is not None: 
            embeddings = []
            lookup_index = []

            # Create DataLoader
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                collate_fn=collect_goal
            )

            # Generate embeddings
            for batch in tqdm(dataloader, desc="Generating embeddings"):            
                batch_embeddings = []
                for text in batch["text"]:
                    # print(text)
                    embedding = self.get_gpt_embedding(text)
                    batch_embeddings.append(embedding)
                    
                embeddings.extend(batch_embeddings)
                embeddings_array = np.array(embeddings, dtype=np.float32)
                lookup_index.extend(
                    [{'department_id':dept_id, 'department_name':dept_name, 'text':text}
                    for dept_id, dept_name in zip(batch['department_id'], batch['department_name'])]
                )
                self.lookup_index = lookup_index
                
            
            #save embeddings as TSV
            embeddings_files = os.path.join(self.args.save_path, 'goal_embeddings.tsv')
            metadata_files = os.path.join(self.args.save_path, 'goal_metadata.tsv')
            
            np.savetxt(embeddings_files, embeddings_array, delimiter='\t')
            with open(metadata_files, 'w') as f:
                for item in lookup_index:
                    f.write(f'{item["department_name"]}\n')
            
            # Save embeddings
            save_file = os.path.join(self.args.save_path, "goal_Dataset.pkl")
            with open(save_file, "wb") as f:
                pickle.dump({"embeddings": embeddings_array, "lookup_index": lookup_index}, f)
                    

        # # Create FAISS index
        embeddings_array /= np.linalg.norm(embeddings_array, axis=1, keepdims=True)  # Normalize

            
        dim = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings_array)

        return self.index, self.lookup_index 


    def query_embedding(self, query):
        return self.get_gpt_embedding(query)
    
    
    
    def retrieve(self, query, top_k=5,threshold_diff=0.015): #threshold_diff=0.015
        # Generate query embedding
        query_emb = np.array(query, dtype=np.float32).reshape(1, -1)
        query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True)  # cosine similarity normalization

        # Perform FAISS search
        similarities, indices = self.index.search(query_emb, k=top_k)

        # 첫 번째 결과는 무조건 포함
        selected_results = []
        prev_score = None

        for i, (idx, score) in enumerate(zip(indices[0], similarities[0])):
            if i >= 2 and prev_score is not None and abs(prev_score - score) >= threshold_diff:
                break  # top 2까지는 선택하고 이후에는 threshold_diff 기준으로 중단
            doc_info = self.lookup_index[idx]
            selected_results.append({
                "department_id": doc_info["department_id"],
                "department_name": doc_info.get("department_name", "Unknown"),
                "score": float(score)
            })
                
            prev_score = score       
        return selected_results


class classRetriever(DenseRetriever):
    def __init__(self, client, args, dataset=None):
        super().__init__(client, args, dataset)
        self.dataset = dataset
        self.class_index = None
        self.class_embeddings = None
        self.class_lookup_index = None


    def doc_embedding(self):
        
        if self.args.class_index_path is not None:
            save_file = os.path.join(self.args.save_path, "class_Dataset.pkl")
            with open(save_file, "rb") as f:
                data = pickle.load(f)
                self.class_embeddings = np.array(data["embeddings"], dtype=np.float32)
                self.class_lookup_index = data["lookup_index"]

        else:
            os.makedirs(self.args.save_path, exist_ok=True)
            
            if self.dataset is not None:
                class_embeddings = []
                class_lookup_index = []
                
                dataloader = DataLoader(
                    self.dataset,
                    batch_size=self.args.batch_size,
                    shuffle=False,
                    collate_fn=collect_class
                )
                
                for batch in tqdm(dataloader, desc="Generating class embddings"):
            
                    batch_embeddings = []
                    for text in batch['text']:
                        print(text)
                        embeddings = self.get_gpt_embedding(text)
                        batch_embeddings.append(embeddings)
                    class_embeddings.extend(batch_embeddings)
                    embeddings_array = np.array(class_embeddings, dtype=np.float32)
                    self.class_embeddings = embeddings_array
                    class_lookup_index.extend([
                        {
                            'class_id': class_id,
                            'class_name': class_name,
                            'student_grade': student_grade,
                            'semester': semester, 
                            'department_id': department_id,
                            'department_name': department_name,
                            'prerequisite': prerequisite,
                            'text': text
                        }
                        for class_id, class_name, student_grade, semester, department_id, department_name, prerequisite in zip(
                            batch['class_id'], batch['class_name'], batch['student_grade'], batch['semester'], batch['department_id'], batch['department_name'],batch['prerequisite']
                        )
                    ])
                self.class_lookup_index = class_lookup_index
        
                np.savetxt(os.path.join(self.args.save_path, 'class_embeddings.tsv'), embeddings_array, delimiter='\t') 
                with open(os.path.join(self.args.save_path, 'class_metadata.tsv'), 'w') as f:
                    for item in class_lookup_index:
                        f.write(f'{item["class_name"]}\n')
                # Save as Pickle
                save_file = os.path.join(self.args.save_path, "class_Dataset.pkl")
                with open(save_file, "wb") as f:
                    pickle.dump({"embeddings": embeddings_array, "lookup_index": class_lookup_index}, f)
          
            return self.class_embeddings, self.class_lookup_index

    
    def filter_by_department(self, selected_depart_list, exclude_ids=None):

        # 선택된 학과 ID 추출
        selected_depart_ids = {dept['department_id'] for dept in selected_depart_list}
        
        filtered_embeddings = []
        filtered_lookup_index = []
        
        for emb, info in zip(self.class_embeddings, self.class_lookup_index):
            if info["department_id"] in selected_depart_ids:
                # exclude_ids에 포함된 클래스는 제외
                if exclude_ids is not None and info["class_id"] in exclude_ids:
                    continue
                filtered_embeddings.append(emb)
                filtered_lookup_index.append(info)
        
        if len(filtered_embeddings) == 0:
            print("⚠ No matching classes found for selected departments.")
            return None, None

        # numpy array로 변환 후 정규화
        filtered_embeddings = np.array(filtered_embeddings, dtype=np.float32)
        filtered_embeddings /= np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)
        
        dim = filtered_embeddings.shape[1]
        self.class_index = faiss.IndexFlatIP(dim)
        self.class_index.add(filtered_embeddings)
        
        self.lookup_index = filtered_lookup_index
        return self.class_index, self.lookup_index


    
    def retrieve(self, query_embedding, selected_depart_list, top_k=15, visited_class_ids=None):
        query_emb = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True)

        results_dict = {}
        all_results = []
        
        for dept in selected_depart_list:
            dept_name = dept["department_name"]
            if dept['department_id']:
                dept_id = dept["department_id"]
                
            # 2) filter_by_department() -> new index is stored in self.class_index
            _index, _lookup = self.filter_by_department([dept],visited_class_ids)
            if _index is None or _lookup is None:
                print(f"⚠ {dept_name}에는 검색할 데이터가 없음")
                results_dict[dept_name] = []
                continue
            
            similarities, indices = self.class_index.search(query_emb, k=top_k)
            
            for idx, score in zip(indices[0], similarities[0]):
                doc_info = _lookup[idx]
                all_results.append({
                    "class_id": doc_info["class_id"],
                    "class_name": doc_info.get("class_name", "Unknown"),
                    "student_grade": doc_info.get("student_grade", "Unknown"),
                    "semester": doc_info.get("semester", "Unknown"),
                    "department_id": doc_info["department_id"],
                    "department_name": doc_info.get("department_name", "Unknown"),
                    "prerequisite": doc_info.get("prerequisite", "Unknown"),
                    "description": doc_info.get("text", "Unknown"),
                    "score": float(score)
                })
        
    
        top_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:top_k]
        
        # department saved
        for result in top_results:
            dept_name = result["department_name"]
            if dept_name not in results_dict:
                results_dict[dept_name] = []
            results_dict[dept_name].append(result)
        
        return results_dict


    
    def retrieve_by_department(self, query_embedding, selected_depart_list, top_k=3, visited_class_ids=None):
    
        query_emb = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True)

        results_dict = {}
        
        for dept in selected_depart_list:
            dept_name = dept["department_name"]
            if dept['department_id']:
                dept_id = dept["department_id"]
                
            _index, _lookup = self.filter_by_department([dept],visited_class_ids)
            if _index is None or _lookup is None:
                print(f"⚠ {dept_name}에는 검색할 데이터가 없음")
                results_dict[dept_name] = []
                continue
            

            similarities, indices = self.class_index.search(query_emb, k=top_k)
            dept_results = []
            for idx, score in zip(indices[0], similarities[0]):
                doc_info = _lookup[idx]
                dept_results.append({
                    "class_id": doc_info["class_id"],
                    "class_name": doc_info.get("class_name", "Unknown"),
                    "student_grade": doc_info.get("student_grade", "Unknown"),
                    "semester": doc_info.get("semester", "Unknown"),
                    "department_id": doc_info["department_id"],
                    "department_name": doc_info.get("department_name", "Unknown"),
                    "prerequisite": doc_info.get("prerequisite", "Unknown"),
                    "description": doc_info.get("text", "Unknown"),
                    "score": float(score)
            })
            dept_results = sorted(dept_results, key=lambda x: x["score"], reverse=True)
            
            results_dict[dept_name] = dept_results
            print(f"🔍 {dept_name} 검색 결과: {len(dept_results)}개")
            
        return results_dict

def filter_by_4grade(self, selected_depart_list, exclude_ids=None, filter_senior=False):
    selected_depart_ids = {dept['department_id'] for dept in selected_depart_list}

    filtered_embeddings = []
    filtered_lookup_index = []

    for emb, info in zip(self.class_embeddings, self.class_lookup_index):
        if filter_senior and info.get("student_grade") != 4:
            continue

        if info["department_id"] in selected_depart_ids:
            if exclude_ids is not None and info["class_id"] in exclude_ids:
                continue
            filtered_embeddings.append(emb)
            filtered_lookup_index.append(info)

    if len(filtered_embeddings) == 0:
        print("⚠ No matching 4학년 classes found for selected departments.")
        return None, None

    # 🔍 FAISS 인덱스 업데이트
    filtered_embeddings = np.array(filtered_embeddings, dtype=np.float32)
    filtered_embeddings /= np.linalg.norm(filtered_embeddings, axis=1, keepdims=True)

    dim = filtered_embeddings.shape[1]
    self.class_index = faiss.IndexFlatIP(dim)
    self.class_index.add(filtered_embeddings)

    self.lookup_index = filtered_lookup_index
    return self.class_index, self.lookup_index


def retrieve_by_4grade(self, query_embedding, selected_depart_list, top_k=3, visited_class_ids=None):
    query_emb = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True)

    results_dict = {}

    for dept in selected_depart_list:
        dept_name = dept["department_name"]
        if dept['department_id']:
            dept_id = dept["department_id"]

        _index, _lookup = self.filter_by_4grade([dept], visited_class_ids, filter_senior=True)
        if _index is None or _lookup is None:
            print(f"⚠ {dept_name}에는 검색할 4학년 데이터가 없음")
            results_dict[dept_name] = []
            continue

        similarities, indices = self.class_index.search(query_emb, k=top_k)
        dept_results = []
        for idx, score in zip(indices[0], similarities[0]):
            doc_info = _lookup[idx]

            dept_results.append({
                "class_id": doc_info["class_id"],
                "class_name": doc_info.get("class_name", "Unknown"),
                "student_grade": doc_info.get("student_grade", "Unknown"),
                "semester": doc_info.get("semester", "Unknown"),
                "department_id": doc_info["department_id"],
                "department_name": doc_info.get("department_name", "Unknown"),
                "prerequisite": doc_info.get("prerequisite", "Unknown"),
                "description": doc_info.get("text", "Unknown"),
                "score": float(score)
            })
        dept_results = sorted(dept_results, key=lambda x: x["score"], reverse=True)
        results_dict[dept_name] = dept_results
        print(f"🔍 {dept_name} 4학년 검색 결과: {len(dept_results)}개")

    return results_dict
