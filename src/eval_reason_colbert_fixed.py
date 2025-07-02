import argparse
import json
import logging
import os
import time
import glob
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from beir.datasets.data_loader import GenericDataLoader
from pylate import models

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
#### /print debug information to stdout

class ReasonColBERTModel:
    def __init__(self, model_name="lightonai/Reason-ModernColBERT"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Use PyLate ColBERT model for efficient batch encoding
        self.model = models.ColBERT(
            model_name_or_path=model_name,
            device=self.device
        )
        
        logging.info(f"Model loaded: {model_name}")
    
    def encode_batch(self, texts, is_query=True, batch_size=8):
        """Encode a batch of texts (queries or documents) efficiently"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            is_query=is_query,
            show_progress_bar=True,
            convert_to_numpy=False,  # Keep as tensors for compatibility
            normalize_embeddings=True
        )
        return embeddings
    
    def maxsim_score(self, query_embeddings, doc_embeddings):
        """Calculate MaxSim score between query and document embeddings"""
        # query_embeddings: [query_seq_len, hidden_size]
        # doc_embeddings: [doc_seq_len, hidden_size]
        
        # Normalize embeddings
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(query_embeddings, doc_embeddings.T)  # [query_seq_len, doc_seq_len]
        
        # MaxSim: for each query token, take max similarity with any doc token
        max_similarities = torch.max(similarity_matrix, dim=1)[0]  # [query_seq_len]
        
        # Sum over all query tokens (excluding padding)
        # Filter out zero embeddings (padding tokens)
        non_zero_mask = torch.sum(torch.abs(query_embeddings), dim=-1) > 1e-8
        score = torch.sum(max_similarities[non_zero_mask])
        
        return score.item()

def evaluate_single_instance(model, instance_path):
    """Evaluate a single instance"""
    try:
        # Load dataset for this instance
        corpus, queries, qrels = GenericDataLoader(data_folder=instance_path).load(split="test")
        
        instance_name = os.path.basename(instance_path)
        logging.info(f"Processing {instance_name}: {len(corpus)} documents, {len(queries)} queries")
        
        # Prepare documents and queries
        documents_list = []
        documents_ids = []
        for doc_id, doc_data in corpus.items():
            documents_list.append(doc_data["text"])
            documents_ids.append(doc_id)
        
        queries_list = []
        queries_ids = []
        for query_id, query_text in queries.items():
            queries_list.append(query_text)
            queries_ids.append(query_id)
        
        # Encode documents in batches
        logging.info(f"Encoding {len(documents_list)} documents...")
        doc_embeddings = model.encode_batch(
            documents_list,
            is_query=False,
            batch_size=4
        )
        
        # Encode queries in batches
        logging.info(f"Encoding {len(queries_list)} queries...")
        query_embeddings = model.encode_batch(
            queries_list,
            is_query=True,
            batch_size=4
        )
        
        # Calculate scores for each query-document pair
        start_time = time.time()
        results = {}
        
        for i, query_id in enumerate(queries_ids):
            query_emb = query_embeddings[i]  # [query_seq_len, hidden_size]
            
            # Calculate scores with all documents
            scores = []
            for j, doc_id in enumerate(documents_ids):
                doc_emb = doc_embeddings[j]  # [doc_seq_len, hidden_size]
                
                # Calculate MaxSim score
                score = model.maxsim_score(query_emb, doc_emb)
                scores.append((doc_id, score))
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Store results
            results[query_id] = {doc_id: score for doc_id, score in scores}
        
        end_time = time.time()
        
        # Calculate metrics manually for k values [1, 3, 5, 10, 100, 1000]
        k_values = [1, 3, 5, 10, 100, 1000]
        instance_results = {}
        
        for k in k_values:
            # NDCG@k, Recall@k, Precision@k, MRR@k
            ndcg_scores = []
            recall_scores = []
            precision_scores = []
            mrr_scores = []
            
            for query_id in queries_ids:
                if query_id not in qrels:
                    continue
                    
                relevant_docs = set(qrels[query_id].keys())
                if not relevant_docs:
                    continue
                
                # Get top-k results for this query
                query_results = results.get(query_id, {})
                sorted_results = sorted(query_results.items(), key=lambda x: x[1], reverse=True)[:k]
                retrieved_docs = [doc_id for doc_id, _ in sorted_results]
                
                # Calculate Recall@k
                retrieved_relevant = len(set(retrieved_docs) & relevant_docs)
                recall = retrieved_relevant / len(relevant_docs) if relevant_docs else 0
                recall_scores.append(recall)
                
                # Calculate Precision@k
                precision = retrieved_relevant / len(retrieved_docs) if retrieved_docs else 0
                precision_scores.append(precision)
                
                # Calculate NDCG@k
                dcg = 0
                for idx, doc_id in enumerate(retrieved_docs):
                    if doc_id in relevant_docs:
                        relevance = qrels[query_id].get(doc_id, 0)
                        dcg += relevance / np.log2(idx + 2)  # NDCG formula
                
                # Calculate IDCG (ideal DCG)
                ideal_relevances = sorted([qrels[query_id].get(doc_id, 0) for doc_id in relevant_docs], reverse=True)[:k]
                idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)
                
                # Calculate MRR@k
                mrr = 0
                for idx, doc_id in enumerate(retrieved_docs):
                    if doc_id in relevant_docs:
                        mrr = 1 / (idx + 1)
                        break
                mrr_scores.append(mrr)
            
            # Average the scores
            instance_results[f"NDCG@{k}"] = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
            instance_results[f"Recall@{k}"] = sum(recall_scores) / len(recall_scores) if recall_scores else 0
            instance_results[f"P@{k}"] = sum(precision_scores) / len(precision_scores) if precision_scores else 0
            instance_results[f"MRR@{k}"] = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
        
        instance_results["time"] = (end_time - start_time) / len(queries) if queries else 0
        
        return instance_results, results
        
    except Exception as e:
        logging.error(f"Error processing {instance_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_pattern', type=str, default='swe-bench-lite-function_*')
    parser.add_argument("--model", type=str, default="lightonai/Reason-ModernColBERT", help="Model to use")
    parser.add_argument("--max_instances", type=int, default=300, help="Maximum number of instances to evaluate")
    parser.add_argument("--output_file", type=str, default="reason_colbert_results.json", help="Name of the output file to save results")
    
    args = parser.parse_args()
    
    # Find all matching dataset directories
    dataset_pattern = f"datasets/{args.dataset_pattern}"
    dataset_dirs = glob.glob(dataset_pattern)
    dataset_dirs.sort()
    
    if args.max_instances:
        dataset_dirs = dataset_dirs[:args.max_instances]
    
    logging.info(f"Found {len(dataset_dirs)} dataset instances to evaluate")
    
    # Load model
    logging.info(f"Loading model: {args.model}")
    try:
        model = ReasonColBERTModel(args.model)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return
    
    # Evaluate all instances
    all_results = {}
    all_detailed_results = {}
    failed_instances = []
    
    for i, dataset_dir in enumerate(dataset_dirs):
        instance_name = os.path.basename(dataset_dir)
        logging.info(f"[{i+1}/{len(dataset_dirs)}] Evaluating {instance_name}")
        
        instance_results, detailed_results = evaluate_single_instance(model, dataset_dir)
        
        if instance_results is not None:
            all_results[instance_name] = instance_results
            all_detailed_results[instance_name] = detailed_results
            logging.info(f"Completed {instance_name}: NDCG@1={instance_results['NDCG@1']:.4f}, Recall@5={instance_results['Recall@5']:.4f}")
        else:
            failed_instances.append(instance_name)
    
    # Calculate average results across all instances
    if all_results:
        k_values = [1, 3, 5, 10, 100, 1000]
        avg_results = {}
        
        for metric in [f"NDCG@{k}" for k in k_values] + [f"Recall@{k}" for k in k_values] + [f"P@{k}" for k in k_values] + [f"MRR@{k}" for k in k_values] + ["time"]:
            values = [results[metric] for results in all_results.values() if metric in results]
            avg_results[metric] = sum(values) / len(values) if values else 0
        
        logging.info("Average Results Across All Instances:")
        logging.info(f"NDCG@1: {avg_results['NDCG@1']:.4f}")
        logging.info(f"NDCG@5: {avg_results['NDCG@5']:.4f}")
        logging.info(f"Recall@5: {avg_results['Recall@5']:.4f}")
        logging.info(f"Recall@10: {avg_results['Recall@10']:.4f}")
        logging.info(f"Average time per query: {avg_results['time']:.2f}s")
        
        # Save results in the same format as eval_beir_sbert_canonical.py
        from datasets import load_dataset
        
        # Load the original dataset
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite")["test"]
        
        # Create a mapping from instance_id to retrieved docs
        instance_to_docs = {}
        
        # Debug: print what's in all_detailed_results
        logging.info(f"Debug: all_detailed_results keys: {list(all_detailed_results.keys())}")
        
        for dataset_dir in dataset_dirs:
            # Extract instance_id from directory name: "swe-bench-lite-function_astropy__astropy-12907" -> "astropy__astropy-12907"
            instance_name = os.path.basename(dataset_dir)
            parts = instance_name.split('-function_')
            if len(parts) == 2:
                instance_id = parts[1]
            else:
                # Fallback: try to extract from the end
                instance_id = instance_name.split('_', 2)[-1]
            
            logging.info(f"Debug: Processing dataset_dir={dataset_dir}, instance_name={instance_name}, instance_id={instance_id}")
            
            # The all_detailed_results is keyed by the instance_name (basename of dataset_dir)
            # and contains {query_id: {doc_id: score}}
            if instance_name in all_detailed_results:
                query_results = all_detailed_results[instance_name]
                logging.info(f"Debug: Found results for {instance_name}, query_results keys: {list(query_results.keys())}")
                
                # For each query in this instance (usually just one)
                retrieved_docs = []
                for query_id, doc_scores in query_results.items():
                    logging.info(f"Debug: Processing query_id={query_id}, num_docs={len(doc_scores)}")
                    # Sort documents by score (descending)
                    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
                    retrieved_docs = [doc_id for doc_id, score in sorted_docs]
                    logging.info(f"Debug: Top 3 docs: {retrieved_docs[:3]}")
                    break  # Usually only one query per instance
                
                instance_to_docs[instance_id] = retrieved_docs
            else:
                logging.info(f"Debug: No results found for {instance_name}")
                instance_to_docs[instance_id] = []
        
        # Create docs column for all instances in the dataset
        all_top_docs = []
        for instance in dataset:
            instance_id = instance["instance_id"]
            if instance_id in instance_to_docs:
                all_top_docs.append(instance_to_docs[instance_id])
            else:
                all_top_docs.append([])  # Empty list for instances not evaluated
        
        # Add docs column to dataset and save
        dataset = dataset.add_column("docs", all_top_docs)
        
        output_dir = "./results/"
        os.makedirs(output_dir, exist_ok=True)
        model_name = args.model.split("/")[-1] if "/" in args.model else args.model
        
        # Save in the same format as eval_beir_sbert_canonical.py
        results_file = f"{output_dir}/model={model_name}_dataset=swe-bench-lite_split=test_level=function_evalmode=fixed_results.json"
        dataset.to_json(results_file)
        
        # Save summary results
        summary_results = {
            "model": args.model,
            "total_instances": len(dataset_dirs),
            "successful_instances": len(all_results),
            "failed_instances": len(failed_instances),
            "average_results": avg_results
        }
        
        output_file = f"{output_dir}/model={model_name}_dataset=swe-bench-lite_split=test_level=function_evalmode=fixed_output.json"
        with open(output_file, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        logging.info(f"Results saved to {results_file}")
        logging.info(f"Summary saved to {output_file}")
        
        if failed_instances:
            logging.warning(f"Failed to process {len(failed_instances)} instances: {failed_instances}")
    
    else:
        logging.error("No instances were successfully evaluated!")

if __name__ == "__main__":
    main()
