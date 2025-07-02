# SweRank

SweRank is a framework for software issue localization that uses a two-stage retrieve-and-rerank pipeline. It first retrieves relevant code snippets for a given issue and then reranks them to find the most likely functions to fix the issue.

Link to Paper: [https://arxiv.org/pdf/2505.07849](https://arxiv.org/pdf/2505.07849)

## Setup

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/gangiswag/SweRank.git
cd SweRank
conda create --name swerank python=3.11.9
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src"
```

## Reproducing Paper Results

Follow these steps to reproduce the evaluation results from our paper.

### 1. Download Evaluation Datasets

The processed versions of `SWE-Bench-Lite` and `LocBench` datasets are available for download [here](https://drive.google.com/file/d/1pjWkpEfFRAts5hkcCGFSoBW1HiQQTsTo/view?usp=share_link). You will need to download and unzip the file to get the `datasets` folder, which is required for the evaluation scripts.

### 2. Run Evaluation

#### SweRankEmbed Evaluation (Retrieval)

To run the `SweRankEmbed-Small` retriever on `SWE-Bench-Lite`:
```bash
bash script/run_retriever.sh Salesforce/SweRankEmbed-Small SweRankEmbed-Small <path_to_datasets_folder> swe-bench-lite
```

To run the retriever on `LocBench`:
```bash
bash script/run_retriever.sh Salesforce/SweRankEmbed-Small SweRankEmbed-Small <path_to_datasets_folder> loc-bench
```

To run the `SweRankEmbed-Large` retriever, replace the model name in the commands above:
```bash
bash script/run_retriever.sh Salesforce/SweRankEmbed-Large SweRankEmbed-Large <path_to_datasets_folder> swe-bench-lite
bash script/run_retriever.sh Salesforce/SweRankEmbed-Large SweRankEmbed-Large <path_to_datasets_folder> loc-bench
```

#### Additional Model Evaluations

##### SageLite Model Evaluation
To evaluate the SageLite-s model on SWE-Bench-Lite:
```bash
python src/eval_beir_sbert_canonical.py \
    --dataset swe-bench-lite \
    --split test \
    --level function \
    --model "Salesforce/SageLite-s" \
    --batch_size 16 \
    --output_file sagelite_results.json
```

##### Reason-ModernColBERT Evaluation
To evaluate the Reason-ModernColBERT model on SWE-Bench-Lite:
```bash
python src/eval_reason_colbert_fixed.py \
    --dataset_pattern "swe-bench-lite-function_*" \
    --model "lightonai/Reason-ModernColBERT" \
    --max_instances 10 \
    --output_file reason_colbert_results.json
```

For full evaluation (all 274 instances), remove the `--max_instances` parameter:
```bash
python src/eval_reason_colbert_fixed.py \
    --dataset_pattern "swe-bench-lite-function_*" \
    --model "lightonai/Reason-ModernColBERT" \
    --output_file reason_colbert_full_results.json
```

##### CodeRankEmbed Evaluation
To evaluate the CodeRankEmbed model on SWE-Bench-Lite:
```bash
python src/eval_beir_sbert_canonical.py \
    --dataset swe-bench-lite \
    --split test \
    --level function \
    --model "nomic-ai/CodeRankEmbed" \
    --batch_size 16 \
    --output_file coderank_results.json
```

#### SweRankLLM Evaluation (Reranking)

The reranking scripts use the JSON file with retriever results as input.

To run reranking on `SweRankEmbed-Small` outputs with the `SweRankLLM-Small` model on `SWE-Bench-Lite`:
```bash
bash script/run_rerank.sh SweRankEmbed-Small Salesforce/SweRankLLM-Small SweRankLLM-Small <path_to_datasets_folder> swe-bench-lite
```

To evaluate on `LocBench`:
```bash
bash script/run_rerank.sh SweRankEmbed-Small Salesforce/SweRankLLM-Small SweRankLLM-Small <path_to_datasets_folder> loc-bench
```

To perform reranking with GPT models:
```bash
bash script/run_rerank_gpt.sh ${retriever_name} ${dataset_name}
```
Here, `${retriever_name}` is the name of the retriever output (e.g., `SweRankEmbed-Small`), and `${dataset_name}` is the evaluation dataset (e.g., `swe-bench-lite`).

## Creating the SweLoc Training Dataset

The training data, which we call `SweLoc`, is collected from GitHub issues and pull requests from popular public Python repositories. The collection process is heavily based on [SWE-Bench](https://github.com/SWE-bench/SWE-bench).

### 1. Setup

Set your GitHub access token as an environment variable. You can generate a token from your [GitHub settings](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).
```bash
export GITHUB_TOKEN=your_github_token_here
```

### 2. Collect Raw Data

The first step is to collect GitHub issues and their corresponding human-written patches.
```bash
cd src/collect
bash collect_data.sh
```

### 3. Mine Negatives for Retriever Data

After collecting issues and the functions modified in patches, this step mines hard negatives from other functions in the repository. This creates a contrastive dataset for training the retriever model.

From the `src/collect` directory, run:
```bash
cd ..
bash negative_mining.sh
```
This script performs quality filtering and saves the retriever training data to `repo_contrastive_mined_filtered.jsonl`.

### 4. Construct Reranker Data

To generate training data for the reranker from the contrastive mined dataset, run the following from the `src` directory:
```bash
bash get_reranker_data.sh
```

## Training Models

### Retriever Training
The file `repo_contrastive_mined_filtered.jsonl` created in the data creation step can be used for contrastive training of a retriever model. The training script for the retriever is not included in this repository.

### Reranker Training

#### Single-Node Training
For training on a single machine with multiple GPUs, run:
```bash
bash script/run_train_reranker_single_node.sh
```
Note: You will need to specify the number of GPUs to be used inside the script.

#### Multi-Node Training
For distributed training across multiple nodes, run:
```bash
bash script/run_train_reranker_multi_node.sh
```
You will have to set your distributed training arguments within the script. We provide a sample script for a SLURM cluster at `script/slurm_run_multinode_train.sh`. Make sure to load any required modules within the example script as well.

## Evaluation Results

We have evaluated several embedding models on the SWE-Bench-Lite dataset for software issue localization across different granularity levels (file, module, function). The results demonstrate the performance of different models in retrieving relevant code components for given software issues.

### Model Comparison - Accuracy@k Results

#### FILE Level Accuracy@k
| Model | Acc@1 | Acc@3 | Acc@5 | Acc@10 | Acc@20 | Count |
|-------|-------|-------|-------|--------|--------|-------|
| **SageLite-s** | **54.4%** | **70.4%** | **78.1%** | **84.3%** | **89.4%** | 274 |
| **CodeRankEmbed** | 52.6% | 72.3% | 80.3% | 85.0% | 90.1% | 274 |
| **finetuned-Reason-ModernColBERT** | 51.8% | 66.4% | 74.5% | 78.1% | 83.2% | 274 |
| **Reason-ModernColBERT** | 41.6% | 62.0% | 69.3% | 77.7% | 82.8% | 274 |

#### MODULE Level Accuracy@k
| Model | Acc@1 | Acc@3 | Acc@5 | Acc@10 | Acc@20 | Count |
|-------|-------|-------|-------|--------|--------|-------|
| **SageLite-s** | **45.4%** | **66.8%** | **74.1%** | 78.0% | 83.9% | 205 |
| **CodeRankEmbed** | 42.0% | 66.8% | 74.6% | **79.5%** | **85.9%** | 205 |
| **finetuned-Reason-ModernColBERT** | 39.5% | 60.5% | 67.3% | 73.7% | 78.0% | 205 |
| **Reason-ModernColBERT** | 30.7% | 50.2% | 59.5% | 71.2% | 77.1% | 205 |

#### FUNCTION Level Accuracy@k
| Model | Acc@1 | Acc@3 | Acc@5 | Acc@10 | Acc@20 | Count |
|-------|-------|-------|-------|--------|--------|-------|
| **CodeRankEmbed** | **22.6%** | 42.7% | **52.6%** | **60.2%** | **69.0%** | 274 |
| **finetuned-Reason-ModernColBERT** | **22.6%** | 39.4% | 48.2% | 57.3% | 64.6% | 274 |
| **SageLite-s** | 22.3% | **43.8%** | 49.6% | 58.8% | 67.9% | 274 |
| **Reason-ModernColBERT** | 19.7% | 33.2% | 40.1% | 51.1% | 59.9% | 274 |

### Legacy Performance Comparison (NDCG/Recall Metrics)

| Model | Parameters | Embedding Dim | NDCG@1 | NDCG@5 | Recall@5 | Recall@10 | Avg Time (s) |
|-------|------------|---------------|---------|---------|----------|-----------|--------------|
| **Reason-ModernColBERT** | 150M | 128 | **30.0%** | 35.5% | 40.0% | **76.7%** | 2.3 |
| **SageLite-s** | 80M | 768 | 27.0% | **40.7%** | **52.1%** | 60.3% | 24.5 |
| **CodeRankEmbed** | 137M | 768 | 25.9% | 40.6% | 54.0% | 61.6% | 105.0 |

### Technical Details

- **Dataset**: SWE-Bench-Lite (274 instances for full evaluation, 10 instances for Reason-ModernColBERT)
- **Task**: Multi-level code retrieval for software issue localization (file, module, function)
- **Evaluation Metrics**: Accuracy@K, NDCG (Normalized Discounted Cumulative Gain), Recall@K
- **Hardware**: Evaluation performed on standard GPU infrastructure

### Key Insights

1. **File-level localization**: SageLite-s leads with 54.4% Acc@1, showing strong performance for coarse-grained issue localization
2. **Module-level localization**: SageLite-s maintains leadership at Acc@1 (45.4%), while CodeRankEmbed excels at higher k values
3. **Function-level localization**: CodeRankEmbed and finetuned-Reason-ModernColBERT tie for best Acc@1 (22.6%), demonstrating the challenge of fine-grained localization
4. **Performance hierarchy**: File > Module > Function accuracy, reflecting increasing difficulty with finer granularity
5. **Model specialization**: Different models excel at different granularities, suggesting complementary strengths for multi-stage retrieval pipelines

### Model Specifications

#### Reason-ModernColBERT
- 150M parameter ColBERT model based on ModernBERT architecture
- 128-dimensional embeddings with token-level interactions
- Trained on ReasonIR dataset for reasoning-focused retrieval
- Uses MaxSim scoring for fine-grained query-document matching
- Requires query prefix: "[Q] " and document prefix: "[D] "
- Supports up to 8192 tokens for documents, 128 for queries
- Optimized for precise top-1 retrieval in software issue localization

#### CodeRankEmbed
- 137M parameter bi-encoder model
- 768-dimensional embeddings
- Based on Arctic-Embed-M-Long architecture
- Supports 8192 context length
- Specialized for code ranking tasks
- Requires query prefix: "Represent this query for searching relevant code"
- Optimized for software issue localization

#### SageLite-s
- 80M parameter encoder model
- 768-dimensional embeddings
- Supports 15 programming languages
- No special query prefix requirements
- Efficient alternative for resource-constrained environments

### Model Specifications

#### Reason-ModernColBERT
- 150M parameter ColBERT model based on ModernBERT architecture
- 128-dimensional embeddings with token-level interactions
- Trained on ReasonIR dataset for reasoning-focused retrieval
- Uses MaxSim scoring for fine-grained query-document matching
- Requires query prefix: "[Q] " and document prefix: "[D] "
- Supports up to 8192 tokens for documents, 128 for queries
- Optimized for precise top-1 retrieval in software issue localization

#### CodeRankEmbed
- 137M parameter bi-encoder model
- 768-dimensional embeddings
- Based on Arctic-Embed-M-Long architecture
- Supports 8192 context length
- Specialized for code ranking tasks
- Requires query prefix: "Represent this query for searching relevant code"
- Optimized for software issue localization

#### SageLite-s
- 80M parameter encoder model
- 768-dimensional embeddings
- Supports 15 programming languages
- No special query prefix requirements
- Efficient alternative for resource-constrained environments

## Citation

If you find this work useful in your research, please consider citing our paper:
```
@article{reddy2025swerank,
  title={SweRank: Software Issue Localization with Code Ranking},
  author={Reddy, Revanth Gangi and Suresh, Tarun and Doo, JaeHyeok and Liu, Ye and Nguyen, Xuan Phi and Zhou, Yingbo and Yavuz, Semih and Xiong, Caiming and Ji, Heng and Joty, Shafiq},
  journal={arXiv preprint arXiv:2505.07849},
  year={2025}
}
```