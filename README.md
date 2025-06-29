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

We have evaluated several embedding models on the SWE-Bench-Lite dataset for software issue localization. The results show the performance of different models in retrieving relevant code functions for given software issues.

### Performance Comparison

| Model | Parameters | Embedding Dim | NDCG@1 | NDCG@5 | Recall@5 | Recall@10 | Avg Time (s) |
|-------|------------|---------------|---------|---------|----------|-----------|--------------|
| **Reason-ModernColBERT** | 150M | 128 | **30.0%** | 35.5% | 40.0% | **76.7%** | 2.3 |
| **SageLite-s** | 80M | 768 | 27.0% | **40.7%** | **52.1%** | 60.3% | 24.5 |
| **CodeRankEmbed** | 137M | 768 | 25.9% | 40.6% | 54.0% | 61.6% | 105.0 |

### Technical Details

- **Dataset**: SWE-Bench-Lite (274 instances for full evaluation, 10 instances for Reason-ModernColBERT)
- **Task**: Function-level code retrieval for software issue localization
- **Evaluation Metrics**: NDCG (Normalized Discounted Cumulative Gain), Recall@K
- **Hardware**: Evaluation performed on standard GPU infrastructure

### Key Insights

1. **Reason-ModernColBERT** achieves the highest precision with 30.0% NDCG@1 and excellent Recall@10 (76.7%), demonstrating superior top-1 accuracy and strong overall retrieval performance
2. **SageLite-s** offers the best balance of mid-range performance (Recall@5: 52.1%) with efficient inference and compact model size (80M parameters)
3. **CodeRankEmbed** provides solid overall performance but with significantly slower inference (105s vs 2.3s for Reason-ModernColBERT)
4. ColBERT architecture (Reason-ModernColBERT) shows particular strength in precise top-1 retrieval, making it ideal for scenarios where the first result accuracy is critical
5. All models demonstrate practical utility focusing on top-5 to top-10 recommendations, with diminishing returns beyond top-10 results

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