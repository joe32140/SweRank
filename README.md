# SweRank

## Setup

Install the required dependencies:
```
conda create --name swerank python=3.11.9
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src"
```

## SweRank Paper Reproduction

### Downloading the Evaluation Dataset

The processed versions of `SWE-Bench-Lite` and `LocBench` datasets are uploaded [here](). You will need to download them locally and unzip to get the `datasets` folder. 

### SweRankEmbed Evaluation

To run the retriever on `SWE-Bench-Lite`:
```
bash script/run_retriever.sh Salesforce/SweRankLLM-Small SweRankLLM-Small <path to datasets folder> swe-bench-lite
```

To run the retriever on `LocBench`:
```
bash script/run_retriever.sh Salesforce/SweRankLLM-Small SweRankLLM-Small <path to datasets folder> loc-bench
```

### SweRankLLM Evaluation
We use a single json file with all the retriever results as the input for reranking run. To make a reranking run on `SweRankEmbed-Small` outputs, run:

```
bash script/run_rerank.sh SweRankEmbed-Small Salesforce/SweRankLLM-Small swe-bench-lite
```
To evaluate on `LocBench`
```
bash script/run_rerank.sh SweRankEmbed-Small Salesforce/SweRankLLM-Small loc-bench
```

To make a reranking run with GPT models, 
```
bash script/run_rerank_gpt.sh ${retriever_name} ${dataset_name}
```

## SweLoc Dara Creation

The training data is collected from github issues and PRs in popular public python repositories. The code is heavily based on [SWE-Bench](https://github.com/SWE-bench/SWE-bench). 

### Setup

You should first get your github access token from [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens), and set it as follows:
```
export GITHUB_TOKEN=<your github token here>
```

### Collecting Data

To run the data collection:
```
cd src/collect
bash collect_data.sh
```

### Negative Mining

After collecting the GitHub issue and corresponding modified functions (from the human edit patches), we extract the other functions in the repositories and mine the hard negatives to create the contrastive data:
```
cd ..
bash negative_mining.sh
```
The above script performs quality filtering and saves the dataset to `repo_contrastive_mined_filtered.jsonl`.

This output file contains the contrastive data which can be used for retriever training. 

### Reranker Data Construction

To generate reranker training data with the filtered contrastive mined dataset, run
```
bash get_reranker_data.sh
```

## Reranker Training

For single node training, run
```
bash script/run_train_reranker_single_node.sh
```
Notice that you have to specify the number of gpus to be used for training.

For multi node training, run
```
bash script/run_train_reranker_multi_node.sh
```
You have to set your distributed arguments for multi node training within the above script.
We have a sample script to run multinode training in slurm cluster at `script/slurm_run_multinode_train.sh`
Make sure to load required modules within the example script for the run as well.