#!/bin/bash

set -e

export REPO_DIR=$PWD
export DATASET_DIR="${REPO_DIR}/datasets"

# Files to save a list of PyPI packages and valid repos
pypi_list_file="pypi_rankings.jsonl"
valid_repo_out_path="valid_top_pypi_gitrepos.jsonl"

# Path to save PRs and tasks from valid repos
prs_path=prs
tasks_path=tasks

# Max number of repos to scrape and process. Increase this number to scrape more repos.
max_num_repos=6

# Scrape PyPI packages
# python get_top_pypi.py \
#     --max-repos ${max_num_repos}

# # Retrieve source GitHub repos of the scraped PyPI packages
# python get_valid_urls.py \
#     --repo_info_path ${pypi_list_file} \
#     --out_path ${valid_repo_out_path}

# Scrape PRs and create tasks
python generate_tasks.py \
    --repo_info_path ${valid_repo_out_path} \
    --prs_path ${prs_path} \
    --tasks_path ${tasks_path} \
    --git_token ${GITHUB_TOKEN} \
    --start_idx 0 \
    --num_repos_to_process ${max_num_repos}

# Build dataset with scraped PRs
python build_dataset_ft.py \
    --instances_path ${tasks_path} \
    --output_dir ${DATASET_DIR} \
    --seed 42 
