#!/usr/bin/env bash

# If you'd like to parallelize, do the following:
# * Create a .env file in this folder
# * Declare GITHUB_TOKENS=token1,token2,token3...

python3 get_tasks_pipeline.py \
    --repos 'scikit-learn/scikit-learn', 'pallets/flask' \
    --path_prs '/Users/user/Desktop/jdoo/SWE-bench/prs' \
    --path_tasks '/Users/user/Desktop/jdoo/SWE-bench/tasks'