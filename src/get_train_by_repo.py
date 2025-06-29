import multiprocessing
from get_repo_structure.get_repo_structure import get_project_structure_from_scratch
from get_repo_structure.get_patch_info import *
from datasets import load_dataset, Dataset
import srsly
import os
import argparse
from tqdm import tqdm


def load_structure(instance):
    try:
        structure = get_project_structure_from_scratch(instance['repo'], instance['base_commit'], 
                                                    instance['instance_id'], 'playground')
    except Exception as e:
        print("Error: ", str(e), " while scraping repo: ", instance['repo'])
        return None 
    
    instance['structure'] = structure 
    return instance


def process_instance(instance):
    query = instance['problem_statement']
    
    structure = instance['structure']
    
    try:
        data = find_py_or_non_dict_with_path(structure['structure'], cond=instance["instance_id"].startswith('pytest-dev__'))
        patch_info = parse_patch_full(instance['model_patch'], structure)
    except Exception as e:
        print("Error: ", str(e), " while processing instance: ", instance["instance_id"])
        return []

    changed_funcs = set()
    for fle, hunks in patch_info.items():
        for hunk in hunks:
            if hunk['function_changed'] and not hunk['newly_added']:
                if hunk['class_changed']:
                    changed_funcs.add(f'{fle}/{hunk["class_changed"]}/{hunk["function_changed"]}')
                else:
                    changed_funcs.add(f'{fle}/{hunk["function_changed"]}')
    
    neg_ids, neg_content = [], []
    pos_ids, pos_content = [], []
    for func, content in data.items():
        if func not in changed_funcs:
            neg_ids.append(func)
            neg_content.append(content)
        else:
            pos_ids.append(func)
            pos_content.append(content)
    
    instances = []
    for i in range(len(pos_ids)):
        instances.append(dict(query=query, positive_id=pos_ids[i], positive_code=pos_content[i], negative_ids=neg_ids, 
                              negative_codes=neg_content, repo=instance['repo'], 
                              base_commit=instance['base_commit'], instance_id=instance['instance_id']))
    
    return instances

def main(args):
    dataset = [d for d in tqdm(srsly.read_jsonl(args.data_path))]
    
    with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
        instances_with_codebase = list(tqdm(pool.imap_unordered(load_structure, dataset), total=len(dataset)))
    
    instances_with_codebase = [inst for inst in instances_with_codebase if inst is not None]
    
    srsly.write_jsonl(args.out_path, instances_with_codebase)  

    with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
        results = list(tqdm(pool.imap_unordered(process_instance, instances_with_codebase), total=len(instances_with_codebase)))
    
    # Flatten the list of lists
    ds = [item for sublist in results for item in sublist]
    ds = Dataset.from_list(ds)
    ds.to_json(args.save_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, default="prs_with_code.jsonl")
    parser.add_argument('--save_file_name', type=str, default="repo_contrastive_premined.jsonl")

    args = parser.parse_args()
    main(args)