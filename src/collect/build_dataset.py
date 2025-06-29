#!/usr/bin/env python3

"""
Convert PR data to task instances for fine-tuning
"""

import json
import os
import re
from typing import Dict, List, Optional
from tqdm import tqdm

from utils import Repo, extract_problem_statement_and_hints, extract_patches


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:100]}... Error: {e}")
                    continue
    return data


def extract_resolved_issues(pr: Dict) -> List[int]:
    """
    Extract issue numbers that this PR resolves from the PR body and title
    
    Args:
        pr (dict): PR dictionary from GitHub API
        
    Returns:
        list: List of issue numbers this PR resolves
    """
    resolved_issues = []
    
    # Keywords that indicate issue resolution
    keywords = {
        "close", "closes", "closed",
        "fix", "fixes", "fixed", 
        "resolve", "resolves", "resolved"
    }
    
    # Check PR title and body
    text_to_check = []
    if pr.get('title'):
        text_to_check.append(pr['title'])
    if pr.get('body'):
        text_to_check.append(pr['body'])
    
    for text in text_to_check:
        if not text:
            continue
            
        # Look for patterns like "fixes #123" or "closes #456"
        for keyword in keywords:
            # Pattern: keyword followed by optional whitespace, then #number
            pattern = rf'\b{keyword}\s*#(\d+)\b'
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                issue_num = int(match)
                if issue_num not in resolved_issues:
                    resolved_issues.append(issue_num)
    
    return resolved_issues


def is_valid_pr(pr: Dict) -> bool:
    """
    Check if PR is valid for task instance creation
    
    Args:
        pr (dict): PR dictionary
        
    Returns:
        bool: True if PR is valid for task creation
    """
    # Must be merged
    if pr.get('state') != 'closed' or not pr.get('merged_at'):
        return False
    
    # Must have a body with some content
    if not pr.get('body') or len(pr.get('body', '').strip()) < 10:
        return False
        
    # Skip bot PRs (like dependabot)
    author = pr.get('user', {}).get('login', '')
    if 'bot' in author.lower() or 'dependabot' in author.lower():
        return False
        
    # Must have some code changes (check if diff_url exists)
    if not pr.get('diff_url'):
        return False
        
    return True


def create_task_instance(pr: Dict, repo: Repo) -> Optional[Dict]:
    """
    Create a task instance from a PR
    
    Args:
        pr (dict): PR dictionary from GitHub API
        repo (Repo): Repository object
        
    Returns:
        dict: Task instance or None if creation failed
    """
    try:
        # Extract resolved issues
        resolved_issues = extract_resolved_issues(pr)
        pr['resolved_issues'] = resolved_issues
        
        # Extract problem statement and hints
        problem_statement, hints = extract_problem_statement_and_hints(pr, repo)
        
        # If no problem statement from issues, use PR title and body
        if not problem_statement.strip():
            title = pr.get('title', '')
            body = pr.get('body', '')
            problem_statement = f"{title}\n{body}" if title or body else ""
        
        # Extract patches
        try:
            model_patch, test_patch = extract_patches(pr, repo)
        except Exception as e:
            print(f"Warning: Failed to extract patches for PR #{pr.get('number')}: {e}")
            model_patch, test_patch = "", ""
        
        # Extract base commit SHA
        base_commit = pr.get('base', {}).get('sha', '')
        
        # Create task instance
        task_instance = {
            'instance_id': f"{repo.owner}_{repo.name}_{pr.get('number')}",
            'repo': f"{repo.owner}/{repo.name}",
            'pr_number': pr.get('number'),
            'pr_url': pr.get('html_url'),
            'problem_statement': problem_statement.strip(),
            'hints': hints.strip() if hints else "",
            'model_patch': model_patch,
            'test_patch': test_patch,
            'base_commit': base_commit,
            'created_at': pr.get('created_at'),
            'merged_at': pr.get('merged_at'),
            'resolved_issues': resolved_issues,
        }
        
        # Only return if we have meaningful content
        if (task_instance['problem_statement'] and 
            (task_instance['model_patch'] or task_instance['test_patch'])):
            return task_instance
            
    except Exception as e:
        print(f"Error creating task instance for PR #{pr.get('number')}: {e}")
        
    return None


def main(pr_file_path: str, task_file_path: str, github_token: str):
    """
    Convert PR data file to task instances
    
    Args:
        pr_file_path (str): Path to PR JSONL file
        task_file_path (str): Path to save task instances JSONL file
        github_token (str): GitHub API token
    """
    print(f"Loading PRs from {pr_file_path}")
    
    # Load PR data
    try:
        prs = load_jsonl(pr_file_path)
        print(f"Loaded {len(prs)} PRs")
    except Exception as e:
        print(f"Error loading PR file: {e}")
        return
    
    if not prs:
        print("No PRs found in file")
        return
    
    # Extract repo info from first PR
    first_pr = prs[0]
    repo_url = first_pr.get('url', '')
    if not repo_url:
        print("Error: Could not determine repository from PR data")
        return
        
    # Parse repo owner and name from URL
    # URL format: https://api.github.com/repos/owner/name/pulls/number
    try:
        parts = repo_url.split('/')
        owner = parts[4]  # owner
        name = parts[5]   # repo name
    except (IndexError, ValueError):
        print(f"Error: Could not parse repository from URL: {repo_url}")
        return
    
    print(f"Processing repository: {owner}/{name}")
    
    # Create repo object
    try:
        repo = Repo(owner, name, github_token)
    except Exception as e:
        print(f"Error creating repo object: {e}")
        return
    
    # Process PRs and create task instances
    task_instances = []
    valid_prs = [pr for pr in prs if is_valid_pr(pr)]
    
    print(f"Found {len(valid_prs)} valid PRs out of {len(prs)} total PRs")
    
    for pr in tqdm(valid_prs, desc="Processing PRs"):
        task_instance = create_task_instance(pr, repo)
        if task_instance:
            task_instances.append(task_instance)
    
    print(f"Created {len(task_instances)} task instances")
    
    # Save task instances
    if task_instances:
        os.makedirs(os.path.dirname(task_file_path), exist_ok=True)
        with open(task_file_path, 'w', encoding='utf-8') as f:
            for instance in task_instances:
                f.write(json.dumps(instance) + '\n')
        print(f"Saved task instances to {task_file_path}")
    else:
        print("No task instances created - creating empty file")
        os.makedirs(os.path.dirname(task_file_path), exist_ok=True)
        with open(task_file_path, 'w', encoding='utf-8') as f:
            pass  # Create empty file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_file_path", help="Path to PR JSONL file")
    parser.add_argument("task_file_path", help="Path to save task instances JSONL file")
    parser.add_argument("github_token", help="GitHub API token")
    
    args = parser.parse_args()
    main(args.pr_file_path, args.task_file_path, args.github_token)
