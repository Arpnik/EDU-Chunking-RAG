import json
from typing import List, Dict, Set

"""Utility functions for loading and validating JSONL files."""
def load_jsonl(filepath: str, max_lines: int = None) -> List[Dict]:
    """Load JSONL file."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                data.append(json.loads(line.strip()))
        print(f"✓ Loaded {len(data)} records from {filepath}")
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
    return data

BASE_DIR = "../../../../dataset/reduced_fever_data/"
data = load_jsonl(BASE_DIR + "paper_dev.jsonl",None)
data.extend(load_jsonl(BASE_DIR + "paper_test.jsonl",None))
data.extend(load_jsonl(BASE_DIR + "train.jsonl",None))
count=set()
count2=0
for entry in data:
    if entry["verifiable"]=="VERIFIABLE":
        count2+=1
        for evidence in entry["evidence"]:
            for ev in evidence:
                count.add((ev[2]))
print(len(count))
print(count2)
print(count.pop())