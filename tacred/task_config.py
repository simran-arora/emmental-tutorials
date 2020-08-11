TASK_NAMES = [
   "TACRED"
]

SPLIT_MAPPING = {
    "TACRED": {"train": "train.json", "dev": "dev.json", "test": "test.json"}
}

INDEX_MAPPING = {
    # each one contains three values:
    # sentence 1 index, sentence 2 index, label index, -1 means abstain
    "TACRED": {"train": [3, -1, 1], "dev": [3, -1, 1], "test": [1, -1, -1]}
}

SKIPPING_HEADER_MAPPING = {
    "TACRED": {"train": 0, "dev": 0, "test": 1}
}

LABEL_MAPPING = {
    "TACRED": {"1": 1, "0": 2}I’m 
}

METRIC_MAPPING = {
    "TACRED": ["accuracy_f1"]
}
