import json
import os
import random
from adalflow.datasets.types import Example

def load_bbh_object_count(n=40, split_ratio=(0.6, 0.2, 0.2)):
    """
    Loads BBH data and splits it into Train/Val/Test for AdalFlow Trainer.
    """
    # 1. Load Data (Mocking BBH loading for simplicity)
    # In a real scenario, you would download the JSON.
    # Here we generate synthetic "hard" cases to test the optimizer.
    
    data = []
    
    # Mix of patterns to prevent overfitting
    base_templates = [
        ("I have {n1} chairs, {n2} tables, and {n3} lamps.", lambda n1,n2,n3: n1+n2+n3),
        ("I have {n1} pigs, {n2} cows, and a chicken.", lambda n1,n2,n3: n1+n2+1),
        ("I see {n1} violins and {n2} flutes.", lambda n1,n2,n3: n1+n2), # Musical trap
        ("I bought {n1} apples, {n2} oranges, {n3} pears, and a watermelon.", lambda n1,n2,n3: n1+n2+n3+1),
    ]
    
    for _ in range(n):
        t, func = random.choice(base_templates)
        nums = [random.randint(1, 5) for _ in range(3)]
        q = t.format(n1=nums[0], n2=nums[1], n3=nums[2])
        ans = str(func(*nums))
        
        # AdalFlow uses 'Example' objects
        ex = Example(id=str(uuid.uuid4()), question=q, answer=ans)
        data.append(ex)

    # 2. Split
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    
    trainset = data[:n_train]
    valset = data[n_train:n_train+n_val]
    testset = data[n_train+n_val:]
    
    print(f"📚 Data Loaded: Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}")
    return trainset, valset, testset

import uuid