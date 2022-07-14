import pandas as pd
import json
import random
from tqdm import tqdm
from ..evaluation import calc_test_metrics

train = pd.read_csv('data/processed/pfam_diverse/train_interactions.csv', index_col=False)
val = pd.read_csv('data/processed/pfam_diverse/val_interactions.csv', index_col=False)
all_interactions = pd.concat([train, val])

targeted_interactions = all_interactions.groupby('UniProt_S_ID')['canonical_SMILES'].apply(list).to_dict()
metrics = {}
for target, designed_molecules in tqdm(targeted_interactions.items()):
    random.shuffle(designed_molecules)
    if len(designed_molecules) <= 40:
        split1 = designed_molecules[:len(designed_molecules)//2]
        split2 = designed_molecules[len(designed_molecules)//2:]
    else:
        split1 = designed_molecules[:20]
        split2 = designed_molecules[20:]
    metrics[target] = calc_test_metrics(split1, split2)
    
with open('data/pfam_diverse/split_similarities.json', 'r') as f:
    f.write(json.dumps(metrics))    

