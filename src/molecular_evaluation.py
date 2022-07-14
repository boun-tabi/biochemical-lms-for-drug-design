import moses
import json
import pandas as pd
import logging
from pathlib import Path
from argparse import ArgumentParser


def read_json(filename):
    return json.loads(open(filename, 'r').read())


def dump_json(data, filename):
    json.dump(data, open(filename, 'w'))


def evaluate_model_generations(folder, train, targeted_interactions):
    output_folder = Path(folder)
    outputs = read_json(output_folder / 'ChemBERTaLM.json')
    moses_train = moses.get_dataset('train').tolist()
    results = {}
    scores_dir = Path('results') / output_folder.parent.name / output_folder.name
    scores_dir.mkdir(parents=True, exist_ok=True)
    results['metrics_moses'] = moses.get_all_metrics(sum(outputs['predictions'].values(),[]), train=(train+moses_train))

    results['targets'] = {}
    for uniprot_id, smiles in outputs['predictions'].items():
        logging.info('Targeting %s' % uniprot_id)
        try:
            results['targets'][uniprot_id] = {
                'bdb': moses.get_all_metrics(smiles, k=1, train=(train+moses_train), test=targeted_interactions[uniprot_id]),
                'moses': moses.get_all_metrics(smiles, k=1)
            }
        except:
            results['error'].append(uniprot_id)
    dump_json(results, output_folder / 'ChemBERTaLM_scores.json')


parser = ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--train', type=str, default='data/splits/train_interactions.csv')
parser.add_argument('--test', type=str, default='data/splits/test_interactions.csv')
args = parser.parse_args()
train = pd.read_csv(args.train)['canonical_SMILES'].values.tolist()
test_interactions = pd.read_csv(args.test, index_col=None)
targeted_interactions = test_interactions.groupby('UniProt_S_ID')['canonical_SMILES'].apply(list).to_dict()
evaluate_model_generations('predictions/' + args.model, train, targeted_interactions)
