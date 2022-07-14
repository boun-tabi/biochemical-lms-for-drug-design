import json
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from evaluation import calc_test_metrics
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)


def read_json(filename):
    return json.loads(open(filename, 'r').read())


def dump_json(data, filename):
    json.dump(data, open(filename, 'w'))


def evaluate_predictions(generations, references, train, n_jobs=1):
    targeted_interactions = references.groupby('UniProt_S_ID')['canonical_SMILES'].apply(list).to_dict()

    overall_metrics, target_metrics = [], []
    for strategy, predictions in generations.items():
        logging.info(f'Strategy: {strategy}')
        scores, molecules = {}, []
        for target, designed_molecules in tqdm(predictions.items()):
            if type(designed_molecules) == list:
                target_metrics.append({'strategy': strategy, 'target': target,
                                       **calc_test_metrics(designed_molecules, targeted_interactions[target], train, n_jobs=n_jobs)})
                molecules.extend(designed_molecules)
            else:
                scores[target] = designed_molecules
        scores.update(calc_test_metrics(molecules, sum(targeted_interactions.values(), []), train))
        logging.info(f'{scores}')
        overall_metrics.append({'strategy': strategy, **scores})
    return pd.DataFrame(overall_metrics), pd.DataFrame(target_metrics)


parser = ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--train', type=str, required='data/splits/train_interactions.csv')
parser.add_argument('--test', type=str, default='data/splits/test_interactions.csv')
parser.add_argument('--val', type=str, default='data/splits/val_interactions.csv')

args = parser.parse_args()
logger.info(args)
train = pd.read_csv(args.train)['canonical_SMILES'].values.tolist()
val = pd.read_csv(args.val, index_col=None)
test = pd.read_csv(args.test, index_col=None)
output_folder = Path('predictions') / args.model
logger.info('Calculating metrics for test')
test_predictions = read_json(output_folder / 'test_predictions.json')
test_overall, test_target = evaluate_predictions(test_predictions, test, train, n_jobs=1)
test_overall.to_csv(output_folder / 'test_overall.csv')
test_target.to_csv(output_folder / 'test_targets.csv')
logger.info('Calculating metrics for val')
val_predictions = read_json(output_folder / 'val_predictions.json')
val_overall, val_target = evaluate_predictions(val_predictions, val, train)
val_overall.to_csv(output_folder / 'val_overall.csv')
val_target.to_csv(output_folder / 'val_targets.csv')
