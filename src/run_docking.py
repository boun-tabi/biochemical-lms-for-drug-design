from evaluation.docking import Docking
from argparse import ArgumentParser

structure_mapping = {
    'P00918': {
        'receptor': '3KS3', #'5EHW',
        'ligand': 'GOL'  # '5OO'
    },
    'Q16790': {
        'receptor': '6QN5',
        'ligand': 'J8N'
    },
    'P00915': {
        'receptor': '2FOY',
        'ligand': 'B30'
    }

}

parser = ArgumentParser()
parser.add_argument('--uniprot_id', type=str)
parser.add_argument('--target_pdb_id', type=str)
parser.add_argument('--ligand_id', type=str)
parser.add_argument('--mol_file', type=str)
parser.add_argument('--cpu', default=10, type=int)
args = parser.parse_args()
if args.uniprot_id:
    target_pdb_id = structure_mapping[args.uniprot_id]['receptor']
    ligand_id = structure_mapping[args.uniprot_id]['ligand']
else:
    target_pdb_id = args.target_pdb_id
    ligand_id = args.ligand_id
docking = Docking(target_pdb_id, ligand_id, args.mol_file, args.cpu)

