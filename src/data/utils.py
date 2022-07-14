import pandas as pd
from rdkit.Chem import SDWriter
from rdkit import Chem
from rdkit.Chem import AllChem
from ast import literal_eval


def convert_to_sdf(ligands, filename, novel=None):
    writer = SDWriter(str(filename))
    mols = [Chem.MolFromSmiles(smi) for smi in ligands]
    for ix, mol in enumerate(mols):
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
            mol.SetProp('_Name', Chem.MolToSmiles(mol))
            if novel:
                mol.SetProp('Novel', '0' if novel[ix] else '1')
            writer.write(mol)
    writer.close()


def read_files(folder, splits=['test', 'val']):
    files = {}
    for split in splits:
        files[f'{split}_overall'] = pd.read_csv(folder / f'{split}_overall.csv', index_col=False).assign(
            path=str(folder)).assign(split=split)
        files[f'{split}_target'] = pd.read_csv(folder / f'{split}_targets.csv', index_col=False).assign(
            path=str(folder)).assign(split=split)
        files[f'{split}_prediction'] = pd.read_json(folder / f'{split}_predictions.json').assign(
            path=str(folder)).assign(split=split)
    return files


def collect_results(path, splits=['test', 'val']):
    output_folders = [f.parent for f in path.glob("**/*") if 'test_overall.csv' == f.name]
    all_files = []
    for folder in output_folders:
        try:
            all_files.append(read_files(folder, splits))
        except:
            print(folder)
    frames = []
    for f_type in ['overall', 'target', 'prediction']:
        frames.append(pd.concat([item[f'{split}_{f_type}'] for item in all_files for split in splits]))

    return frames


def collect_predictions(path, filename):
    output_folders = [f.parent for f in path.glob("**/*") if f'{filename}.json' == f.name]
    all_files = []
    for folder in output_folders:
        try:
            all_files.append(pd.read_json(folder / f'{filename}.json').assign(path=str(folder)))
        except:
            print(folder)
    return pd.concat(all_files)


def literal_return(val):
    try:
        if not pd.isnull(val):
            return literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val


def collect_tuple_items(items):
    if type(items) == list:
        return [item[0] + ':' + item[1] for item in items if not pd.isnull(item[0])]
    else:
        return []


def format_proteins(df):
    df['clans'] = df['clans'].str.replace('(nan, nan)', '"-", "-"')
    for col in ['families', 'DrugBank', 'GO_Component', 'GO_Function', 'GO_Process', 'clans']:
        df[col] = df[col].apply(literal_return)
        df[f'{col}_ids'] = df[col].apply(collect_tuple_items)
    return df
