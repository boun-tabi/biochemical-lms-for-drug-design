import logging

from subprocess import check_output, CalledProcessError
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools


class Docking:

    def __init__(self, target_pdb_id, ligand_id, molecule_path, num_cpu):
        self.ligands_path = 'docking/ligands/sdf'
        self.targets_path = 'docking/targets'
        self.complex_path = 'docking/complex'
        self.prepare_receptor(target_pdb_id, ligand_id)
        self.perform_docking(receptor_path=f'{self.targets_path}/{target_pdb_id}-receptor.pdbqt',
                             ligand_path=f'{self.ligands_path}/{molecule_path}',
                             output_path=f'{self.complex_path}/{target_pdb_id}_{molecule_path}.gz',
                             ligand_from_protein=f'{str(self.targets_path)}/{target_pdb_id}-{ligand_id}.pdbqt',
                             num_cpu=num_cpu)

    def prepare_ligands(self, smiles_list):
        inputMols = [Chem.MolFromSmiles(x) for x in smiles_list]
        inputMols = [m for m in inputMols if m is not None]
        for ix, m in enumerate(inputMols):
            m.SetProp("_Name", str(ix))
        for i, mol in enumerate(inputMols):
            if mol is None:
                logging.warning('Failed to convert molecule %s' % (i))
            if not mol.GetProp('_Name'):
                logging.warning('No name for molecule %s' % (i))
        return inputMols

    def ligands_to_sdf(self, mols, confoutputFilePath):
        writer = Chem.SDWriter(confoutputFilePath)

        for ix, mol in enumerate(mols):
            if mol:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
                mol.SetProp('_Name', Chem.MolToSmiles(mol))
                writer.write(mol)
        writer.close()

    def perform_docking(self, receptor_path, ligand_path, output_path, ligand_from_protein, num_cpu=1):
        command = ['./gnina',
                   '--cpu', str(num_cpu),
                   '--seed', '0',
                   '--autobox_ligand', ligand_from_protein,
                   '-r', receptor_path,
                   '-l', ligand_path,
                   '-o', output_path]
        try:
            stream = check_output(command, universal_newlines=True)
            docked_df = PandasTools.LoadSDF(output_path, molColName='Molecule')
            docked_df.groupby('ID')['minimizedAffinity'].min().reset_index().to_csv(
                output_path.replace('.sdf.gz', 'csv'))

        except CalledProcessError:
            logging.warning('Docking failed %s %s' % (receptor_path, ligand_path))

    def prepare_receptor(self, target, ligand):
        logging.info('Downloading target %s ' % target)
        command = ['wget',
                   '-O',
                   f'{str(self.targets_path)}/{target}.pdb',
                   f'https://files.rcsb.org/download/{target}.pdb']

        self.run_command(command)

        com_file = open('fetch_and_clean.pml', 'w')
        com_file.write(f'''
        load {str(self.targets_path)}/{target}.pdb
        remove resn HOH
        h_add elem O or elem N
        select {target}-{ligand}, resn {ligand} 
        select {target}-receptor,  {target} and not {target}-{ligand} #Select all that is not the ligand
        save {str(self.targets_path)}/{target}-{ligand}.pdb,  {target}-{ligand}
        save {str(self.targets_path)}/{target}-receptor.pdb,  {target}-receptor
        ''')
        com_file.close()
        command = ['pymol',
                   '-c',
                   'fetch_and_clean.pml']

        self.run_command(command)

        command = ['obabel',
                   f'{str(self.targets_path)}/{target}-receptor.pdb',
                   '-xr',
                   '-O',
                   f'{str(self.targets_path)}/{target}-receptor.pdbqt']

        self.run_command(command)

        command = ['obabel', f'{str(self.targets_path)}/{target}-{ligand}.pdb',
                   '-r', '-O', f'{str(self.targets_path)}/{target}-{ligand}.pdbqt']
        self.run_command(command)

    def run_command(self, command):
        try:
            return check_output(command)
        except CalledProcessError:
            logging.warning('Failed %s' % command)
