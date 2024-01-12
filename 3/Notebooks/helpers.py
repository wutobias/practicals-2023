def ad4v_dock(df, receptor_pdbqt, center, box_size=[30, 30, 30], SEED=42):

    """
    Run Autodock-vina from molecules stored in a dataframe

    Parameters
    ----------
    df : pandas.Dataframe
        data with query ligands stored in column "ROMol".
        name of the ligands must be stored in column "molecule_chembl_id"
    receptor_pdbqt : str
        Path to receptor pdbqt file
    center : list
        list with xyz coordinates of the center of the docking box
    box_size : list
        box size (edge length) in xyz directions
    SEED : int
        random seed

    Returns
    -------
    None
    """

    import os
    import subprocess
    env = os.environ.copy()

    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem import SaltRemover
    remover = SaltRemover.SaltRemover()
    for mol_idx, row in df.iterrows():
        name  = row["molecule_chembl_id"]

        if not os.path.exists(f'{name}_dock.pdbqt'):
            print(f"Docking {name}")
            rdmol = row["ROMol"]
            if Descriptors.MolWt(rdmol) > 600:
                continue

            res = remover.StripMol(rdmol,dontRemoveEverything=True)
            res.SetProp("_Name", name)
            smi   = Chem.MolToSmiles(res)

            cmd = f"obabel -:\"{smi}\" -osdf -O {name}.sdf -p 7 --gen3d"
            subprocess.run(cmd, shell=True, env=env)
            cmd = f"obabel -isdf {name}.sdf -opdbqt -O {name}.pdbqt -xhnbs -c"
            subprocess.run(cmd, shell=True, env=env)

            cmd = f"vina --receptor {receptor_pdbqt} --ligand {name}.pdbqt \
--scoring vina --center_x {center[0]} --center_y {center[1]} --center_z {center[2]} \
--size_x {box_size[0]} --size_y {box_size[1]} --size_z {box_size[2]} \
--seed {SEED} --out {name}_dock.pdbqt --exhaustiveness 32"
            subprocess.run(cmd, shell=True, env=env)

    return None


def read_pdbqt(df):

    """
    Run Autodock-vina from molecules stored in a dataframe

    Parameters
    ----------
    df : pandas.Dataframe
        data with query ligands stored in column "ROMol".
        name of the ligands must be stored in column "molecule_chembl_id"

    Returns
    -------
    None
    """

    import os

    for mol_idx, row in df.iterrows():
        name  = row["molecule_chembl_id"]
        if os.path.exists(f'{name}_dock.pdbqt'):
            with open(f'{name}_dock.pdbqt', 'r') as fopen:
                counts = 0
                for line in fopen:
                    line = line.rstrip().lstrip()
                    if line.startswith("REMARK VINA RESULT"):
                        line = line.split()
                        if f"score-{counts}" not in df.columns:
                            df.insert(
                                df.shape[1], f"score-{counts}", 
                                [None for _ in range(df.shape[0])])
                        df.loc[mol_idx, f"score-{counts}"] = float(line[3])
                        counts += 1

    return df


def combine_pdbqt(df, receptor_pdbqt):

    """
    Save combined pdb of receptor and ligand for
    docked structure.

    Parameters
    ----------
    df : pandas.Dataframe
        data with query ligands
    receptor_pdbqt : str
        path to receptor pdbqt

    Returns
    -------
    None
    """
    
    from pymol import cmd
    import os
    cmd.load(receptor_pdbqt)
    for mol_idx, row in df.iterrows():
        name  = row["molecule_chembl_id"]
        if os.path.exists(f'{name}_dock.pdbqt'):
            cmd.load(f'{name}_dock.pdbqt')
            #cmd.select("sele", f"protein or {name}_dock and state 1")
            #cmd.save(f'sele', state=1)
            cmd.save(f"{name}_dock.pdb")
            cmd.delete(f'{name}_dock')
            
    return None


def ClusterFps(fps,cutoff=0.2):
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina
    import math

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])
        #dists.extend([math.log2(x) for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
    return cs
