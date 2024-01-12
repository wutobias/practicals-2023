#!/usr/bin/env python3

import pandas as pd
output_df = pd.read_csv("./binders.csv")

from rdkit.Chem import Descriptors, Draw, PandasTools
molecules = pd.DataFrame(
    {"molecule_chembl_id"   : output_df["molecule_chembl_id"], 
     "smiles" : output_df["smiles"],
     "pIC50"  : output_df["pIC50"],
     "active" : [1 for _ in range(output_df.shape[0])]
    })
smiles_list = list()
import glob
for path in glob.glob("dude-decoys/decoys/decoys.*.picked"):
    with open(path, "r") as fopen:
        for line in fopen:
            line = line.replace("ligand", "")
            c = line.rstrip().lstrip().split()
            smiles_list.append(c[0])
for idx, smi in enumerate(smiles_list[:100]):
    Nrows = molecules.shape[0]
    molecules.loc[Nrows] = [f"Decoy-{idx}", smi, 99999., 0]
PandasTools.RenderImagesInAllDataFrames(images=True)
PandasTools.AddMoleculeColumnToFrame(molecules, "smiles", includeFingerprints=True)

from helpers import ad4v_dock
ad4v_dock(molecules, "3ovv_prot.pdbqt", [ -7.731, -8.501, 19.163])
