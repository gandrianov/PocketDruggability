import pytest

import glob
import numpy as np

from biopandas.pdb import PandasPdb
from PocketDruggability.utils import remove_hydrogens
from PocketDruggability.utils import extract_ligand
from PocketDruggability.utils import get_pocket
from PocketDruggability.cmds import featurize_pocket


complexes = glob.glob("tests/Complexes/*pdb")

apo = [c.replace("/Complexes/", "/Apo/").replace("_Complex", "") for c in complexes]
ligands = [c.replace("/Complexes/", "/Ligands/").replace("_Complex", "_ligand") for c in complexes]
pockets = [c.replace("/Complexes/", "/Pocket/").replace("_Complex", "_prox4_pock") for c in complexes]
reports = [c.replace("/Complexes/", "/Reports/Features_").replace("_Complex.pdb", ".rpt") for c in complexes]

@pytest.mark.parametrize("complex_fname, expected_fname", zip(complexes, apo))
def test_apo(complex_fname, expected_fname):
    
    test_apo = PandasPdb().read_pdb(complex_fname)
    test_apo = remove_hydrogens(test_apo.df["ATOM"])
    test_apo = test_apo.drop("line_idx", axis=1)

    expected_apo = PandasPdb().read_pdb(expected_fname)
    expected_apo = expected_apo.df["ATOM"].drop("line_idx", axis=1)
    
    assert test_apo.equals(expected_apo)

@pytest.mark.parametrize("complex_fname, expected_fname", zip(complexes, ligands))
def test_ligand(complex_fname, expected_fname):
    
    lig_name = "TMP"
    test_ligand = PandasPdb().read_pdb(complex_fname)
    test_ligand = remove_hydrogens(test_ligand.df["HETATM"])
    test_ligand  = extract_ligand(test_ligand, lig_name)
    test_ligand = test_ligand.drop("line_idx", axis=1).reset_index(drop=True)

    expected_ligand = PandasPdb().read_pdb(expected_fname)
    expected_ligand = expected_ligand.df["HETATM"].drop("line_idx", axis=1)

    assert test_ligand.equals(expected_ligand)


@pytest.mark.parametrize("complex_fname, expected_fname", zip(complexes, pockets))
def test_pocket(complex_fname, expected_fname):

    lig_name = "TMP"
    interface_cutoff = 4.0000001

    test_pdb = PandasPdb().read_pdb(complex_fname)

    test_apo = remove_hydrogens(test_pdb.df["ATOM"])

    test_ligand = remove_hydrogens(test_pdb.df["HETATM"])
    test_ligand = extract_ligand(test_ligand, lig_name)

    test_pocket = get_pocket(test_ligand, test_apo, interface_cutoff)
    test_pocket = test_pocket.drop("line_idx", axis=1).reset_index(drop=True)

    expected_pocket = PandasPdb().read_pdb(expected_fname)
    expected_pocket = expected_pocket.df["ATOM"].drop("line_idx", axis=1)

    assert test_pocket.equals(expected_pocket)

@pytest.mark.parametrize("complex_fname, expected_fname", zip(complexes, reports))
def test_reports(complex_fname, expected_fname):
    
    lig_name = "TMP"
    interface_cutoff = 4.0
    test_features = featurize_pocket(complex_fname, lig_name, interface_cutoff)
    test_features["PDBid"] = test_features["PDBid"].replace("_Complex", "")

    with open(expected_fname, "r") as fr:
        fr = fr.read().split("\n")[:-1]
        features = fr[-1].split(" ")[1:-1]

    expected_features = {}
    expected_features["PDBid"] = fr[0].split(" ")[0]
    expected_features["Seq"] = fr[1].split(": ")[-1]
    expected_features["C_RESIDUE"] = float(features[0])
    expected_features["INERTIA_3"]     = float(features[1])
    expected_features["SMALLEST_SIZE"] = float(features[2])
    expected_features["SURFACE_HULL"]  = float(features[3])
    expected_features["VOLUME_HULL"]   = float(features[4])
    expected_features["hydrophobic_kyte"] = float(features[5])
    expected_features["hydrophobicity_pocket"] = float(features[6])
    expected_features["p_Ccoo"]   = float(features[7])
    expected_features["p_N_atom"] = float(features[8])
    expected_features["p_Ooh"]    = float(features[9])
    expected_features["p_aliphatic_residue"] = float(features[10])
    expected_features["p_aromatic_residue"]  = float(features[11])
    expected_features["p_negative_residue"]  = float(features[12])

    for feature_name, test_value in test_features.items():
        expected_value = expected_features[feature_name]

        if isinstance(test_value, str):
            if test_value != expected_value:
                print(feature_name, test_value, expected_value)
                assert False
        else:
             if not np.isclose(test_value, expected_value, atol=0.01):
                print(feature_name, test_value, expected_value)
                assert False