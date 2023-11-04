"""
This module provides a suite of tests to validate the functionality of the PocketDruggability package.
It utilizes the pytest framework for testing different aspects of the system including the apo form,
ligand extraction, pocket detection, and report generation from protein structures.

The tests are designed to ensure that the implemented functions for processing PDB files and
extracting relevant features work as expected and are consistent with pre-calculated results.
"""

import pytest
import glob
import numpy as np
from biopandas.pdb import PandasPdb

# Importing utility functions and commands from PocketDruggability package
from PocketDruggability.utils import remove_hydrogens, extract_ligand, get_pocket
from PocketDruggability.cmds import featurize_pocket

# Defining the paths for the test data files.
complexes = glob.glob("tests/Complexes/*pdb")
apo = [c.replace("/Complexes/", "/Apo/").replace("_Complex", "") for c in complexes]
ligands = [c.replace("/Complexes/", "/Ligands/").replace("_Complex", "_ligand") for c in complexes]
pockets = [c.replace("/Complexes/", "/Pocket/").replace("_Complex", "_prox4_pock") for c in complexes]
reports = [c.replace("/Complexes/", "/Reports/Features_").replace("_Complex.pdb", ".rpt") for c in complexes]

@pytest.mark.parametrize("complex_fname, expected_fname", zip(complexes, apo))
def test_apo(complex_fname, expected_fname):
    """
    Test the extraction of the apo form from a protein complex by comparing the result
    to a pre-calculated apo structure.

    Parameters:
    complex_fname (str): The filename of the complex PDB file.
    expected_fname (str): The filename of the expected apo PDB file.
    """
    
    # Read in the complex, remove hydrogens, and drop the 'line_idx' column.
    test_apo = PandasPdb().read_pdb(complex_fname)
    test_apo = remove_hydrogens(test_apo.df["ATOM"])
    test_apo = test_apo.drop("line_idx", axis=1)

    # Read in the expected apo structure and prepare it for comparison.
    expected_apo = PandasPdb().read_pdb(expected_fname)
    expected_apo = expected_apo.df["ATOM"].drop("line_idx", axis=1)
    
    # Assert that the structures are equal.
    assert test_apo.equals(expected_apo)

@pytest.mark.parametrize("complex_fname, expected_fname", zip(complexes, ligands))
def test_ligand(complex_fname, expected_fname):
    """
    Test the extraction of the ligand from a protein complex by comparing the result
    to a pre-calculated ligand structure.

    Parameters:
    complex_fname (str): The filename of the complex PDB file.
    expected_fname (str): The filename of the expected ligand PDB file.
    """
    
    # Define the ligand name and process the complex accordingly.
    lig_name = "TMP"
    test_ligand = PandasPdb().read_pdb(complex_fname)
    test_ligand = remove_hydrogens(test_ligand.df["HETATM"])
    test_ligand  = extract_ligand(test_ligand, lig_name)
    test_ligand = test_ligand.drop("line_idx", axis=1).reset_index(drop=True)

    # Read in the expected ligand structure and prepare it for comparison.
    expected_ligand = PandasPdb().read_pdb(expected_fname)
    expected_ligand = expected_ligand.df["HETATM"].drop("line_idx", axis=1)

    # Assert that the structures are equal.
    assert test_ligand.equals(expected_ligand)

@pytest.mark.parametrize("complex_fname, expected_fname", zip(complexes, pockets))
def test_pocket(complex_fname, expected_fname):
    """
    Test the identification of the pocket within a protein complex by comparing the result
    to a pre-calculated pocket structure.

    Parameters:
    complex_fname (str): The filename of the complex PDB file.
    expected_fname (str): The filename of the expected pocket PDB file.
    """
    
    # Define the ligand name and the cutoff for the interface.
    lig_name = "TMP"
    interface_cutoff = 4.0

    # Read in the complex and process the apo and ligand parts separately.
    test_pdb = PandasPdb().read_pdb(complex_fname)
    test_apo = remove_hydrogens(test_pdb.df["ATOM"])
    test_ligand = remove_hydrogens(test_pdb.df["HETATM"])
    test_ligand = extract_ligand(test_ligand, lig_name)

    # Identify the pocket and prepare the resulting structure.
    test_pocket = get_pocket(test_ligand, test_apo, interface_cutoff)
    test_pocket = test_pocket.drop("line_idx", axis=1).reset_index(drop=True)

    # Read in the expected pocket structure and prepare it for comparison.
    expected_pocket = PandasPdb().read_pdb(expected_fname)
    expected_pocket = expected_pocket.df["ATOM"].drop("line_idx", axis=1)

    # Assert that the structures are equal.
    assert test_pocket.equals(expected_pocket)

@pytest.mark.parametrize("complex_fname, expected_fname", zip(complexes, reports))
def test_reports(complex_fname, expected_fname):
    """
    Test the generation of feature reports for a protein pocket by comparing the result
    to a pre-calculated report.

    Parameters:
    complex_fname (str): The filename of the complex PDB file.
    expected_fname (str): The filename of the expected report file.
    """
    
    # Define parameters for feature extraction.
    lig_name = "TMP"
    interface_cutoff = 4.0

    # Extract features from the complex.
    test_features = featurize_pocket(complex_fname, lig_name, interface_cutoff)
    test_features["PDBid"] = test_features["PDBid"].replace("_Complex", "")

    # Read and process the expected report.
    with open(expected_fname, "r") as fr:
        fr = fr.read().split("\n")[:-1]
        features = fr[-1].split(" ")[1:-1]

    # Parse and store expected features from the report.
    expected_features = {
        "PDBid": fr[0].split(" ")[0],
        "Seq": fr[1].split(": ")[-1],
        "C_RESIDUE": float(features[0]),
        "INERTIA_3": float(features[1]),
        "SMALLEST_SIZE": float(features[2]),
        "SURFACE_HULL": float(features[3]),
        "VOLUME_HULL": float(features[4]),
        "hydrophobic_kyte": float(features[5]),
        "hydrophobicity_pocket": float(features[6]),
        "p_Ccoo": float(features[7]),
        "p_N_atom": float(features[8]),
        "p_Ooh": float(features[9]),
        "p_aliphatic_residue": float(features[10]),
        "p_aromatic_residue": float(features[11]),
        "p_negative_residue": float(features[12]),
    }

    # Compare each feature value in the test features with the expected features.
    for feature_name, test_value in test_features.items():
        expected_value = expected_features[feature_name]
        # Handle string comparison separately.
        if isinstance(test_value, str):
            assert test_value == expected_value, f"{feature_name}: {test_value} != {expected_value}"
        else:
            # Numeric values are compared within a tolerance.
            assert np.isclose(test_value, expected_value, atol=0.001), f"{feature_name}: {test_value} != {expected_value}"
