import os
import csv
import argparse
from xgboost import XGBRegressor
from biopandas.pdb import PandasPdb

# Import utility functions from a relative package
from .utils import remove_hydrogens, extract_ligand, extract_alt, get_pocket

# Import calculator classes from a relative package
from .calculators import FreeSASACalculator, ResidueCalculator, AtomCalculator, GeometryCalculator

def pocket_features(pdb_fname, lig_name, lig_number, lig_chain, lig_alt, protein_alt, interface_cutoff):
    """
    Extract features of the interaction pocket between a protein and a ligand.

    Given a PDB file, the name of the ligand, and an interface cutoff distance, this
    function computes various biophysical and geometric features of the pocket where
    the ligand is situated.

    Parameters
    ----------
    pdb_fname : str
        Path to the PDB file containing protein and ligand structure data.
    lig_name : str
        The residue name of the ligand.
    interface_cutoff : float
        The distance cutoff to define the interface between the ligand and protein.

    Returns
    -------
    features : dict
        A dictionary containing calculated features of the protein-ligand pocket.

    """
    # Extract the base name of the PDB file without path and extension

    if isinstance(pdb_fname, str):
        pdb_name = pdb_fname.split("/")[-1].split(".")[0]
        # Read the PDB file into a PandasPdb object
        pdb = PandasPdb().read_pdb(pdb_fname)
    elif isinstance(pdb_fname, list):
        pdb_name = None
        pdb = PandasPdb().read_pdb_from_list(pdb_fname)

    # Process the protein and ligand data
    protein = remove_hydrogens(pdb.df["ATOM"])
    protein = extract_alt(protein, protein_alt)

    ligand = remove_hydrogens(pdb.df["HETATM"])
    ligand = extract_ligand(ligand, lig_name, lig_number, lig_chain)
    ligand = extract_alt(ligand, lig_alt)
    # print(ligand.shape)

    import pandas as pd

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    # print(ligand)

    if ligand.shape[0] == 0:
        return  {
            "PDBid": pdb_name,
            "Seq": None,
            "C_ATOM":None,
            "C_RESIDUE": None,
            "INERTIA_3": None,
            "SMALLEST_SIZE": None,
            "SURFACE_HULL": None,
            "VOLUME_HULL": None,
            "hydrophobic_kyte": None,
            "hydrophobicity_pocket": None,
            "p_Ccoo": None,
            "p_N_atom": None,
            "p_Ooh": None,
            "p_aliphatic_residue": None,
            "p_aromatic_residue": None,
            "p_negative_residue": None,
        }
    
    # Identify the pocket based on the ligand and interface cutoff
    pocket = get_pocket(ligand, protein, interface_cutoff)

    if pocket.shape[0] <= 3:
        return  {
            "PDBid": pdb_name,
            "Seq": None,
            "C_ATOM":None,
            "C_RESIDUE": None,
            "INERTIA_3": None,
            "SMALLEST_SIZE": None,
            "SURFACE_HULL": None,
            "VOLUME_HULL": None,
            "hydrophobic_kyte": None,
            "hydrophobicity_pocket": None,
            "p_Ccoo": None,
            "p_N_atom": None,
            "p_Ooh": None,
            "p_aliphatic_residue": None,
            "p_aromatic_residue": None,
            "p_negative_residue": None,
        }

    # Initialize feature calculators
    sasa_calc = FreeSASACalculator(protein)
    geom_calc = GeometryCalculator(pocket)
    resi_calc = ResidueCalculator(pocket)
    atom_calc = AtomCalculator(pocket)

    # Calculate features and store them in a dictionary
    features = {
        "PDBid": pdb_name,
        "Seq": resi_calc.get_sequence(),
        "C_ATOM": pocket.shape[0],
        "C_RESIDUE": len(resi_calc.get_sequence()),
        "INERTIA_3": geom_calc.get_inertia(),
        "SMALLEST_SIZE": geom_calc.get_smallest_height(),
        "SURFACE_HULL": geom_calc.get_surface_hull(),
        "VOLUME_HULL": geom_calc.get_volume_hull(),
        "hydrophobic_kyte": resi_calc.get_hydropathy_score(),
        "hydrophobicity_pocket": sasa_calc.get_hydrophobic_area(pocket),
        "p_Ccoo": atom_calc.get_num_Ccoo(),
        "p_N_atom": atom_calc.get_num_N(),
        "p_Ooh": atom_calc.get_num_Ooh(),
        "p_aliphatic_residue": resi_calc.get_num_aliphatic_residues(),
        "p_aromatic_residue": resi_calc.get_num_aromatic_residues(),
        "p_negative_residue": resi_calc.get_num_negative_residues(),
    }

    return features

def predict_activity(features):

    feature_cols = ["C_RESIDUE", "INERTIA_3", "SMALLEST_SIZE", "SURFACE_HULL", 
                    "VOLUME_HULL", "hydrophobic_kyte", "hydrophobicity_pocket", 
                    "p_Ccoo", "p_N_atom", "p_Ooh", "p_aliphatic_residue", 
                    "p_aromatic_residue", "p_negative_residue"]


    features = [[f[c] for c in feature_cols] for f in features]

    model = XGBRegressor()
    model.load_model(os.path.dirname(__file__) + "/models/regressor.json")

    p_act = model.predict(features)

    return p_act 


def cmd_featurize_pocket():
    """
    Command-line interface for extracting pocket features from PDB files.

    This function parses command-line arguments for PDB filenames, the ligand name,
    interface cutoff, and the output CSV filename. It then processes each PDB file
    to extract pocket features and writes the results to the specified CSV file.
    """
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Extract features of protein-ligand pockets from PDB files.")
    parser.add_argument("-pdb", nargs="+", required=True, help="PDB file(s) to process.")
    parser.add_argument("-csv", required=True, help="Output CSV file to store the features.")
    parser.add_argument("-predict", default=True, help="Ligand residue name to consider.")
    parser.add_argument("-lig_name", default="LG1", help="Ligand residue name to consider.")
    parser.add_argument("-lig_number", default=1, help="Ligand residue number to consider.")
    parser.add_argument("-lig_chain", default="X", help="Ligand chain name to consider.")
    parser.add_argument("-lig_alt", default="A", help="Ligand state to consider.")
    parser.add_argument("-protein_alt", default="A", help="Protein state to consider.")
    parser.add_argument("-interface_cutoff", default=4.0, help="Interface cutoff distance.")

    args = parser.parse_args()
    
    results = []  # List to store feature dictionaries

    # Process each PDB file and collect features
    for pdb_fname in args.pdb:
        print("Processing:", pdb_fname)
        features = pocket_features(pdb_fname, args.lig_name, args.lig_number, args.lig_chain, args.lig_alt, args.protein_alt, args.interface_cutoff)
        results.append(features)

    if args.predict:
        p_acts = predict_activity(results)
        results = [{**f, "PredictedActivity": p_act} for f, p_act in zip(results, p_acts)]

    # Write the features to the specified CSV file
    with open(args.csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for features in results:
            writer.writerow(features)

