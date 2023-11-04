import csv
import argparse
from biopandas.pdb import PandasPdb

# Import utility functions from a relative package
from .utils import remove_hydrogens, extract_ligand, get_pocket

# Import calculator classes from a relative package
from .calculators import FreeSASACalculator, ResidueCalculator, AtomCalculator, GeometryCalculator


def featurize_pocket(pdb_fname, lig_name, interface_cutoff):
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
    pdb_name = pdb_fname.split("/")[-1].split(".")[0]

    # Read the PDB file into a PandasPdb object
    pdb = PandasPdb().read_pdb(pdb_fname)

    # Process the protein and ligand data
    protein = remove_hydrogens(pdb.df["ATOM"])
    ligand = remove_hydrogens(pdb.df["HETATM"])
    ligand = extract_ligand(ligand, lig_name)
    
    # Identify the pocket based on the ligand and interface cutoff
    pocket = get_pocket(ligand, protein, interface_cutoff)

    # Initialize feature calculators
    sasa_calc = FreeSASACalculator(protein)
    geom_calc = GeometryCalculator(pocket)
    resi_calc = ResidueCalculator(pocket)
    atom_calc = AtomCalculator(pocket)

    # Calculate features and store them in a dictionary
    features = {
        "PDBid": pdb_name,
        "Seq": resi_calc.get_sequence(),
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
    parser.add_argument("-lig_name", default="LG1", help="Ligand residue name to consider.")
    parser.add_argument("-interface_cutoff", type=float, default=4.0, help="Interface cutoff distance.")

    args = parser.parse_args()
    
    results = []  # List to store feature dictionaries

    # Process each PDB file and collect features
    for pdb_fname in args.pdb:
        print("Processing:", pdb_fname)
        features = featurize_pocket(pdb_fname, args.lig_name, args.interface_cutoff)
        results.append(features)

    # Write the features to the specified CSV file
    with open(args.csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        for features in results:
            writer.writerow(features)

