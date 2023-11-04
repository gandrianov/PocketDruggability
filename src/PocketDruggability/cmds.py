import csv
import argparse
from biopandas.pdb import PandasPdb

from .utils import remove_hydrogens
from .utils import extract_ligand
from .utils import get_pocket

from .calculators import FreeSASACalculator
from .calculators import ResidueCalculator
from .calculators import AtomsCalculator
from .calculators import GeometryCalculator


def featurize_pocket(pdb_fname, lig_name, interface_cutoff):

    pdb_name = pdb_fname.split("/")[-1].split(".")[0]

    pdb = PandasPdb().read_pdb(pdb_fname)

    protein = remove_hydrogens(pdb.df["ATOM"])
    ligand  = extract_ligand(pdb.df["HETATM"], lig_name)

    pocket = get_pocket(ligand, protein, interface_cutoff)

    sasa_calc = FreeSASACalculator(protein)
    geom_calc = GeometryCalculator(pocket)
    resi_calc = ResidueCalculator(pocket)
    atom_calc = AtomCalculator(pocket)

    features = {}
    features["PDBid"] = pdb_name
    features["Seq"] = resi_calc.get_sequence()
    features["C_RESIDUE"] = len(features["Seq"])
    
    features["INERTIA_3"]     = geom_calc.get_inertia()
    features["SMALLEST_SIZE"] = geom_calc.get_smallest_height()
    features["SURFACE_HULL"]  = geom_calc.get_surface_hull()
    features["VOLUME_HULL"]   = geom_calc.get_volume_hull()
    
    features["hydrophobic_kyte"] = resi_calc.get_hydropathy_score()
    features["hydrophobicity_pocket"] = sasa_calc.get_hydrophobic_area(pocket)
    
    features["p_Ccoo"]   = atom_calc.get_num_Ccoo()
    features["p_N_atom"] = atom_calc.get_num_N()
    features["p_Ooh"]    = atom_calc.get_num_Ooh()
    
    features["p_aliphatic_residue"] = resi_calc.get_num_aliphatic_residues()
    features["p_aromatic_residue"]  = resi_calc.get_num_aromatic_residues()
    features["p_negative_residue"]  = resi_calc.get_num_negative_residues()

    return features

def cmd_featurize_pocket():

    parser = argparse.ArgumentParser()

    parser.add_argument("-pdb", nargs="+", required=True)
    parser.add_argument("-csv", required=True)
    parser.add_argument("-lig_name", default="LG1")
    parser.add_argument("-interface_cutoff", default=4.0)
    

    args = parser.parse_args()   
    
    results = []

    for pdb_fname in args.pdb:
        features = process_pdb(pdb_fname, args.ligname)
        results.append(features)

    with open(args.csv, 'w') as f:
        w = csv.DictWriter(f, results[0].keys())
        w.writeheader()

        for features in results:
            w.writerow(features)

