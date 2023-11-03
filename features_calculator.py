from biopandas.pdb import PandasPdb

import freesasa

import numpy as np
import pandas as pd
from scipy.linalg import eigvals
from scipy.spatial import ConvexHull

from sklearn.neighbors import KDTree

def get_interface(ligand, protein, interface_cutoff):

    ligand_coords  = ligand[["x_coord", "y_coord", "z_coord"]]
    protein_coords = protein[["x_coord", "y_coord", "z_coord"]]
    
    kdtree = KDTree(protein_coords, metric="euclidean")
    indicies = kdtree.query_radius(ligand_coords, r=interface_cutoff)

    indicies = set([ii for i in indicies for ii in i])
    indicies = sorted(indicies)

    pocket = protein.iloc[indicies]
 
    return pocket
        
 
class FreeSASACalculator():

    def __init__(self, protein):

        
        self.clf = freesasa.Classifier("naccess.config")
        self.structure = freesasa.Structure(classifier=self.clf)

        self.protein = protein
        self.protein = self.protein.reset_index(drop=False)

        for atomName, residueName, residueNumber, chainLabel, x, y, z in self.protein[["atom_name", "residue_name", "residue_number", "chain_id", "x_coord", "y_coord", "z_coord"]].values:
            atomName = " " + atomName + " "

            self.structure.addAtom(atomName, residueName, residueNumber, chainLabel, x, y, z)

        self.scores = freesasa.calc(self.structure)

    def get_hydrophobic_area(self, pocket, allowed_atoms=["C", "S"]):

        pocket_allowed = pocket.query("element_symbol in @ allowed_atoms")
        columns = list(pocket.columns)[:-1]

        indicies = self.protein.merge(pocket_allowed, on=columns)["index"]

        area = 0

        for i in indicies:
            area += self.scores.atomArea(i)

        return area


class GeometryCalculator():

    def __init__(self, pocket):

        self.atoms = pocket[["x_coord", "y_coord", "z_coord"]].values
        self.hull = ConvexHull(self.atoms)

    def get_volume_hull(self):
        return self.hull.volume

    def get_surface_hull(self):
        return self.hull.area

    def get_inertia(self):
        
        center = self.atoms.mean(axis=0)
        atoms_norm = self.atoms - center

        I =  np.zeros((3,3))

        I[0,0] = np.sum(atoms_norm[:,0] ** 2)
        I[1,1] = np.sum(atoms_norm[:,1] ** 2)
        I[2,2] = np.sum(atoms_norm[:,2] ** 2)
        I[1,0] = np.sum(atoms_norm[:,1] * atoms_norm[:,0])
        I[2,0] = np.sum(atoms_norm[:,2] * atoms_norm[:,0])
        I[2,1] = np.sum(atoms_norm[:,2] * atoms_norm[:,1])
        I[0,1] = I[1,0]
        I[0,2] = I[2,0]
        I[1,2] = I[2,1]

        eigen_values = np.linalg.eigvals(I)

        return min(eigen_values)


    def get_smallest_height(self):

        closest_distance = float('inf')
        closest_pair = None

        for simplex in self.hull.simplices:

            normal = self.hull.equations[simplex[0], :-1]
            offset = self.hull.equations[simplex[0], -1]
            
            projections = np.dot(self.atoms, normal)
            min_point = self.atoms[projections.argmin()]
            max_point = self.atoms[projections.argmax()]
            
            distance = np.dot(normal, max_point - min_point)
            
            if distance < closest_distance:
                closest_distance = distance
                closest_pair = (min_point, max_point, distance, normal)

        return closest_pair[2]



class HydropathyCalculator():

    def __init__(self):
        
        self.scores = {"ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, 
                       "CYS": 2.5, "MET": 1.9, "ALA": 1.8, "GLY": -0.4,
                       "THR": -0.7, "TRP": -0.9, "SER": -0.8, "TYR": -1.3,
                       "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5,
                       "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5}

    def get_hydropathy_score(self, protein):

        df = protein.drop_duplicates(["residue_name", "residue_number"])

        score = [self.scores.get(r, 0.0) for r in df["residue_name"]]
        score = sum(score) / len(score)

        return score


class ResiduesCalculator():

    def __init__(self, protein):
        self.protein = protein
        self.protein = self.protein.drop_duplicates(["residue_name", "residue_number"])

    def get_aromatic_residues(self):

        residues = ["PHE", "TYR", "HIS", "TRP"]

        total_count = 0

        for r in residues:
            count = self.protein.query(f"residue_name == '{r}'").shape[0]
            total_count += count

        return total_count / self.protein.shape[0]
        
    def get_aliphatic_residues(self):

        residues = ["ILE", "LEU", "VAL"]

        total_count = 0

        for r in residues:
            count = self.protein.query(f"residue_name == '{r}'").shape[0]
            total_count += count

        return total_count / self.protein.shape[0]

    def get_negative_residues(self):

        residues = ["ASP", "GLU"]

        total_count = 0

        for r in residues:
            count = self.protein.query(f"residue_name == '{r}'").shape[0]
            total_count += count

        return total_count / self.protein.shape[0]




class AtomsCalculator():

    def __init__(self, protein):
        self.protein = protein

    def get_N(self):

        atoms = [[None, "N"], ["GLN", "NE2"], ["ASN", "ND1"]]

        total_count = 0

        for res_name, atom_name in atoms:

            df = self.protein

            if res_name is not None:
                df = df.query(f"residue_name == '{res_name}'")
            else:
                df = df[~df["residue_name"].isin(["GLN", "ASN"])]

            if atom_name is not None:
                df = df.query(f"atom_name == '{atom_name}'")

            total_count += df.shape[0]
        
        return total_count / self.protein.shape[0]

    def get_Ccoo(self):

        atoms = [["ASP", "CB"], ["GLU", "CG"]]

        total_count = 0

        for res_name, atom_name in atoms:

            df = self.protein

            if res_name is not None:
                df = df.query(f"residue_name == '{res_name}'")

            if atom_name is not None:
                df = df.query(f"atom_name == '{atom_name}'")

            total_count += df.shape[0]
        
        return total_count / self.protein.shape[0]

    def get_Ooh(self):
        
        atoms = [["SER", "CB"], ["THR", "CB"]]

        total_count = 0

        for res_name, atom_name in atoms:

            df = self.protein

            if res_name is not None:
                df = df.query(f"residue_name == '{res_name}'")

            if atom_name is not None:
                df = df.query(f"atom_name == '{atom_name}'")

            total_count += df.shape[0]
        
        return total_count / self.protein.shape[0]
    

if __name__ == '__main__':


    pdb = PandasPdb().read_pdb("mini_complex_PV-001797102843_6.pdb")
    
    protein = pdb.df["ATOM"]
    ligand  = pdb.df["HETATM"]

    protein = protein.query("element_symbol != 'H'").reset_index(drop=True)
    # ligand = ligand.query("element_symbol != 'H'")

    pocket = get_interface(ligand, protein, interface_cutoff=4.0)


    sasa_calc = FreeSASACalculator(protein)
    gcalc = GeometryCalculator(pocket)
    h_calc = HydropathyCalculator()
    res_calc = ResiduesCalculator(pocket)
    atom_calc = AtomsCalculator(pocket)


    features = {}
    features["hydrophobicity_pocket"] = sasa_calc.get_hydrophobic_area(pocket)
    features["VOLUME_HULL"] = gcalc.get_volume_hull()
    features["SURFACE_HULL"] = gcalc.get_surface_hull()
    features["INERTIA_3"] = gcalc.get_inertia()
    features["SMALLEST_SIZE"] = gcalc.get_smallest_height()
    features["hydrophobic_kyte"] = h_calc.get_hydropathy_score(pocket)
    features["p_aromatic_residue"] = res_calc.get_aromatic_residues()
    features["p_aliphatic_residue"] = res_calc.get_aliphatic_residues()
    features["p_negative_residue"] = res_calc.get_negative_residues()
    features["p_N_atom"] = atom_calc.get_N()
    features["p_Ccoo"] = atom_calc.get_Ccoo()
    features["p_Ooh"] = atom_calc.get_Ooh()

    print(features)

