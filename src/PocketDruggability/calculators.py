import freesasa
import numpy as np

from scipy.linalg import eigvals
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree

class FreeSASACalculator():

    def __init__(self, protein, config_path=None):

        self.atom_cols = ["atom_name", "residue_name", "residue_number", 
                          "chain_id", "x_coord", "y_coord", "z_coord"]

        self.protein = protein[self.atom_cols].reset_index(drop=False)

        self.config = self.initialize_config(config_path)
        self.structure = self.initialize_structure(self.protein, self.config)

        self.scores = freesasa.calc(self.structure)

    def initialize_config(self, config_path):

        if config_path is None:

            config_dir = __file__[:__file__.rfind("/")+1]
            config_dir = config_dir + "/configs"

            config_path = f"{config_dir}/naccess.config"
        
        return freesasa.Classifier(config_path)
        
    def initialize_structure(self, protein, config):

        structure = freesasa.Structure(classifier=config)

        for row in protein.itertuples(index=False):
            atom_name = " " + row.atom_name + " "
            residue_name = row.residue_name
            residue_number = row.residue_number
            chain_id = row.chain_id
            x_coord = row.x_coord
            y_coord = row.y_coord
            z_coord = row.z_coord

            structure.addAtom(atom_name, residue_name, residue_number, chain_id,
                              x_coord, y_coord, z_coord)

        return structure

        
    def get_hydrophobic_area(self, pocket, allowed_atoms=["C", "S"]):

        pocket_ = pocket.query("element_symbol in @ allowed_atoms")
        indicies = self.protein.merge(pocket_, on=self.atom_cols)["index"]

        area = 0

        for i in indicies:
            area += self.scores.atomArea(i)

        return area


class ResidueCalculator():

    def __init__(self, protein):
        
        self.scores = {"ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, 
                       "CYS": 2.5, "MET": 1.9, "ALA": 1.8, "GLY": -0.4,
                       "THR": -0.7, "TRP": -0.9, "SER": -0.8, "TYR": -1.3,
                       "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5,
                       "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5}

        unique_cols = ["residue_name", "residue_number"]
        self.sequence = protein.drop_duplicates(unique_cols)["residue_name"]

    def get_hydropathy_score(self):

        score = [self.scores.get(r, 0.0) for r in self.sequence]
        score = sum(score) / len(self.sequence)

        return score

    def _get_num_residues(self, residues=[]):

        count = self.sequence.query(f"residue_name in @ residues")
        return len(count)

    def get_num_aromatic_residues(self):

        residues = ["PHE", "TYR", "HIS", "TRP"]
        count = self._get_num_residues(residues)
        
        return count / len(self.sequence)
        
    def get_num_aliphatic_residues(self):

        residues = ["ILE", "LEU", "VAL"]
        count = self._get_num_residues(residues)
        
        return count / len(self.sequence)

    def get_num_negative_residues(self):

        residues = ["ASP", "GLU"]

        count = self._get_num_residues(residues)
        
        return count / len(self.sequence)


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










    

