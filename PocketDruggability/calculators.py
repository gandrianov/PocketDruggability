import freesasa
import numpy as np

from scipy.linalg import eigvals
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree

from biopandas.pdb.engines import amino3to1dict as AMINO3TODICT


class FreeSASACalculator:
    """
    A class for calculating solvent accessible surface areas of proteins
    using the FreeSASA library.

    Attributes:
        atom_cols (list of str): Column names for the atoms in the protein.
        protein (pandas.DataFrame): The protein data.
        config (freesasa.Classifier): The configuration for FreeSASA.
        structure (freesasa.Structure): The structure object for FreeSASA calculation.
        scores (freesasa.Result): The computed SASA scores for the structure.
    
    Methods:
        initialize_config(config_path): Initializes the FreeSASA classifier.
        initialize_structure(protein, config): Initializes the FreeSASA structure.
        get_hydrophobic_area(pocket, allowed_atoms): Calculates the hydrophobic surface area of the given pocket.
    """

    def __init__(self, protein, config_path=None):
        """
        Initializes the FreeSASACalculator instance.

        Args:
            protein (pandas.DataFrame): The protein data as a DataFrame, expected to have the necessary columns.
            config_path (str, optional): Path to the configuration file for FreeSASA. Defaults to None, 
                                         in which case a default config path is used.
        """
        self.atom_cols = ["atom_name", "residue_name", "residue_number", 
                          "chain_id", "x_coord", "y_coord", "z_coord"]

        self.protein = protein[self.atom_cols].reset_index(drop=False)
        self.config = self.initialize_config(config_path)
        self.structure = self.initialize_structure(self.protein, self.config)
        self.scores = freesasa.calc(self.structure)

    def initialize_config(self, config_path):
        """
        Initializes the FreeSASA configuration.

        Args:
            config_path (str): The file path to the FreeSASA configuration file.

        Returns:
            freesasa.Classifier: A FreeSASA classifier configured with the specified config file.
        """
        if config_path is None:
            # Set the default configuration directory path.
            config_dir = __file__[:__file__.rfind("/")+1] + "/configs"
            config_path = f"{config_dir}/naccess.config"

        return freesasa.Classifier(config_path)

    def initialize_structure(self, protein, config):
        """
        Initializes a FreeSASA structure from the protein data.

        Args:
            protein (pandas.DataFrame): The protein data.
            config (freesasa.Classifier): The FreeSASA configuration.

        Returns:
            freesasa.Structure: A structure ready for SASA calculation.
        """
        structure = freesasa.Structure(classifier=config)

        # Add each atom in the protein data to the structure.
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
        """
        Calculates the hydrophobic surface area of a pocket within the protein.

        Args:
            pocket (pandas.DataFrame): The pocket data.
            allowed_atoms (list of str, optional): A list of atom element symbols to consider for the hydrophobic area.
                                                    Defaults to ["C", "S"].

        Returns:
            float: The total hydrophobic surface area of the specified pocket.
        """
        pocket_ = pocket.query("element_symbol in @allowed_atoms")
        indices = self.protein.merge(pocket_, on=self.atom_cols)["index"]

        area = 0
        for i in indices:
            area += self.scores.atomArea(i)

        return area


class ResidueCalculator:
    """
    A calculator for analyzing protein sequences.

    This class provides methods to calculate various properties of protein sequences,
    including hydropathy score, the count of different types of residues like aromatic,
    aliphatic, and negative residues based on predefined scores for each residue type.

    Attributes:
        scores (dict): A dictionary mapping residue names to hydropathy scores.
        sequence (DataFrame): A pandas DataFrame representing the unique residues in the
                              protein sequence after duplicates based on residue name and
                              number are dropped.

    Args:
        protein (DataFrame): A pandas DataFrame containing protein sequence data.
                             It must contain columns 'residue_name' and 'residue_number'.
    """

    def __init__(self, protein):
        """
        Initialize the ResidueCalculator with a protein sequence.

        Args:
            protein (DataFrame): The protein data as a pandas DataFrame.
        """
        self.scores = {
            "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8,
            "CYS": 2.5, "MET": 1.9, "ALA": 1.8, "GLY": -0.4,
            "THR": -0.7, "TRP": -0.9, "SER": -0.8, "TYR": -1.3,
            "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5,
            "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5
        }

        unique_cols = ["residue_name", "residue_number"]
        self.sequence = protein.drop_duplicates(unique_cols)["residue_name"]
        self.sequence = self.sequence.to_frame()

    def get_sequence(self):
        """
        Convert the sequence of 3-letter amino acid codes to a single string of 1-letter codes.

        Returns:
            str: The protein sequence in 1-letter amino acid codes. Unknown amino acids are
                 represented by '?'.
        """
        sequence = [AMINO3TODICT.get(s, "?") for s in self.sequence.T.values[0]]
        sequence = "".join(sequence)

        return sequence

    def get_hydropathy_score(self):
        """
        Calculate the average hydropathy score for the protein sequence.

        Returns:
            float: The average hydropathy score.
        """
        score = [self.scores.get(r, 0.0) for r in self.sequence.T.values[0]]
        score = sum(score) / len(self.sequence)

        return score

    def _get_num_residues(self, residues):
        """
        Calculate the proportion of given residues in the protein sequence.

        Args:
            residues (list): A list of residue names to be counted in the sequence.

        Returns:
            float: The proportion of specified residues in the sequence.
        """
        count = self.sequence.query(f"residue_name in @residues")
        return len(count) / len(self.sequence)

    def get_num_aromatic_residues(self):
        """
        Calculate the proportion of aromatic residues in the protein sequence.

        Returns:
            float: The proportion of aromatic residues.
        """
        residues = ["PHE", "TYR", "HIS", "TRP"]
        return self._get_num_residues(residues)

    def get_num_aliphatic_residues(self):
        """
        Calculate the proportion of aliphatic residues in the protein sequence.

        Returns:
            float: The proportion of aliphatic residues.
        """
        residues = ["ILE", "LEU", "VAL"]
        return self._get_num_residues(residues)

    def get_num_negative_residues(self):
        """
        Calculate the proportion of negatively charged residues in the protein sequence.

        Returns:
            float: The proportion of negatively charged residues.
        """
        residues = ["ASP", "GLU"]
        return self._get_num_residues(residues)


class AtomCalculator:
    """
    A class used to calculate the occurrence of specific atoms in a protein structure.

    Attributes
    ----------
    protein : DataFrame
        A pandas DataFrame representing the protein structure with columns including
        'atom_name' and 'residue_name'.

    Methods
    -------
    _get_num_atoms(atom_name, include_res=[], exclude_res=[])
        A helper method to count the occurrences of a specific atom within the protein.
        It can include or exclude specific residues.

    get_num_N()
        Calculates the total number of nitrogen atoms from the side chains of asparagine (ASN)
        and glutamine (GLN), along with the backbone nitrogen atoms.

    get_num_Ccoo()
        Calculates the total number of carbon atoms which are part of carboxylate groups
        in aspartate (ASP) and glutamate (GLU).

    get_num_Ooh()
        Calculates the total number of oxygen atoms bound to hydroxyl groups in serine (SER)
        and threonine (THR).
    """

    def __init__(self, protein):
        """
        Parameters
        ----------
        protein : DataFrame
            A pandas DataFrame representing the protein structure which should have columns
            for 'atom_name' and 'residue_name'.
        """
        self.protein = protein

    def _get_num_atoms(self, atom_name, include_res=[], exclude_res=[]):
        """
        Count the occurrences of a specific atom in the protein, with the ability to
        include or exclude certain residues.

        Parameters
        ----------
        atom_name : str
            The name of the atom to be counted.
        include_res : list, optional
            A list of residue names to include in the count (the default is an empty list, which
            implies no residues are specifically included).
        exclude_res : list, optional
            A list of residue names to exclude from the count (the default is an empty list, which
            implies no residues are specifically excluded).

        Returns
        -------
        float
            The fraction of the specific atoms with respect to the total number of atoms in the protein.
        """
        count = 0

        atoms = self.protein.query(f"atom_name == '{atom_name}'")

        if include_res:
            atoms = atoms.query("residue_name in @include_res")

        if exclude_res:
            atoms = atoms.query("residue_name not in @exclude_res")

        return atoms.shape[0] / self.protein.shape[0]

    def get_num_N(self):
        """
        Calculate the total normalized count of nitrogen atoms from the side chains of
        asparagine and glutamine, and the backbone nitrogen atoms.

        Returns
        -------
        float
            The total normalized count of nitrogen atoms as specified.
        """
        atoms = [[None, "N"], ["GLN", "NE2"], ["ASN", "ND1"]]
        residues = [a for r, a in atoms if r]

        total = 0

        for resn, atomn in atoms:
            if resn is None:
                total += self._get_num_atoms(atomn, exclude_res=residues)
            else:
                total += self._get_num_atoms(atomn, include_res=[resn])

        return total

    def get_num_Ccoo(self):
        """
        Calculate the total normalized count of carbon atoms that are part of the carboxylate
        groups in aspartate and glutamate.

        Returns
        -------
        float
            The total normalized count of carbon atoms in carboxylate groups as specified.
        """
        atoms = [["ASP", "CB"], ["GLU", "CG"]]

        total = 0

        for resn, atomn in atoms:
            total += self._get_num_atoms(atomn, include_res=[resn])

        return total

    def get_num_Ooh(self):
        """
        Calculate the total normalized count of oxygen atoms bound to hydroxyl groups in
        serine and threonine.

        Returns
        -------
        float
            The total normalized count of oxygen atoms bound to hydroxyl groups as specified.
        """
        atoms = [["SER", "CB"], ["THR", "CB"]]

        total = 0

        for resn, atomn in atoms:
            total += self._get_num_atoms(atomn, include_res=[resn])

        return total


class GeometryCalculator:
    """
    A class used to perform geometric calculations on a set of points in 3D space.

    Attributes
    ----------
    atoms : numpy.ndarray
        Array of coordinates for the points in 3D space.
    hull : scipy.spatial.ConvexHull
        The convex hull encompassing all points.

    Methods
    -------
    get_volume_hull()
        Calculates the volume of the convex hull.
        
    get_surface_hull()
        Calculates the surface area of the convex hull.
        
    get_inertia()
        Calculates and returns the minimum eigenvalue of the inertia tensor, which can be interpreted as the principal moment of inertia about the axis of rotation.
        
    get_smallest_height()
        Finds and returns the smallest distance between parallel planes that sandwich the convex hull. This can be thought of as the smallest height of the convex hull if it were to be 'stood up' on one of its faces.
    """

    def __init__(self, pocket):
        """
        Parameters
        ----------
        pocket : pandas.DataFrame
            A DataFrame containing at least 'x_coord', 'y_coord', 'z_coord' columns which represent the coordinates of points in 3D space.
        """
        # Extracts the relevant coordinate columns and converts to a numpy array for calculations
        self.atoms = pocket[["x_coord", "y_coord", "z_coord"]].values
        
        # Constructs the convex hull for the set of points
        self.hull = ConvexHull(self.atoms)

    def get_volume_hull(self):
        """Calculates the volume of the convex hull.

        Returns
        -------
        float
            The volume of the convex hull.
        """
        return self.hull.volume

    def get_surface_hull(self):
        """Calculates the surface area of the convex hull.

        Returns
        -------
        float
            The surface area of the convex hull.
        """
        return self.hull.area

    def get_inertia(self):
        """Calculates the minimum principal moment of inertia for the point cloud about its centroid.

        Returns
        -------
        float
            The minimum eigenvalue of the inertia tensor, representing the smallest principal moment of inertia.
        """
        # Calculate the centroid of the atoms
        center = self.atoms.mean(axis=0)
        # Normalize the atoms by subtracting the centroid
        atoms_norm = self.atoms - center

        # Initialize the inertia tensor
        I = np.zeros((3,3))

        # Compute the diagonal components of the inertia tensor
        I[0,0] = np.sum(atoms_norm[:,0] ** 2)
        I[1,1] = np.sum(atoms_norm[:,1] ** 2)
        I[2,2] = np.sum(atoms_norm[:,2] ** 2)

        # Compute the off-diagonal components of the inertia tensor
        I[1,0] = I[0,1] = np.sum(atoms_norm[:,1] * atoms_norm[:,0])
        I[2,0] = I[0,2] = np.sum(atoms_norm[:,2] * atoms_norm[:,0])
        I[2,1] = I[1,2] = np.sum(atoms_norm[:,2] * atoms_norm[:,1])

        # Compute eigenvalues and return the minimum one
        eigen_values = np.linalg.eigvals(I)
        return min(eigen_values)

    def get_smallest_height(self):
        """Finds the smallest distance between two parallel planes that sandwich the convex hull.

        Returns
        -------
        float
            The smallest height of the convex hull.
        """
        # Initialize the variable for the closest distance
        closest_distance = float('inf')
        closest_pair_planes = None

        # Iterate over the faces of the convex hull
        for i, normal in enumerate(self.hull.equations[:, :-1]):
            # Get the plane offset from the equation of the plane
            plane_offset = self.hull.equations[i, -1]

            # Compute distances from the plane to all points
            distances = np.dot(self.atoms, normal) + plane_offset

            # Find the maximum and minimum distances to identify the furthest points
            max_distance_point = self.atoms[np.argmax(distances)]
            min_distance_point = self.atoms[np.argmin(distances)]

            # The distance between these two points is the height for this plane
            distance = np.max(distances) - np.min(distances)

            # If this distance is the smallest thus far, update closest_distance
            if distance < closest_distance:
                closest_distance = distance
                closest_pair_planes = (normal, -normal, closest_distance)

        # Return the smallest height found
        return closest_pair_planes[2]