from sklearn.neighbors import KDTree

def get_pocket(ligand, protein, interface_cutoff):
    """
    Identify the pocket on a protein surface that interacts with a given ligand.

    This function takes the coordinates of the atoms in the ligand and the protein,
    and uses a KDTree to efficiently search for protein atoms that are within a specified
    distance (interface_cutoff) from any of the ligand atoms. The identified protein
    atoms are considered to be the 'pocket' that the ligand interacts with.

    Parameters
    ----------
    ligand : DataFrame
        A pandas DataFrame containing the coordinates and additional information for
        the ligand atoms. Must include 'x_coord', 'y_coord', 'z_coord' columns.
    protein : DataFrame
        A pandas DataFrame similar to `ligand`, containing data for the protein atoms.
    interface_cutoff : float
        The distance cutoff to consider for the ligand-protein interface. Atoms within
        this distance from the ligand are considered to be part of the pocket.

    Returns
    -------
    pocket : DataFrame
        A pandas DataFrame containing the subset of protein atoms that are within the
        specified interface_cutoff from the ligand atoms.

    Examples
    --------
    >>> pocket = get_pocket(ligand_df, protein_df, 5.0)
    """
    coords_cols = ["x_coord", "y_coord", "z_coord"]

    ligand_coords = ligand[coords_cols]
    protein_coords = protein[coords_cols]
    
    # Create a KDTree for efficient spatial searching
    kdtree = KDTree(protein_coords, metric="euclidean")
    
    # Query the KDTree to find all protein atoms within the interface cutoff of the ligand atoms
    indices = kdtree.query_radius(ligand_coords, r=interface_cutoff)

    # Flatten the list of indices and remove duplicates
    indices = set([ii for sublist in indices for ii in sublist])
    indices = sorted(indices)

    # Extract the pocket atoms from the protein DataFrame
    pocket = protein.iloc[indices]
 
    return pocket
        
def remove_hydrogens(atoms):
    """
    Remove hydrogen atoms from an atom DataFrame.

    Given a DataFrame of atoms, this function filters out all the rows where the element
    symbol is 'H', thus removing all hydrogen atoms from the dataset.

    Parameters
    ----------
    atoms : DataFrame
        A pandas DataFrame containing atom information, including an 'element_symbol'
        column to identify the type of each atom.

    Returns
    -------
    DataFrame
        The input DataFrame minus the rows corresponding to hydrogen atoms.

    Examples
    --------
    >>> no_hydrogens = remove_hydrogens(atom_df)
    """
    return atoms.query("element_symbol != 'H'").reset_index(drop=True)

def extract_ligand(atoms, resn):
    """
    Extract atoms of a ligand with a specified residue name from a DataFrame of atoms.

    Parameters
    ----------
    atoms : DataFrame
        A pandas DataFrame containing atom information, including a 'residue_name'
        column to identify the ligand to which each atom belongs.
    resn : str
        The residue name of the ligand to be extracted.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing only the atoms that belong to the ligand
        with the specified residue name.

    Examples
    --------
    >>> ligand_df = extract_ligand(atom_df, 'LIG')
    """
    return atoms.query(f"residue_name == '{resn}'")
