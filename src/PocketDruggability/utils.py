from sklearn.neighbors import KDTree

def get_pocket(ligand, protein, interface_cutoff):

    coords_cols = ["x_coord", "y_coord", "z_coord"]

    ligand_coords  = ligand[coords_cols]
    protein_coords = protein[coords_cols]
    
    kdtree = KDTree(protein_coords, metric="euclidean")
    indicies = kdtree.query_radius(ligand_coords, r=interface_cutoff)

    indicies = set([ii for i in indicies for ii in i])
    indicies = sorted(indicies)

    pocket = protein.iloc[indicies]
 
    return pocket
        

def remove_hydrogens(atoms):

	return atoms.query("element_symbol != 'H'").reset_index(drop=True)


def extract_ligand(atoms, resn):

    return atoms.query(f"residue_name != '{resn}'")
