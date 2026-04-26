import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np 
from pymatgen.core import Molecule as PmgMol 
from pymatgen.symmetry.analyzer import PointGroupAnalyzer 

iupac_name = input("Nom IUPAC de la molécule :").strip()
print(f"Recherche de : {iupac_name}")


compounds =pcp.get_compounds(iupac_name, 'name')

if not compounds: 
    raise ValueError(f"Aucun composé trouvé pour {iupac_name}")

c = compounds[0]

smiles = c.canonical_smiles
formule = c.molecular_formula
masse = c.molecular_weight 
cid = c.cid

print(f" CID : {cid}")
print(f"Formule : {formule}")
print(f"Masse : {masse} g/mol")
print( f"SMILES : {smiles}")



mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol) # ajoute les H explicitements 

#placement initale 3D
result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
if result ==-1:
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())

AllChem.MMFFOptimizeMolecule(mol)

#extraction des positions atome par atome 
conf = mol.GetConformer()
symbols = []
coords = []

for i, atom in enumerate(mol.GetAtoms()):
    pos= conf.GetAtomPosition(i)
    symbols.append(atom.GetSymbol())
    coords.append([pos.x, pos.y, pos.z])

#tableau numpy 
coords= np.array(coords)

print(f"{len(symbols)} atomes positionnés")
print(f"Shape coords : {coords.shape}")
print(f"Symboles: {symbols}")



#molecule pymatgen (symboles +coords numpy)
pmg_mol = PmgMol(symbols, coords)
analyzer= PointGroupAnalyzer(pmg_mol)

#Group point 
point_group = str(analyzer.get_pointgroup())

# operation de symetrie 
sym_ops = analyzer.get_symmetry_operations()

print(f"Group ponctuel :{point_group}")
print(f"Nombre d'opérations :{len(sym_ops)}")

#affichage des matrices de chaque opération 
for i, op in enumerate(sym_ops, 1):
    print(f"\nOpération {i}:")
    print(op.rotation_matrix)

