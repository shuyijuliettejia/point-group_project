import numpy as np
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem
from pymatgen.core import Molecule as PmgMol
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from rapidfuzz import process


iupac_name = input("IUPAC name: ")

compounds = pcp.get_compounds(iupac_name, 'name')

if not compounds:
    print("Molecule not found")
    exit()
 
c = compounds[0]
c       = compounds[0]
smiles  = c.connectivity_smiles
formule = c.molecular_formula
masse   = float(c.molecular_weight)

#construction de la molecule 
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
if result == -1:
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
AllChem.MMFFOptimizeMolecule(mol)

#extraction des coordonnées 
conf = mol.GetConformer()
symbols, coords = [], []
for i, atom in enumerate(mol.GetAtoms()):
    pos = conf.GetAtomPosition(i)
    symbols.append(atom.GetSymbol())
    coords.append([pos.x, pos.y, pos.z])
coords = np.array(coords)

#analyse du point group 
pmg_mol     = PmgMol(symbols, coords)
analyzer    = PointGroupAnalyzer(pmg_mol)
point_group = str(analyzer.get_pointgroup())
sym_ops     = analyzer.get_symmetry_operations()

axes, plans, inversion = [], [], False

#analyse des symetries 
for op in sym_ops:
    mat   = op.rotation_matrix
    trace = np.trace(mat)
    det   = round(np.linalg.det(mat), 3)
    eigenvalues, eigenvectors = np.linalg.eig(mat)

    if det == 1.0 and abs(trace - 3) > 0.1:
        for j in range(3):
            if abs(eigenvalues[j].real - 1.0) < 0.1 and abs(eigenvalues[j].imag) < 0.1:
                ax        = eigenvectors[:, j].real
                ax        = ax / np.linalg.norm(ax)
                cos_angle = np.clip((trace - 1) / 2, -1, 1)
                angle     = np.arccos(cos_angle)
                order     = round(2 * np.pi / angle) if angle > 0.01 else 1
                direction = [round(float(x), 3) for x in ax]
                if not any(np.allclose(a["direction"], direction, atol=0.1) for a in axes):
                    axes.append({"direction": direction, "ordre": order, "label": f"C{order}"})

    elif det == -1.0:
        if abs(trace + 3) < 0.1:
            inversion = True
        elif abs(trace - 1) < 0.5:
            for j in range(3):
                if abs(eigenvalues[j].real + 1.0) < 0.1 and abs(eigenvalues[j].imag) < 0.1:
                    normal  = eigenvectors[:, j].real
                    normal  = normal / np.linalg.norm(normal)
                    normale = [round(float(x), 3) for x in normal]
                    if not any(np.allclose(p["normale"], normale, atol=0.1) for p in plans):
                        plans.append({"normale": normale, "type": "σv", "label": "σv"})

if len(plans) == 1:
    plans[0]["type"], plans[0]["label"] = "σh", "σh"
elif len(plans) > 1:
    for k, p in enumerate(plans):
        suffix     = ["", "'", "''", "'''"][k] if k < 4 else str(k)
        p["label"] = f"σv{suffix}"

#propriétés physiques 
is_chiral       = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) > 0
nonpolar_groups = {"Ih", "Oh", "Td", "D6h", "D4h", "D3h", "D2h"}
is_polar        = point_group not in nonpolar_groups and not inversion

molecule_data = {
    "nom":          iupac_name,
    "formule":      formule,
    "point_group":  point_group,
    "chiral":       is_chiral,
    "polar":        is_polar,
    "ir_active":    not inversion,
    "raman_active": True,
    "atomes": [
        {"element": symbols[i],
         "x": round(float(coords[i][0]), 3),
         "y": round(float(coords[i][1]), 3),
         "z": round(float(coords[i][2]), 3)}
        for i in range(len(symbols))
    ],
    "liaisons":  [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()],
    "axes":      axes,
    "plans":     plans,
    "inversion": inversion,
}

print(f"\n{'='*30}")
print(f"Name           : {iupac_name}  |  Formula : {formule}  |  Mass : {masse} g/mol")
print(f"SMILES         : {smiles}")
print(f"Point group    : {point_group}  |  Chiral : {is_chiral}  |  Polar : {is_polar}  |  Inversion : {inversion}")
print(f"Axes  : {axes}")
print(f"Plans : {plans}")

from irreps import print_character_table
print_character_table(point_group)

#Question marhce pas pour Oh et Ih comment faire
#Question comment fiare si on ecrit un mauvais nom iupac et on veut proposer des solutions qui ressemble 