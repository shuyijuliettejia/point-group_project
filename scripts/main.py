from scripts.molecule import molecule_from_iupac
from scripts.point_group import assign_point_group
from scripts.symmetry import has_inversion_center, has_mirror_plane, has_rotation_axis

name = input("Nom IUPAC de la molécule : ")

symbols, coords = molecule_from_iupac(name)

point_group = assign_point_group(symbols, coords)

print("Point group déterminé par notre code :", point_group)

print("Symétries détectées :")
print("Inversion :", has_inversion_center(symbols, coords))
print("Mirror plane :", has_mirror_plane(symbols, coords))
print("C2 axis :", has_rotation_axis(symbols, coords, 2))
print("C3 axis :", has_rotation_axis(symbols, coords, 3))