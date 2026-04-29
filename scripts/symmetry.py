import numpy as np


def center_molecule(coords):
    """
    Recentre la molécule autour de son centre géométrique.
    """
    center = np.mean(coords, axis=0)
    return coords - center


def atoms_match(symbols, coords1, coords2, tol=1e-2):
    """
    Vérifie si deux ensembles de coordonnées représentent la même molécule,
    en tenant compte des types d'atomes.
    """
    used = set()

    for i, pos1 in enumerate(coords1):
        found = False

        for j, pos2 in enumerate(coords2):
            if j in used:
                continue

            if symbols[i] != symbols[j]:
                continue

            if np.linalg.norm(pos1 - pos2) < tol:
                used.add(j)
                found = True
                break

        if not found:
            return False

    return True


def has_inversion_center(symbols, coords, tol=1e-2):
    """
    Teste la présence d'un centre d'inversion.
    Transformation : r -> -r
    """
    coords = center_molecule(coords)
    inverted = -coords

    return atoms_match(symbols, coords, inverted, tol)


def reflect(coords, normal):
    """
    Réflexion par rapport à un plan passant par l'origine.
    normal = vecteur normal au plan.
    """
    normal = np.array(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    reflected = coords - 2 * np.outer(coords @ normal, normal)
    return reflected


def has_mirror_plane(symbols, coords, tol=1e-2):
    """
    Teste quelques plans miroir simples : xy, xz, yz.
    """
    coords = center_molecule(coords)

    candidate_normals = [
        [1, 0, 0],  # plan yz
        [0, 1, 0],  # plan xz
        [0, 0, 1],  # plan xy
    ]

    for normal in candidate_normals:
        reflected = reflect(coords, normal)

        if atoms_match(symbols, coords, reflected, tol):
            return True

    return False


def rotation_matrix(axis, angle):
    """
    Matrice de rotation autour d'un axe quelconque.
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c

    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ])


def has_rotation_axis(symbols, coords, n, tol=1e-2):
    """
    Teste la présence d'un axe Cn parmi les axes x, y, z.
    """
    coords = center_molecule(coords)
    angle = 2 * np.pi / n

    candidate_axes = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    for axis in candidate_axes:
        R = rotation_matrix(axis, angle)
        rotated = coords @ R.T

        if atoms_match(symbols, coords, rotated, tol):
            return True

    return False