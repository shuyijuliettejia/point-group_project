CHARACTER_TABLES = {
    "C1": {
        "classes": ["E"],
        "irreps": {
            "A": [1],
        },
        "basis": {
            "A": "x, y, z ; rotations ; toutes vibrations"
        }
    },

    "C2": {
        "classes": ["E"],
        "irreps": {
            "A": [1, 1],
            "B": [1, -1]
        },
        "basis": {
            "A": "z ; Rz ; z² ; x²,y²",
            "B": "x,y ; Rx,Ry ; xz,yz"
        }
    },    
    

    "Ci": {
        "classes": ["E", "i"],
        "irreps": {
            "Ag": [1, 1],
            "Au": [1, -1],
        },
        "basis": {
            "Ag": "fonctions paires, Raman",
            "Au": "x, y, z ; IR"
        }
    },

    "Cs": {
        "classes": ["E", "σ"],
        "irreps": {
            "A'": [1, 1],
            "A''": [1, -1],
        },
        "basis": {
            "A'": "x, y ; Raman",
            "A''": "z ; rotation perpendiculaire"
        }
    },

    "C2v": {
        "classes": ["E", "C2", "σv(xz)", "σv'(yz)"],
        "irreps": {
            "A1": [1, 1, 1, 1],
            "A2": [1, 1, -1, -1],
            "B1": [1, -1, 1, -1],
            "B2": [1, -1, -1, 1],
        },
        "basis": {
            "A1": "z ; x², y², z²",
            "A2": "Rz ; xy",
            "B1": "x, Ry ; xz",
            "B2": "y, Rx ; yz"
        }
    },

    "C3": {
        "classes": ["E", "2C3"],
        "irreps": {
            "A": [1, 1],
            "E": [2, -1],
        },
        "basis": {
            "A": "z ; Rz",
            "E": "(x,y) ; (Rx,Ry)"
        }
    },

    "C3v": {
        "classes": ["E", "2C3", "3σv"],
        "irreps": {
            "A1": [1, 1, 1],
            "A2": [1, 1, -1],
            "E": [2, -1, 0],
        },
        "basis": {
            "A1": "z ; x² + y², z²",
            "A2": "Rz",
            "E": "(x, y), (Rx, Ry) ; (xz, yz), (x²-y², xy)"
        }
    },

    "D2": {
        "classes": ["E", "C2(z)", "C2(y)", "C2(x)"],
        "irreps": {
            "A":  [1, 1, 1, 1],
            "B1": [1, 1, -1, -1],
            "B2": [1, -1, 1, -1],
            "B3": [1, -1, -1, 1],
        },
        "basis": {
            "A": "rotations totales",
            "B1": "z ; Rz",
            "B2": "y ; Ry",
            "B3": "x ; Rx"
        }
    },

    "D2h": {
        "classes": ["E", "C2(z)", "C2(y)", "C2(x)", "i", "σxy", "σxz", "σyz"],
        "irreps": {
            "Ag":  [1, 1, 1, 1, 1, 1, 1, 1],
            "B1g": [1, 1, -1, -1, 1, 1, -1, -1],
            "B2g": [1, -1, 1, -1, 1, -1, 1, -1],
            "B3g": [1, -1, -1, 1, 1, -1, -1, 1],
            "Au":  [1, 1, 1, 1, -1, -1, -1, -1],
            "B1u": [1, 1, -1, -1, -1, -1, 1, 1],
            "B2u": [1, -1, 1, -1, -1, 1, -1, 1],
            "B3u": [1, -1, -1, 1, -1, 1, 1, -1],
        },
        "basis": {
            "Ag": "x², y², z²",
            "B1g": "Rz, xy",
            "B2g": "Ry, xz",
            "B3g": "Rx, yz",
            "B1u": "z",
            "B2u": "y",
            "B3u": "x"
        }
    },

    "Td": {
        "classes": ["E", "8C3", "3C2", "6S4", "6σd"],
        "irreps": {
            "A1": [1, 1, 1, 1, 1],
            "A2": [1, 1, 1, -1, -1],
            "E":  [2, -1, 2, 0, 0],
            "T1": [3, 0, -1, 1, -1],
            "T2": [3, 0, -1, -1, 1],
        },
        "basis": {
            "A1": "x²+y²+z²",
            "E": "(2z²-x²-y², x²-y²)",
            "T1": "(Rx,Ry,Rz)",
            "T2": "(x,y,z)"
        }
    },

    "Oh": {
        "classes": ["E", "8C3", "6C2", "6C4", "3C2", "i", "6S4", "8S6", "3σh", "6σd"],
        "irreps": {
            "A1g": [1,1,1,1,1,1,1,1,1,1],
            "A2g": [1,1,-1,-1,1,1,-1,1,1,-1],
            "Eg":  [2,-1,0,0,2,2,0,-1,2,0],
            "T1g": [3,0,-1,1,-1,3,1,0,-1,-1],
            "T2g": [3,0,1,-1,-1,3,-1,0,-1,1],
            "A1u": [1,1,1,1,1,-1,-1,-1,-1,-1],
            "A2u": [1,1,-1,-1,1,-1,1,-1,-1,1],
            "Eu":  [2,-1,0,0,2,-2,0,1,-2,0],
            "T1u": [3,0,-1,1,-1,-3,-1,0,1,1],
            "T2u": [3,0,1,-1,-1,-3,1,0,1,-1],
        },
        "basis": {
            "A1g": "x²+y²+z²",
            "Eg": "(2z²-x²-y², x²-y²)",
            "T1g": "(Rx,Ry,Rz)",
            "T1u": "(x,y,z)",
            "T2g": "(xy,xz,yz)"
        }
    }
    
}


def get_character_table(point_group):
    """
    Retourne la table de caractères du groupe ponctuel.
    """
    pg = str(point_group).replace(" ", "")

    if pg not in CHARACTER_TABLES:
        return None

    return CHARACTER_TABLES[pg]


def print_character_table(point_group):
    table = get_character_table(point_group)

    if table is None:
        print(f"Table de caractères non disponible pour {point_group}.")
        print("Ajoutez ce groupe dans CHARACTER_TABLES.")
        return

    print(f"\nReprésentations irréductibles pour {point_group}")
    print("-" * 60)

    header = ["Irrep"] + table["classes"] + ["Fonctions de base"]
    print("{:<8}".format(header[0]), end="")
    for h in header[1:-1]:
        print("{:<12}".format(h), end="")
    print(header[-1])

    for irrep, chars in table["irreps"].items():
        print("{:<8}".format(irrep), end="")
        for c in chars:
            print("{:<12}".format(c), end="")
        print(table.get("basis", {}).get(irrep, ""))