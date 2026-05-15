from pointgroup.visualisation import construire_xyz, deduire_proprietes

def test_construire_xyz_water():
    molecule_data = {
        "nom": "water",
        "atomes": [
            {"element": "O", "x": 0.0, "y": 0.0, "z": 0.0},
            {"element": "H", "x": 0.7, "y": 0.6, "z": 0.0},
            {"element": "H", "x": -0.7, "y": 0.6, "z": 0.0},
        ],
    }

    xyz = construire_xyz(molecule_data)

    assert xyz.startswith("3\nwater")
    assert "O  0.0000  0.0000  0.0000" in xyz
    assert "H  0.7000  0.6000  0.0000" in xyz

def test_construire_xyz_default_name():
    molecule_data = {
        "atomes": [
            {"element": "He", "x": 0.0, "y": 0.0, "z": 0.0},
        ]
    }

    xyz = construire_xyz(molecule_data)

    assert xyz.startswith("1\nmolecule")

def test_deduire_proprietes_without_inversion():
    molecule_data = {
        "point_group": "C2v",
        "chiral": False,
        "polar": True,
        "ir_active": True,
        "raman_active": True,
        "inversion": False,
    }

    chiral, polaire, ir_txt, raman_txt = deduire_proprietes(molecule_data)

    assert chiral is False
    assert polaire is True
    assert ir_txt == "Oui"
    assert raman_txt == "Oui"


def test_deduire_proprietes_with_inversion():
    molecule_data = {
        "point_group": "D2h",
        "chiral": False,
        "polar": False,
        "ir_active": False,
        "raman_active": True,
        "inversion": True,
    }

    chiral, polaire, ir_txt, raman_txt = deduire_proprietes(molecule_data)

    assert chiral is False
    assert polaire is False
    assert ir_txt == "Partiel (règle d'exclusion)"
    assert raman_txt == "Partiel (règle d'exclusion)"