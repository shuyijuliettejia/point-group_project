import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pointgroup.molecule_reel import construire_molecule_data

def test_point_group_water():
    data = construire_molecule_data("water")
    assert data["point_group"] == "C2v"

def test_point_group_ammonia():
    data = construire_molecule_data("ammonia")
    assert data["point_group"] == "C3v"

def test_molecule_not_found():
    with pytest.raises(ValueError):
        construire_molecule_data("xyzabc123invalide")

def test_water_is_polar():
    data = construire_molecule_data("water")
    assert data["polar"] == True

def test_water_not_chiral():
    data = construire_molecule_data("water")
    assert data["chiral"] == False

def test_construire_molecule_data_invalid_name():
    with pytest.raises(ValueError):
        construire_molecule_data("not_a_real_molecule_abcxyz")