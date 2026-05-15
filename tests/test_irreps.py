from pointgroup.irreps import get_character_table

def test_get_character_table_c2v():
    table = get_character_table("C2v")

    assert table is not None
    assert table["classes"] == ["E", "C2", "σv(xz)", "σv'(yz)"]
    assert table["irreps"]["A1"] == [1, 1, 1, 1]


def test_get_character_table_unknown_group():
    table = get_character_table("ABC")

    assert table is None

def test_get_character_table_removes_spaces():
    table = get_character_table(" C2v ")

    assert table is not None
    assert "A1" in table["irreps"]