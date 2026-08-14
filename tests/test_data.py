from pathlib import Path

from scania_aps.data import read_raw_csv


def test_parser_finds_header_and_converts_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "mini.csv"
    path.write_text(
        "metadata line\nanother metadata line\nclass,aa_000,ab_000\nneg,1,na\npos,2,3\n",
        encoding="utf-8",
    )

    dataset = read_raw_csv(path)

    assert dataset.X.shape == (2, 2)
    assert dataset.y.tolist() == [0, 1]
    assert dataset.X["ab_000"].isna().sum() == 1
