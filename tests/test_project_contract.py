from pathlib import Path


def test_core_project_contract_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "AGENTS.md").is_file()
    assert (root / "PROGRAM.md").is_file()
    assert (root / "project.yaml").is_file()
