import tomllib
from pathlib import Path


def test_core_project_contract_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "LICENSE").is_file()
    assert (root / "AGENTS.md").is_file()
    assert (root / "SKILL.md").is_file()
    assert (root / "agents" / "openai.yaml").is_file()
    assert (root / "references" / "agent-contract.md").is_file()
    assert (root / "references" / "forward-evidence-contract.md").is_file()
    assert (root / "PROGRAM.md").is_file()
    assert (root / "project.yaml").is_file()
    assert (root / "config" / ".env.example").is_file()
    assert not (root / ".env.example").exists()


def test_public_skill_entrypoint_is_separate_from_developer_instructions() -> None:
    root = Path(__file__).resolve().parents[1]
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "只面向修改、测试和发布本仓库的开发 Agent" in agents
    assert "不是 Skill 的运行入口" in agents
    assert "## 什么情况下应该使用这个 Skill" in skill
    assert "不要把本 Skill 用作行情下载器、策略生成器、收益保证或自动上线工具" in skill
    assert "`AGENTS.md` 仅用于开发、测试和发布本仓库" in readme


def test_distribution_declares_mit_license() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["license"] == "MIT"
    assert metadata["project"]["license-files"] == ["LICENSE"]
