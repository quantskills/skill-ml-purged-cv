from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_chinese_readme_covers_complete_public_workflow() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required_sections = (
        "## 一分钟快速开始",
        "## 为什么常规 K-Fold 不适合金融时序",
        "## 核心逻辑",
        "## 五类互补证据",
        "## 真正的时序策略过拟合 Benchmark",
        "## 可训练时序模型：验证 Purge 与 Embargo 是否真正进入训练链路",
        "## 整体工作流程",
        "## 使用自己的数据",
        "## 三个输入文件分别做什么",
        "## 如何判断结果",
        "## 任意特征与平稳性",
        "## 一次性最终 Holdout",
        "## 前向证据：真正回答“未来还生效吗”",
        "## 标准结果应该怎样解读",
        "## 许可证",
    )
    for section in required_sections:
        assert section in readme

    required_commands = (
        "purged-cv-skill demo",
        "purged-cv-skill example",
        "purged-cv-skill run",
        "purged-cv-upload audit",
        "purged-cv-upload evaluate",
        "purged-cv-strategy demo",
        "benchmark-temporal-models",
        "purged-cv-forward init",
        "purged-cv-forward record",
        "purged-cv-forward settle",
        "purged-cv-forward status",
    )
    for command in required_commands:
        assert command in readme

    required_boundaries = (
        "WAITING_FOR_FUTURE_DATA",
        "LOCAL_APPEND_ONLY_NOT_EXTERNALLY_NOTARIZED",
        "production_authorization=NOT_AUTHORIZED",
        "不能证明模型赚钱",
    )
    for boundary in required_boundaries:
        assert boundary in readme

    assert "Markdown 是否会生效" not in readme
    assert "tickets 如何拆分" not in readme
    assert "[MIT License](LICENSE)" in readme


def test_skill_metadata_and_progressive_references_cover_v090() -> None:
    skill_text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    frontmatter_text = skill_text.split("---", 2)[1].strip()
    frontmatter = dict(
        line.split(":", 1) for line in frontmatter_text.splitlines() if ":" in line
    )
    frontmatter = {key.strip(): value.strip() for key, value in frontmatter.items()}
    assert set(frontmatter) == {"name", "description"}
    assert frontmatter["name"] == "skill-ml-purged-cv"
    for trigger in (
        "Purged K-Fold",
        "CPCV",
        "策略选择过拟合",
        "Temporal Forward Evidence",
        "预测是否在标签成熟前登记",
    ):
        assert trigger in frontmatter["description"]
    for reference in (
        "references/agent-contract.md",
        "references/strategy-benchmark-contract.md",
        "references/forward-evidence-contract.md",
    ):
        assert reference in skill_text

    metadata = (ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    assert "$skill-ml-purged-cv" in metadata
    short_line = next(
        line for line in metadata.splitlines() if "short_description:" in line
    )
    short_description = short_line.split(":", 1)[1].strip().strip('"')
    assert 25 <= len(short_description) <= 64


def test_scratch_is_local_only_release_state() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/.scratch/" in ignore
