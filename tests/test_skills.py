from rlm_kit.skills import discover_skills, load_skills_as_tools
from rlm_kit.trace import EVENT_TOOL_CALL, TraceRecorder, load_events


def _make_skill_dir(tmp_path):
    # Folder-style skill with SKILL.md + frontmatter
    sd = tmp_path / "skills"
    (sd / "recon").mkdir(parents=True)
    (sd / "recon" / "SKILL.md").write_text(
        "---\nname: recon\ndescription: Recon helper\n---\nDo recon steps.\n",
        encoding="utf-8",
    )
    # Flat .md skill, no frontmatter
    (sd / "triage.md").write_text("Triage the finding.\n", encoding="utf-8")
    return str(sd)


def test_discover_skills(tmp_path):
    skills = {s.name: s for s in discover_skills(_make_skill_dir(tmp_path))}
    assert set(skills) == {"recon", "triage"}
    assert skills["recon"].description == "Recon helper"
    assert "recon steps" in skills["recon"].read()


def test_discover_missing_dir_returns_empty():
    assert discover_skills("/no/such/dir") == []


def test_skills_tools_list_and_read(tmp_path):
    list_skills, read_skill = load_skills_as_tools(_make_skill_dir(tmp_path))
    listing = list_skills()
    assert "recon" in listing and "triage" in listing
    assert "recon steps" in read_skill("recon")
    assert "No such skill" in read_skill("ghost")


def test_skills_tools_record_events(tmp_path):
    list_skills, read_skill = load_skills_as_tools(_make_skill_dir(tmp_path))
    path = str(tmp_path / "t.jsonl")
    with TraceRecorder(path, run_id="r1"):
        list_skills()
        read_skill("recon")
    tools = [e for e in load_events(path) if e["type"] == EVENT_TOOL_CALL]
    assert [t["payload"]["tool"] for t in tools] == ["list_skills", "read_skill"]
