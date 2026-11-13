"""1단계 기록(`scan_meta.json`), 기록과 지금 설정의 비교, 이름 바꾸기 재시도 시험. 전체 검토 수정에서 더했다.

수치는 가상 자료의 것이다.
"""
import copy
import datetime as dt
import json
import pathlib

import pandas as pd
import pytest

import synth
from test_s00_scan import load_script

SCAN_KEYS = ["data.years", "data.codebook", "data.ksic", "data.ked_firm", "period.start", "period.end",
             "occupations.keco4", "treatment.text_columns", "treatment.keywords", "treatment.agent_skills",
             "salary_band_edges"]


def _scan(synth_data, tmp_path, cfg=None):
    s00 = load_script()
    out = s00.scan(cfg or synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    return s00, out


def test_scan_writes_meta_without_posting_values(synth_data, tmp_path):
    s00, out = _scan(synth_data, tmp_path)
    assert out["scan_meta"] == tmp_path / "s0" / "scan_meta.json"
    text = out["scan_meta"].read_text(encoding="utf-8")
    meta = json.loads(text)
    assert set(meta) == {"scan_started", "scan_finished", "git_commit", "git_uncommitted_changes", "config"}
    assert list(meta["config"]) == SCAN_KEYS == list(s00.SCAN_CONFIG_KEYS)
    started, finished = (dt.datetime.fromisoformat(meta[k]) for k in ("scan_started", "scan_finished"))
    assert started.tzinfo is not None and started <= finished
    assert meta["git_uncommitted_changes"] in (True, False, None)
    assert meta["config"]["period.start"] == synth_data["cfg"]["period"]["start"]
    assert meta["config"]["occupations.keco4"] == ["1332", "0262"]
    # 설정 가운데 2단계에서만 쓰는 항목은 남기지 않는다
    for key in ("skills", "sparse_filter", "definition", "keyword_dictionary", "estimation", "unit",
                "min_use_postings", "require_bzno", "data.keco"):
        assert key not in meta["config"] and f'"{key}"' not in text
    # 공고 단위 값(공고 번호, 사업자번호, 공고 본문)이 없다
    raw = synth._POSTINGS
    for value in [p[0] for p in raw] + [p[2] for p in raw if p[2]] + [p[4] for p in raw if p[4]]:
        assert value not in text


def _set(cfg, key, value):
    out = copy.deepcopy(cfg)
    *parents, last = key.split(".")
    d = out
    for p in parents:
        d = d[p]
    d[last] = value
    return out


def test_build_tables_refuses_interim_files_from_another_config(synth_data, tmp_path):
    s00, _ = _scan(synth_data, tmp_path)
    cfg = _set(synth_data["cfg"], "period.start", "2025-02-01")
    with pytest.raises(RuntimeError, match=r"period\.start") as e:
        s00.build_tables(cfg, synth_data["root"], tmp_path / "s0")
    assert "scan" in str(e.value) and "period.end" not in str(e.value)
    assert not (tmp_path / "s0" / "tables").exists()

    # 다른 항목이 여럿이면 모두 적는다. 묶음 항목은 안쪽 항목 이름까지 적는다.
    cfg = _set(_set(cfg, "treatment.keywords.strict", ["AI\\s*agent"]), "data.years", [2024, 2025])
    with pytest.raises(RuntimeError) as e:
        s00.build_tables(cfg, synth_data["root"], tmp_path / "s0")
    for key in ("period.start", "treatment.keywords.strict", "data.years"):
        assert key in str(e.value)
    assert "treatment.keywords.exclude" not in str(e.value)


@pytest.mark.parametrize("key, value", [
    ("skills.sparse_filter", {"min_rate": 0.002, "min_treated": 1}),
    ("treatment.definition", "all"),
    ("treatment.keyword_dictionary", "broad"),
    ("period.unit", "year"),
    ("occupations.min_use_postings", 2),
    ("sample.require_bzno", False),
    ("estimation.n_folds", 3),
])
def test_report_only_keys_do_not_require_a_new_scan(synth_data, tmp_path, key, value):
    s00, _ = _scan(synth_data, tmp_path)
    tables = s00.build_tables(_set(synth_data["cfg"], key, value), synth_data["root"], tmp_path / "s0")
    assert "t01_exclusions" in tables


def test_missing_or_broken_meta_stops_the_report(synth_data, tmp_path):
    s00, out = _scan(synth_data, tmp_path)
    out["scan_meta"].write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="scan_meta.json"):
        s00.build_tables(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    out["scan_meta"].unlink()
    with pytest.raises(FileNotFoundError, match="scan_meta.json"):
        s00.build_tables(synth_data["cfg"], synth_data["root"], tmp_path / "s0")


def test_failed_scan_removes_meta(synth_data, tmp_path, monkeypatch):
    s00, out = _scan(synth_data, tmp_path)
    assert out["scan_meta"].exists()
    (tmp_path / "s0" / "scan_meta.json.part").write_text("old", encoding="utf-8")

    def fail(*args, **kwargs):
        raise RuntimeError("시험용 실패")
    monkeypatch.setattr(s00.treatment, "build_treat", fail)
    with pytest.raises(RuntimeError, match="시험용 실패"):
        s00.scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    assert not out["scan_meta"].exists() and not (tmp_path / "s0" / "scan_meta.json.part").exists()


def test_report_header_shows_scan_and_report_separately(synth_data, tmp_path):
    s00, out = _scan(synth_data, tmp_path)
    tables = s00.build_tables(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    meta = json.loads(out["scan_meta"].read_text(encoding="utf-8"))
    meta.update(git_commit="scancommit", git_uncommitted_changes=True)
    timing = {"skill_code": None, "n_train": 0, "n_features": 14, "seconds": None, "n_skills_kept": 0,
              "est_total_hours": None}
    text = s00.write_report(tables, timing, synth_data["cfg"], tmp_path / "r.md", commit="reportcommit",
                            scan_meta=meta).read_text(encoding="utf-8")
    head = text[:text.index("## 1.")]
    assert "reportcommit" in head and "scancommit" in head and meta["scan_started"] in head
    assert "커밋하지 않은 변경" in head and head.index("reportcommit") < head.index("scancommit")
    meta["git_uncommitted_changes"] = False
    text = s00.write_report(tables, timing, synth_data["cfg"], tmp_path / "r.md", commit="reportcommit",
                            scan_meta=meta).read_text(encoding="utf-8")
    assert "커밋하지 않은 변경" not in text[:text.index("## 1.")]


def _replace_failing(monkeypatch, s00, n_failures):
    """`.part` 파일의 이름 바꾸기가 파일마다 처음 `n_failures`번 PermissionError로 실패하게 한다."""
    monkeypatch.setattr(s00, "RENAME_WAIT_SECONDS", 0)
    real = pathlib.Path.replace
    tries = {}

    def replace(self, target):
        if self.name.endswith(".part"):
            tries[self.name] = tries.get(self.name, 0) + 1
            if tries[self.name] <= n_failures:
                raise PermissionError("시험용 이름 바꾸기 실패")
        return real(self, target)
    monkeypatch.setattr(pathlib.Path, "replace", replace)
    return tries


def test_rename_is_retried_on_permission_error(synth_data, tmp_path, monkeypatch):
    s00 = load_script()
    tries = _replace_failing(monkeypatch, s00, 2)
    out = s00.scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    assert len(tries) == 9 and set(tries.values()) == {3}
    assert all(p.exists() for p in out.values())
    assert len(pd.read_parquet(out["base"])) == 13   # 과제 3: p13이 더해져 10에서 11로, 과제 3A: p15, p16이 더해져 13으로 바꿨다


def test_rename_gives_up_and_cleans_up(synth_data, tmp_path, monkeypatch):
    s00 = load_script()
    tries = _replace_failing(monkeypatch, s00, 10 ** 6)
    out_dir = tmp_path / "s0"
    with pytest.raises(PermissionError):
        s00.scan(synth_data["cfg"], synth_data["root"], out_dir)
    assert list(tries.values()) == [s00.RENAME_ATTEMPTS]   # 첫 파일에서 멈춘다
    assert sorted(p.name for p in out_dir.iterdir()) == []


def test_meta_keeps_git_values_from_the_start_of_the_scan(synth_data, tmp_path, monkeypatch):
    # 재검토 뒤 더한 시험. 훑는 도중에 커밋이 생겨도 기록에는 실행을 시작할 때의 코드 시점이 남아야 한다.
    # 처치 판별이 처음 불린 뒤(훑는 도중)에는 git이 다른 커밋과 변경 여부를 돌려주게 한다.
    s00 = load_script()
    state = {"scanning": False, "calls": 0}
    real_build_treat = s00.treatment.build_treat

    def build_treat(*args, **kwargs):
        state["scanning"] = True
        return real_build_treat(*args, **kwargs)

    def git_commit():
        state["calls"] += 1
        return "latercommit" if state["scanning"] else "firstcommit"
    monkeypatch.setattr(s00.treatment, "build_treat", build_treat)
    monkeypatch.setattr(s00, "_git_commit", git_commit)
    monkeypatch.setattr(s00, "_git_uncommitted_changes", lambda: state["scanning"])
    out = s00.scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    meta = json.loads(out["scan_meta"].read_text(encoding="utf-8"))
    assert state["scanning"] and state["calls"] == 1
    assert meta["git_commit"] == "firstcommit" and meta["git_uncommitted_changes"] is False
