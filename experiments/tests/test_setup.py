import paths

def test_paths_point_inside_project():
    assert paths.DATA == paths.ROOT / "data"
    assert paths.INTERIM == paths.ROOT / "experiments" / "outputs" / "interim"
    assert paths.S0 == paths.ROOT / "experiments" / "outputs" / "s0"

def test_config_carries_decisions():
    cfg = paths.load_config()
    assert len(cfg["occupations"]["keco4"]) == 20          # 결정 39에서 25에서 20으로 바꿨다
    assert cfg["occupations"]["min_use_postings"] == 30
    assert cfg["treatment"]["agent_skills"]["v31_prefix"] == "IT-06-012-"
    assert len(cfg["treatment"]["agent_skills"]["move_to_dev"]) == 5
    assert cfg["treatment"]["agent_skills"]["move_to_use"] == ["IT-05-0396"]
    assert cfg["treatment"]["definition"] == "use_any"     # 결정 40에서 use에서 바꿨다
    assert cfg["sample"]["require_bzno"] is True
    assert cfg["estimation"]["n_folds"] == 5

def test_package_importable():
    import aabundle  # noqa: F401
