"""자료와 산출물의 경로, 설정 파일 읽기.

폴더를 만드는 일은 하지 않는다. 쓰는 쪽이 만든다.
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
POSTINGS = DATA / "postings"
KED = DATA / "ked"
OUT = ROOT / "experiments" / "outputs"
INTERIM = OUT / "interim"
RUNS = OUT / "runs"
S0 = OUT / "s0"
CONFIG = ROOT / "experiments" / "config.yaml"


def load_config(path: Path = CONFIG) -> dict:
    """YAML 설정 파일을 UTF-8로 읽어 사전으로 돌려준다."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
