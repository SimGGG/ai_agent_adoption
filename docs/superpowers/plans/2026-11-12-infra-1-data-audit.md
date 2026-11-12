# 연구 수행 환경 1차 구축: 틀, 자료 처리, 자료 점검 — 실행 계획

> **작업자(에이전트)에게:** 이 계획은 superpowers:subagent-driven-development(권장) 또는 superpowers:executing-plans 스킬로 과제 단위로 실행한다. 각 단계는 체크박스(`- [ ]`)로 진행 상황을 표시한다.

**목표:** 실험 공간과 논문 공간의 틀을 만들고, 원자료를 분석용 자료로 바꾸는 모듈과 자료 점검(S0) 스크립트를 만들어 자료 점검 보고서 `docs/s0-data-audit.md`를 낸다.

**구조:** 공용 함수는 `experiments/src/aabundle/` 모듈에 두고 pytest로 시험한다. 자료 점검 스크립트는 원자료를 한 번 훑어 공고 단위 중간 자료를 만드는 단계와, 그 중간 자료로 표와 보고서를 만드는 단계로 나눈다. 시험은 손으로 만든 12건짜리 가상 공고 자료로 하고, 실제 자료는 마지막 과제에서만 읽는다.

**기술:** Python 3.12, conda 가상환경, pandas, pyarrow, LightGBM, PyYAML, openpyxl, pytest.

**설계 문서:** `docs/superpowers/specs/2026-11-12-research-infra-design.md` (이하 "설계 문서"). 작업자는 이 계획과 설계 문서를 함께 읽는다.

**범위:** 설계 문서 10장의 구축 순서 1번(틀과 환경)과 2번(자료 처리와 자료 점검)까지이다. 3~5번(추정 모듈, 단계 스크립트 `s01`~`s12`와 내보내기, 논문 원고 틀)은 저자가 자료 점검 보고서를 보고 확정 사항을 정한 뒤 두 번째 계획으로 작성한다. `skills.py`의 `bundle_metrics`와 공고 × 숙련 0/1 표 만들기, `confounders.py`, `dml.py`, `network.py`, `runlog.py`, `export.py`는 두 번째 계획의 몫이다.

## 공통 제약

- 원자료(`data/`)는 읽기만 한다. 어떤 코드도 `data/` 아래에 쓰지 않는다.
- `data/ked/KED5002.txt`의 19번째 항목(0부터 세면 18번)은 개인사업자의 주민등록번호이다. 어떤 코드도 이 항목을 읽지 않는다. 화면 출력, 로그, 시험 자료에도 실제 값을 넣지 않는다.
- 공고 원문 열(`detail`, `title`)은 처치 판별에서만 읽고, 중간 자료와 보고서에 원문을 저장하지 않는다.
- 기준값은 `experiments/config.yaml`에만 둔다. 모듈과 스크립트에 기준값을 직접 적지 않는다.
- 시험과 가상 자료에서 나온 수치는 연구 결과로 적지 않는다. `docs/s0-data-audit.md`에는 실제 자료로 실행한 수치만 들어간다.
- Python은 3.12, 가상환경은 conda 가상환경 `aabundle`을 쓴다(저자 결정, 설계 문서 결정 21). uv는 쓰지 않는다. 패키지는 `environment.yml`에 적어 conda-forge에서 설치하고, 설치된 버전은 `environment.lock.yml`에 남긴다. 가상환경은 conda의 기본 위치(`anaconda3\envs\aabundle`)에 생기므로 구글 드라이브 밖에 있다. 다른 conda 가상환경과 `base`는 건드리지 않는다.
- 모든 파이썬 명령은 저장소 최상위 폴더에서 `./tools/py.ps1`로 실행한다(과제 1에서 만든다). 이 스크립트는 `conda run -n aabundle --no-capture-output python -X utf8` 뒤에 받은 인자를 붙여 실행한다. 예: `./tools/py.ps1 -m pytest -q`. Git Bash에서는 `conda run`이 가상환경이 아닌 다른 파이썬을 부르므로(과제 1에서 확인) 가상환경의 파이썬을 직접 부른다: `PYTHONNOUSERSITE=1 PYTHONUTF8=1 "$(conda info --base)/envs/aabundle/python.exe" -X utf8 -m pytest -q`. `tools/py.ps1`은 사용자 폴더에 따로 설치된 패키지가 가상환경의 패키지를 가리지 않도록 `PYTHONNOUSERSITE=1`도 설정한다.
- 작업은 이 저장소 폴더에서 `infra-1` 브랜치로 한다. 별도 작업 폴더(git worktree)는 만들지 않는다. `data/`가 git 밖에 있어 다른 폴더에서는 원자료를 읽을 수 없다.
- 과제가 끝날 때마다 커밋한다. `data/`, `experiments/outputs/`, `test.ipynb`는 커밋하지 않는다. 커밋 메시지는 한국어로 쓰고 끝에 `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`를 붙인다.
- 사용자에게 보이는 한국어 문서(`README.md`, `CLAUDE.md`, 보고서, 주석)는 사용자의 용어 원칙을 따른다: 영어 개념을 줄여 새 용어를 만들지 않는다. 은유를 쓰지 않는다. 개념은 기능을 풀어서 적는다. 표준 학술 용어는 처음 나올 때 한 문장으로 정의한다.
- 입력 파일이나 필요한 열이 없으면 무엇이 없는지 적은 예외를 내고 바로 멈춘다. 결합 전후로 행 수가 달라지면 멈춘다.

## 검토 때 특히 볼 곳

설계 문서가 직접 말하지 않지만 실제 자료에서 만날 입력이다. 각 줄의 시험은 해당 과제에 넣었다.

1. **KED 원자료에서 구분자가 하나 더 들어간 행.** 항목이 한 칸씩 밀리면 주민등록번호가 설립일자 자리로 올 수 있다. 구분자 수가 47개가 아닌 줄은 표로 읽기 전에 통째로 버리고, 설립일자는 8자리 날짜 형식이 아니면 결측으로 둔다. (과제 4)
2. **숙련 코드 열과 숙련명 열의 개수가 다른 공고.** 숙련명으로 코드를 찾을 수 없으므로 코드만으로 찾고 건수를 센다. 멈추지 않는다. (과제 5)
3. **근무지(`region`)에 시도 이름이 없는 공고.** `광산구`, `아시아·중동`, `서울전체,`처럼 형식이 제각각이다. 알 수 없으면 `기타`로 두고 멈추지 않는다. (과제 7)
4. **형식이 다른 산업 코드.** `Q86201`, `1220`, `C30`, 빈 문자열이 섞여 있다. 변환할 수 없으면 결측으로 둔다. (과제 3)
5. **처치 공고가 한 건도 없는 기간과 직종.** 2021~2022년에는 생성형 AI 활용 숙련 공고가 거의 없다. 집계표에는 0으로 나와야 하고 희소 필터와 학습 시간 측정이 멈추면 안 된다. (과제 5, 과제 9)

---

## 파일 구성

| 파일 | 역할 | 과제 |
|---|---|---|
| `environment.yml`, `environment.lock.yml`, `pyproject.toml`, `tools/py.ps1` | 환경 고정, 실행 명령 | 1 |
| `CLAUDE.md`, `experiments/CLAUDE.md`, `paper/CLAUDE.md`, `README.md`, `docs/decisions.md` | 작업 규칙과 안내 | 1 |
| `experiments/config.yaml`, `experiments/paths.py` | 설정과 경로 | 1 |
| `experiments/src/aabundle/rawio.py` | 원자료와 중간 자료 읽기·쓰기 | 2 |
| `experiments/tests/synth.py` | 시험용 가상 자료 만들기 | 2 |
| `experiments/src/aabundle/classif.py` | 산업·직종 분류표, 코드 형식 맞추기 | 3 |
| `experiments/src/aabundle/ked.py` | KED 기업개요에서 정해진 항목만 추출 | 4 |
| `experiments/src/aabundle/skills.py` | 여섯 숙련 열 풀기, 최종판 코드 찾기, 희소 필터 | 5 |
| `experiments/src/aabundle/treatment.py` | 키워드 처치, 숙련코드 처치, 처치 정의별 공고 고르기 | 6 |
| `experiments/src/aabundle/sample.py` | 공고 기본 자료 구성, 제외 규칙 | 7 |
| `experiments/scripts/s00_data_audit.py` | 자료 점검 실행 | 8, 9 |
| `experiments/tests/test_realdata.py`, `docs/s0-data-audit.md` | 실제 자료 확인, 보고서 | 10 |

---

### 과제 1: 틀과 환경

**파일:**
- 생성: `environment.yml`, `environment.lock.yml`, `pyproject.toml`, `tools/py.ps1`, `README.md`, `CLAUDE.md`, `docs/decisions.md`
- 생성: `experiments/CLAUDE.md`, `experiments/config.yaml`, `experiments/paths.py`, `experiments/src/aabundle/__init__.py`, `experiments/scripts/.gitkeep`, `experiments/notebooks/.gitkeep`, `experiments/tests/test_setup.py`
- 생성: `paper/CLAUDE.md`, `paper/exhibits/.gitkeep`, `paper/manuscript/.gitkeep`, `paper/references/.gitkeep`, `paper/reviews/.gitkeep`, `paper/submissions/.gitkeep`, `study-notes/.gitkeep`

**주고받는 것:**
- 만드는 것: `paths.ROOT`, `paths.DATA`, `paths.POSTINGS`, `paths.KED`, `paths.OUT`, `paths.INTERIM`, `paths.RUNS`, `paths.S0`, `paths.CONFIG` (모두 `pathlib.Path`), `paths.load_config(path: Path = CONFIG) -> dict`

- [ ] **1단계: 브랜치를 만든다**

```bash
git checkout -b infra-1
```

- [ ] **2단계: 실패하는 시험을 쓴다** (`experiments/tests/test_setup.py`)

```python
import paths

def test_paths_point_inside_project():
    assert paths.DATA == paths.ROOT / "data"
    assert paths.INTERIM == paths.ROOT / "experiments" / "outputs" / "interim"
    assert paths.S0 == paths.ROOT / "experiments" / "outputs" / "s0"

def test_config_carries_decisions():
    cfg = paths.load_config()
    assert len(cfg["occupations"]["keco4"]) == 28
    assert cfg["occupations"]["min_use_postings"] == 30
    assert cfg["treatment"]["agent_skills"]["v31_prefix"] == "IT-06-012-"
    assert len(cfg["treatment"]["agent_skills"]["move_to_dev"]) == 5
    assert cfg["treatment"]["agent_skills"]["move_to_use"] == ["IT-05-0396"]
    assert cfg["treatment"]["definition"] == "use"
    assert cfg["sample"]["require_bzno"] is True
    assert cfg["estimation"]["n_folds"] == 5

def test_package_importable():
    import aabundle  # noqa: F401
```

- [ ] **3단계: 환경 파일을 만든다**

`environment.yml`: 이름 `aabundle`, 채널은 `conda-forge` 하나. 패키지: `python=3.12`, `pip`, `hatchling`, `numpy`, `pandas`, `pyarrow`, `scipy`, `scikit-learn`, `lightgbm`, `statsmodels`, `doubleml`, `networkx`, `matplotlib`, `pyyaml`, `openpyxl`, `pytest`, `jupyterlab`. 버전은 `python`만 적고 나머지는 적지 않는다(설치된 버전은 `environment.lock.yml`에 남긴다).

`pyproject.toml`: 프로젝트 이름 `aabundle`, 버전 `0.1.0`, `requires-python = ">=3.12,<3.13"`. 의존 패키지 목록은 비워 둔다(패키지는 `environment.yml`에서만 관리한다. pip가 conda 패키지 위에 다른 판을 덮어 설치하지 않게 하기 위해서다). 빌드는 `hatchling`, 패키지 위치는 `[tool.hatch.build.targets.wheel] packages = ["experiments/src/aabundle"]`. pytest 설정은 `testpaths = ["experiments/tests"]`, `pythonpath = ["experiments", "experiments/tests"]`, `markers = ["realdata: data/ 폴더의 실제 자료를 읽는 시험"]`, `addopts = "-m 'not realdata'"`(실제 자료 시험은 `-m realdata`를 줄 때만 돈다).

`openpyxl`은 설계 문서 9장의 목록에 없지만 분류표(`.xlsx`)를 읽는 데 필요해 더한다.

`tools/py.ps1`:

```powershell
$env:PYTHONUTF8 = "1"
conda run -n aabundle --no-capture-output python -X utf8 @args
exit $LASTEXITCODE
```

- [ ] **4단계: `experiments/config.yaml`을 만든다** (아래 내용 그대로)

```yaml
# 확정 사항을 담는 설정. 값의 근거는 docs/decisions.md와 설계 문서 2장, 6.7절에 있다.
data:
  years: [2021, 2022, 2023, 2024, 2025, 2026]
  codebook: data/postings/ojp_codebook_skill.csv
  ksic: data/postings/KSIC10.xlsx
  keco: data/postings/KECO2025.xlsx
  ked_firm: data/ked/KED5002.txt

period:                      # 설계 문서 6.7절 13번: 자료 점검 뒤 확정. 지금은 자료 전체
  start: "2021-01-01"
  end: "2026-12-31"
  unit: quarter              # proposal.md 10장 2번: 자료 점검 뒤 확정

occupations:                 # 결정 20
  min_use_postings: 30
  keco4: ["0122", "0241", "0243", "0261", "0262", "0265", "0271", "0272", "0284", "0292",
          "0295", "0299", "1320", "1331", "1332", "1333", "1339", "1342", "1351", "1352",
          "1361", "2143", "2151", "4154", "4155", "4166", "6121", "6154"]

sample:
  require_bzno: true         # 6.7절 1번 기본값

treatment:
  text_columns: [detail]     # 6.7절 2번 기본값: 본문만
  keywords:                  # proposal.md 3.2절의 예시와 저자가 더한 표현(결정 22). 임시값이며 확정 필요
    strict: ['AI\s*agent', 'AI\s*에이전트', 'agentic', '에이전틱', 'LLM\s*agent', '멀티\s*에이전트', 'multi[- ]?agent']
    broad_extra: ['자율\s*에이전트', '에이전트\s*오케스트레이션', 'LangChain', 'LangGraph', 'AutoGen', 'CrewAI',
                  'Claude', '클로드', 'Codex', '코덱스', 'Ollama', '올라마', 'Gemini', '제미나이',
                  'Deep\s*seek', '딥\s*시크', 'sub[- ]?agent', '서브\s*에이전트']
    exclude: ['보험\s*에이전트', '여행\s*에이전[트시]', '부동산\s*에이전트', '스포츠\s*에이전트', '에이전시']
  agent_skills:              # 결정 13, 19
    v31_prefix: "IT-06-012-"
    use_suffix: "활용하기"
    move_to_dev: ["IT-05-0382", "IT-05-0413", "IT-05-0414", "IT-05-0420", "IT-05-0421"]
    move_to_use: ["IT-05-0396"]
  definition: use            # 6.7절 11번 기본값. all | dev | use | keyword | code
  keyword_dictionary: strict

skills:
  adjacent: []               # 6.7절 3번: 자료 점검 뒤 확정
  placebo: []                # 결과를 보기 전에 정한다
  sparse_filter:
    min_rate: 0.001
    min_treated: 30

salary_band_edges: []        # proposal.md 10장 12번: 자료 점검 뒤 확정. 단위는 원
occ_group: keco2             # 6.7절 9번 기본값

estimation:
  seed: 42
  n_folds: 5
  trim: [0.01, 0.99]
  lgbm: {n_estimators: 300, learning_rate: 0.05, num_leaves: 31}
```

- [ ] **5단계: `experiments/paths.py`를 만든다**

`ROOT`는 이 파일의 두 단계 위 폴더이다. `DATA = ROOT/"data"`, `POSTINGS = DATA/"postings"`, `KED = DATA/"ked"`, `OUT = ROOT/"experiments"/"outputs"`, `INTERIM = OUT/"interim"`, `RUNS = OUT/"runs"`, `S0 = OUT/"s0"`, `CONFIG = ROOT/"experiments"/"config.yaml"`. `load_config(path: Path = CONFIG) -> dict`는 YAML을 UTF-8로 읽어 돌려준다. 폴더를 만드는 일은 하지 않는다(쓰는 쪽이 만든다).

- [ ] **6단계: 작업 규칙 파일과 안내 문서를 만든다**

`CLAUDE.md`(최상위)에 담을 내용:
- 연구 개요 두세 문장(생성형 AI 활용 숙련이 있는 공고와 없는 공고의 숙련 묶음 차이를 DML로 추정. 방법은 `proposal.md`, 환경 설계는 설계 문서).
- 두 공간의 경계: 논문 공간은 `paper/exhibits/`만 읽는다. `paper/exhibits/`에는 `experiments/scripts/export_to_paper.py`만 쓴다.
- 원자료(`data/`)는 읽기 전용이다.
- `data/ked/KED5002.txt`의 19번째 항목(주민등록번호)은 읽지 않는다.
- 확정 사항은 `docs/decisions.md`에 먼저 적고 `experiments/config.yaml`에 반영한다.
- 설계 문서의 결정 표와 `proposal.md`가 다르면 설계 문서를 따른다.
- 파이썬 명령은 `./tools/py.ps1`로 실행한다(conda 가상환경 `aabundle`).

`experiments/CLAUDE.md`에 담을 내용:
- 기준값은 `config.yaml`에만 둔다.
- 추정 결과를 보고 하이퍼파라미터를 조정하지 않는다.
- placebo 숙련 목록은 결과를 보기 전에 정한다.
- 교란변수는 같은 기간의 처치 변수로 만들지 않는다.
- 대용량 자료는 필요한 열만 읽는다. 공고 원문 열은 처치 판별에서만 읽는다.
- 시험과 가상 자료의 수치는 결과로 적지 않는다.

`paper/CLAUDE.md`에 담을 내용:
- 수치는 `paper/exhibits/`와 `MANIFEST.csv`에서만 가져온다.
- 인과 표현은 `proposal.md` 9장의 해석 범위를 따른다.
- 참고문헌은 동료심사 학술지 논문만 쓴다.
- 그림은 `paper-figures` 규약을 따른다.

`README.md`: 폴더 구조(설계 문서 4장의 그림), 환경 만들기(7단계의 명령 세 개), 가상환경 이름(`aabundle`)과 위치를 확인하는 명령(`conda env list`), 시험 실행, 자료 점검 실행 명령(과제 8~9의 명령), 실행 순서.

`docs/decisions.md`: 제목과 표 머리(`항목 | 확정값 | 정한 날 | 근거`)만 두고, 설계 문서 6.7절의 13개 항목과 `proposal.md` 10장의 12개 항목을 "미확정" 상태의 행으로 옮겨 적는다. 설계 문서 2장에서 이미 정해진 항목은 확정값과 날짜를 채운다.

- [ ] **7단계: 환경을 만들고 시험을 돌린다**

실행 (순서대로):

```bash
conda env create -f environment.yml
conda run -n aabundle python -m pip install --no-deps --no-build-isolation -e .
conda env export -n aabundle --no-builds > environment.lock.yml
./tools/py.ps1 -m pytest -q
```

`aabundle` 가상환경이 이미 있으면 새로 만들지 말고 저자에게 알린다. `environment.lock.yml`에서 `prefix:`로 시작하는 줄은 지운다(사용자 폴더 경로가 들어 있다). 파일은 UTF-8로 저장한다.
기대: `3 passed`. `conda env list`에 `aabundle`이 있고 그 경로가 구글 드라이브 밖이다. 프로젝트 폴더에 `.venv`가 없다.

`conda run`이 인자를 제대로 넘기지 못하면(작은따옴표나 한글 경로에서 실패하는 경우) `tools/py.ps1`이 가상환경의 `python.exe`를 직접 부르도록 고친다. 이때 가상환경 폴더는 `conda info --base`로 찾고, 가상환경의 `Library\bin`을 `PATH` 앞에 넣는다. 고친 내용은 `README.md`와 과제 보고에 적는다.

- [ ] **8단계: 커밋한다**

```bash
git add environment.yml environment.lock.yml pyproject.toml tools README.md CLAUDE.md docs/decisions.md experiments paper study-notes
git commit -m "실험 공간과 논문 공간의 틀, 환경 설정 추가"
```

---

### 과제 2: 원자료 읽기와 시험용 가상 자료

**파일:**
- 생성: `experiments/src/aabundle/rawio.py`, `experiments/tests/synth.py`, `experiments/tests/conftest.py`, `experiments/tests/test_rawio.py`

**주고받는 것:**
- 만드는 것 (`rawio`):
  - `posting_path(postings_dir: Path, year: int) -> Path` — `pfile_{year}.parquet`
  - `iter_postings(postings_dir: Path, year: int, columns: list[str]) -> Iterator[pd.DataFrame]` — 행 묶음(row group) 하나씩 돌려준다
  - `write_parquet(df: pd.DataFrame, path: Path) -> Path` — 상위 폴더를 만들고 쓴다. `path`가 `data` 폴더 아래면 `ValueError`
  - `read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame`
- 만드는 것 (`synth`): `write_all(root: Path) -> dict` — `root` 아래에 가상 자료 한 벌을 쓰고 경로와 설정을 담은 dict를 돌려준다. 키: `root`(가상 자료의 최상위 폴더), `postings_dir`, `codebook`, `ksic`, `keco`, `ked_firm`, `cfg`. `cfg`는 과제 1의 설정을 읽어 다음만 바꾼 것이다: `data.years`는 `[2025]`, `occupations.keco4`는 `["1332", "0262"]`, `treatment.agent_skills.move_to_dev`는 `["IT-05-0413"]`, `data`의 경로는 `root` 기준 상대 경로. 가상 자료의 폴더 배치는 실제와 같게 한다(`root/data/postings/…`, `root/data/ked/…`).
- 만드는 것 (`conftest`): 픽스처 `synth_data`(`tmp_path`에 `synth.write_all`을 실행한 결과).

**가상 자료의 내용.** 뒤 과제의 시험이 모두 이 12건에 기대므로 아래 그대로 만든다.

공고 12건(`pfile_2025.parquet`, 행 묶음 크기 5건). 열은 `posting_id`, `source`, `post_start`(날짜형), `bzno`, `ksic10`, `firm_emp`, `keco_2025`, `salary_annual_min`, `emp_type`, `edu_level`, `career_type`, `career_years_min`, `region`, `dup_count`, `ai_peripheral`(bool), `detail`, 그리고 숙련 열 12개(`skills_{열}`, `skills_{열}_name`. 열은 `occupation`, `transversal`, `qualification`, `competency`, `soft`, `physical`)이다. 아래 표에 없는 열은 임의의 유효한 값으로 채우되 `salary_annual_min`과 `region`에는 결측을 몇 건 섞는다. `ksic10`은 순서대로 `62010`, `Q86201`, `1220`, `C30`, 결측을 돌려 쓴다.

| posting_id | keco_2025 | bzno | post_start | detail | 숙련 (열: 코드 = 숙련명) |
|---|---|---|---|---|---|
| p01 | 1332 | 1000000001 | 2025-01-10 | `AI 에이전트 개발` | occupation: `IT-05-0398` = RAG 파이프라인 구축하기, `IT-01-0001` = 숙련A |
| p02 | 0262 | 1000000001 | 2025-02-10 | `마케팅 담당` | occupation: `IT-05-0410` = ChatGPT 활용하기, `SM-01-0001` = 숙련B |
| p03 | 0262 | 1000000002 | 2025-04-10 | `보험 에이전트 모집` | occupation: `SM-01-0001` = 숙련B |
| p04 | 0262 | 결측 | 2025-04-11 | `사무 보조` | occupation: `IT-05-0396` = 프롬프트 엔지니어링하기 |
| p05 | 1332 | 1000000003 | 2025-07-01 | `LangChain 경험자` | occupation: `IT-05-0413` = LangChain 프레임워크 활용하기, `IT-05-0410` = ChatGPT 활용하기 |
| p06 | 1332 | 1000000003 | 2025-07-02 | 결측 | 없음 |
| p07 | 9052 | 1000000004 | 2025-07-03 | `AI agent` | occupation: `IT-01-0001` = 숙련A |
| p08 | 결측 | 1000000004 | 2025-07-04 | `일반` | occupation: `SM-01-0001` = 숙련B |
| p09 | 0262 | 1000000005 | 2025-10-01 | `multi-agent 시스템` | transversal: `SS-01-02-0009` = 러시아어 구사하기 |
| p10 | 0262 | 1000000005 | 2025-10-02 | `일반` | soft: `SS-99-99-9999` = 설명 파일에 없는 숙련 |
| p11 | 1332 | 1000000006 | 2025-11-01 | `Claude Code 사용 경험` | occupation: `IT-01-0001` = 숙련A / qualification: `XT-02-009-0000` = 학사 학위 |
| p12 | 1332 | 1000000006 | 2025-11-02 | `일반` | occupation: 코드 `IT-01-0001\|SM-01-0001`, 숙련명은 `숙련A` 하나만 (개수 불일치) |

숙련코드 설명 파일(열은 실제 파일과 같음: `skill_id`, `skill_code_v4`, `skill_code_v31`, `skill_name`, `domain_v4`, `sub_v4`, `detail_v4`, `domain_v31`, `sub_v31`, `detail_v31`, `description`). 9행:

| skill_code_v4 | skill_code_v31 | skill_name | domain_v4 | sub_v4 |
|---|---|---|---|---|
| IT-01-0001 | IT-01-001-0001 | 숙련A | IT·디지털·컴퓨팅 | 소프트웨어 개발 |
| SM-01-0001 | SM-01-001-0001 | 숙련B | 영업·마케팅·대외관계 | 영업 |
| IT-05-0396 | IT-06-012-0028 | 프롬프트 엔지니어링하기 | IT·디지털·컴퓨팅 | AI·머신러닝 시스템 |
| IT-05-0398 | IT-06-012-0030 | RAG 파이프라인 구축하기 | IT·디지털·컴퓨팅 | AI·머신러닝 시스템 |
| IT-05-0410 | IT-06-012-0044 | ChatGPT 활용하기 | IT·디지털·컴퓨팅 | AI·머신러닝 시스템 |
| IT-05-0413 | IT-06-012-0047 | LangChain 프레임워크 활용하기 | IT·디지털·컴퓨팅 | AI·머신러닝 시스템 |
| IT-05-0200 | IT-06-011-0001 | MLOps 구축하기 | IT·디지털·컴퓨팅 | AI·머신러닝 시스템 |
| SS-02-0009 | SS-01-02-0009 | 러시아어 구사하기 | 의사소통 | 외국어 |
| XT-01-0001 | XT-02-009-0000 | 학사 학위 | 자격·인증 | 학위 |

분류표: 실제 파일과 같은 배치(위쪽 머리 3줄, 병합된 칸은 첫 줄만 값이 있음)의 `.xlsx` 두 개. KSIC에는 소분류 `620`, `862`, `012`가 들어가게 세세분류 3행, KECO에는 세분류 `1332`, `0262`, `9052` 3행.

KED 파일: 과제 4에 적는다.

- [ ] **1단계: 실패하는 시험을 쓴다** (`experiments/tests/test_rawio.py`)

```python
import pandas as pd, pytest
from aabundle import rawio

def test_iter_postings_yields_row_groups(synth_data):
    chunks = list(rawio.iter_postings(synth_data["postings_dir"], 2025, ["posting_id", "bzno"]))
    assert [len(c) for c in chunks] == [5, 5, 2]
    assert list(chunks[0].columns) == ["posting_id", "bzno"]

def test_missing_file_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="pfile_2030.parquet"):
        list(rawio.iter_postings(tmp_path, 2030, ["posting_id"]))

def test_missing_column_names_the_column(synth_data):
    with pytest.raises(KeyError, match="no_such_column"):
        list(rawio.iter_postings(synth_data["postings_dir"], 2025, ["posting_id", "no_such_column"]))

def test_write_refuses_raw_data_folder(tmp_path):
    with pytest.raises(ValueError):
        rawio.write_parquet(pd.DataFrame({"a": [1]}), tmp_path / "data" / "x.parquet")

def test_write_then_read_roundtrip(tmp_path):
    p = rawio.write_parquet(pd.DataFrame({"a": [1, 2]}), tmp_path / "outputs" / "x.parquet")
    assert rawio.read_parquet(p)["a"].tolist() == [1, 2]
```

- [ ] **2단계: 시험이 실패하는지 확인한다**

실행: `./tools/py.ps1 -m pytest experiments/tests/test_rawio.py -q`
기대: `synth` 또는 `rawio`가 없어 실패.

- [ ] **3단계: `synth.py`, `conftest.py`, `rawio.py`를 만든다**

`write_parquet`이 거부하는 조건은 "경로의 구성 요소 가운데 이름이 `data`인 폴더가 있음"이다. `iter_postings`는 `pyarrow.parquet.ParquetFile`로 파일의 열 목록을 먼저 확인한다.

- [ ] **4단계: 시험이 통과하는지 확인한다**

실행: `./tools/py.ps1 -m pytest -q`
기대: `8 passed`

- [ ] **5단계: 커밋한다**

```bash
git add experiments/src/aabundle/rawio.py experiments/tests
git commit -m "원자료 읽기 모듈과 시험용 가상 자료 추가"
```

---

### 과제 3: 산업·직종 분류

**파일:**
- 생성: `experiments/src/aabundle/classif.py`, `experiments/tests/test_classif.py`

**주고받는 것:**
- 만드는 것:
  - `KSIC_NOT_IN_TABLE = "분류표에 없음"`
  - `load_ksic_sub(path: Path) -> set[str]` — 소분류 3자리 코드의 집합 (실제 파일은 232개)
  - `load_keco(path: Path) -> pd.DataFrame` — 열 `keco4`, `keco4_name`, `keco2`, `keco2_name` (실제 파일은 495행). 맨 아래 합계 줄(`495개`)은 뺀다
  - `normalize_ksic(s: pd.Series) -> pd.Series` — 5자리 숫자 문자열 또는 결측
  - `ksic_sub(s: pd.Series, valid: set[str]) -> pd.Series` — 소분류 3자리, 분류표에 없으면 `KSIC_NOT_IN_TABLE`, 원래 결측이면 결측
  - `keco_mid(s: pd.Series) -> pd.Series` — 앞 2자리, 결측은 결측

**분류표 파일의 배치.** KSIC는 첫 시트, 위 3줄이 머리, 열 4가 소분류 코드. KECO는 첫 시트, 위 3줄이 머리, 열 2가 중분류 코드, 열 3이 중분류명, 열 6이 세분류 코드, 열 7이 세분류명이며 병합된 칸은 아래로 채워야 한다. 모두 문자열로 읽는다.

- [ ] **1단계: 실패하는 시험을 쓴다**

```python
import pandas as pd
from aabundle import classif

def test_normalize_ksic_formats():
    s = pd.Series(["62010", "Q86201", "1220", "C30", "", None, "q86201", "123456"])
    out = classif.normalize_ksic(s)
    assert out.tolist()[:3] == ["62010", "86201", "01220"]
    assert out[3:].isna().all()

def test_ksic_sub_marks_codes_missing_from_table():
    out = classif.ksic_sub(pd.Series(["62010", "Q86201", "99999", None]), {"620", "862"})
    assert out.tolist()[:3] == ["620", "862", classif.KSIC_NOT_IN_TABLE]
    assert pd.isna(out[3])

def test_keco_mid():
    out = classif.keco_mid(pd.Series(["1332", "0262", None]))
    assert out.tolist()[:2] == ["13", "02"] and pd.isna(out[2])

def test_loaders_read_synthetic_tables(synth_data):
    assert classif.load_ksic_sub(synth_data["ksic"]) == {"620", "862", "012"}
    keco = classif.load_keco(synth_data["keco"])
    assert keco["keco4"].tolist() == ["1332", "0262", "9052"]
    assert set(keco.columns) == {"keco4", "keco4_name", "keco2", "keco2_name"}
```

- [ ] **2단계: 시험이 실패하는지 확인한다** — `./tools/py.ps1 -m pytest experiments/tests/test_classif.py -q`, `classif`가 없어 실패.

- [ ] **3단계: `classif.py`를 만든다**

`normalize_ksic`의 규칙: 대문자 한 글자 + 숫자 5자리면 글자를 뗀다. 숫자 4자리면 앞에 `0`을 붙인다. 숫자 5자리면 그대로 둔다. 그 밖은 결측이다. 소문자는 받아들이지 않는다.

- [ ] **4단계: 시험이 통과하는지 확인한다** — `./tools/py.ps1 -m pytest -q`, 기대 `12 passed`

- [ ] **5단계: 커밋한다** — `git add experiments/src/aabundle/classif.py experiments/tests/test_classif.py`, 메시지 `산업·직종 분류 모듈 추가`

---

### 과제 4: KED 기업 자료 추출

**파일:**
- 생성: `experiments/src/aabundle/ked.py`, `experiments/tests/test_ked.py`
- 수정: `experiments/tests/synth.py` (KED 가상 파일 쓰기)

**주고받는 것:**
- 만드는 것:
  - `N_FIELDS = 48`, `FORBIDDEN_INDEX = 18`
  - `FIELDS = {"ked_size": 5, "ked_venture": 12, "estab_date": 19, "ked_ipo": 20, "info_date": 31, "bzno": 32}` (0부터 센 항목 번호)
  - `extract_ked_firm(path: Path, chunksize: int = 500_000) -> pd.DataFrame` — 열 `bzno`(10자리 문자열, 고유), `ked_estab_year`(`Int16`), `ked_size`, `ked_ipo`, `ked_venture`(문자열 코드, 빈 값은 결측)

**원자료 형식.** `|` 구분, CP949, 머리 줄 없음, 한 줄에 48개 항목(구분자 47개).

**읽는 방법.** (실행 중 수정, 2026-11-12) 파일을 바이트로 한 줄씩 읽는다(줄은 줄바꿈 문자 `\n`으로만 나눈다). 줄 끝의 `\n` 또는 `\r\n`을 뗀 뒤에도 캐리지 리턴 문자(`\r`)가 남아 있는 줄과 `|`의 개수가 정확히 47개가 아닌 줄은 통째로 버린다. 남은 줄은 `|`로 한 번 나누어 `FIELDS`의 여섯 항목만 꺼내 CP949로 바꾼다. 18번 항목은 꺼내지도 문자로 바꾸지도 않는다. `pandas.read_csv`는 쓰지 않는다. 처음 계획은 구분자 수로 거른 줄을 `read_csv`에 넘기는 것이었으나, 줄 안에 `\r`이 하나 있으면 `read_csv`가 그 자리에서 줄을 둘로 나누어 18번 항목이 결과 열로 밀려 들어가는 것이 검토에서 확인되어 바꾸었다. 줄과 항목의 경계를 정하는 동작을 하나만 두기 위해서다. 읽은 줄 수와 버린 줄 수는 결과의 `attrs["n_lines"]`, `attrs["n_dropped_lines"]`에 숫자로만 남긴다.

**가상 KED 파일**(`synth.py`에 추가, CP949로 씀). 모든 행의 18번 항목에는 `SECRET18`을 넣는다.

| 행 | bzno(32) | estab_date(19) | info_date(31) | 비고 |
|---|---|---|---|---|
| 1 | 1000000001 | 20100315 | 20230501 | 정상 |
| 2 | 1000000002 | 19991201 | 20230501 | 정상 |
| 3 | 1000000002 | 19991201 | 20220101 | 같은 사업자번호의 옛 행 (`ked_size`를 다르게) |
| 4 | 1000000003 | (빈 값) | 20230501 | 설립일 없음 |
| 5 | 12345 | 20000101 | 20230501 | 사업자번호가 10자리가 아님 |
| 6 | 1000000005 | 20150101 | 20230501 | 5번 항목 안에 `\|`가 하나 더 있어 49개 항목 |
| 7 | 1000000006 | SECRET18 | 20230501 | 설립일자 자리에 날짜가 아닌 값 |
| 8 | 1000000008 | 20120101 | 20230501 | 40번 항목 안에 `\|`가 하나 더 있어 49개 항목. 읽는 항목은 제자리에 있지만 줄 전체를 버려야 함 |

- [ ] **1단계: 실패하는 시험을 쓴다**

```python
import pandas as pd
from aabundle import ked

def test_forbidden_field_is_never_requested():
    assert ked.FORBIDDEN_INDEX == 18
    assert ked.FORBIDDEN_INDEX not in ked.FIELDS.values()

def test_extract_columns_and_types(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"])
    assert list(out.columns) == ["bzno", "ked_estab_year", "ked_size", "ked_ipo", "ked_venture"]
    assert out["bzno"].is_unique and out["bzno"].str.fullmatch(r"\d{10}").all()

def test_secret_marker_never_appears(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"])
    assert not out.astype(str).apply(lambda c: c.str.contains("SECRET18")).any().any()

def test_rows_with_extra_delimiter_are_dropped(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"])
    assert not {"1000000005", "1000000008"} & set(out["bzno"])

def test_non_date_establishment_becomes_missing(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"]).set_index("bzno")
    assert pd.isna(out.loc["1000000006", "ked_estab_year"])
    assert pd.isna(out.loc["1000000003", "ked_estab_year"])
    assert out.loc["1000000001", "ked_estab_year"] == 2010

def test_latest_info_date_wins_and_bad_bzno_dropped(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"])
    assert set(out["bzno"]) == {"1000000001", "1000000002", "1000000003", "1000000006"}
    # 1000000002는 info_date가 20230501인 행의 ked_size를 가진다 (synth.py에 정한 값과 비교)
```

- [ ] **2단계: 시험이 실패하는지 확인한다** — `./tools/py.ps1 -m pytest experiments/tests/test_ked.py -q`

- [ ] **3단계: `ked.py`를 만든다**

모듈을 불러올 때 `assert FORBIDDEN_INDEX not in FIELDS.values()`를 실행한다. 설립연도는 `estab_date`가 `^\d{8}$`이고 연도가 1800~2026일 때만 값을 둔다. 같은 `bzno`가 여러 행이면 `info_date`가 가장 큰 행을 남긴다. `info_date`와 `estab_date`는 결과에 넣지 않는다.

- [ ] **4단계: 시험이 통과하는지 확인한다** — `./tools/py.ps1 -m pytest -q`, 기대 `18 passed`

- [ ] **5단계: 커밋한다** — 메시지 `KED 기업 자료 추출 모듈 추가 (주민등록번호 항목은 읽지 않음)`

---

### 과제 5: 숙련 정리

**파일:**
- 생성: `experiments/src/aabundle/skills.py`, `experiments/tests/test_skills.py`

**주고받는 것:**
- 만드는 것:
  - `SKILL_COLUMNS = ["occupation", "transversal", "qualification", "competency", "soft", "physical"]`
  - `raw_skill_columns() -> list[str]` — `skills_{c}`와 `skills_{c}_name` 12개
  - `load_codebook(path: Path) -> pd.DataFrame` — 열 `skill_code`(`skill_code_v4`), `skill_code_v31`, `skill_name`, `domain`(`domain_v4`), `sub`(`sub_v4`), `detail`(`detail_v4`). `skill_code`나 `skill_name`이 겹치면 `ValueError`
  - `explode_skills(raw: pd.DataFrame, codebook: pd.DataFrame) -> tuple[pd.DataFrame, dict]` — 입력은 `posting_id`와 원자료 숙련 열 12개. 출력은 `(long, stats)`. `long`의 열은 `posting_id`, `skill_code`, `src`(그 숙련이 나온 열 이름을 `SKILL_COLUMNS` 순서로 `+`로 이은 문자열)이고 `(posting_id, skill_code)`가 고유하다. `stats`의 키는 `n_tokens`, `n_unmatched`, `n_length_mismatch_rows`
  - `attach_hierarchy(long: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame` — `domain`, `sub` 열을 붙인다
  - `drop_codes(long: pd.DataFrame, codes: set[str]) -> pd.DataFrame`
  - `sparse_filter(long: pd.DataFrame, treated_ids: set[str], n_postings: int, min_rate: float, min_treated: int) -> list[str]` — 출현율이 `min_rate` 이상이고 처치 공고 안에서 `min_treated`건 이상 나온 `skill_code`를 정렬해 돌려준다

**최종판 코드를 찾는 규칙.** 원자료의 코드와 숙련명은 `|`로 이어져 있다. 한 숙련에 대해 ① 코드가 설명 파일의 `skill_code`에 있으면 그 코드를 쓴다. ② 없으면 숙련명이 설명 파일의 `skill_name`에 있을 때 그 행의 `skill_code`를 쓴다. ③ 둘 다 아니면 버리고 `n_unmatched`에 센다. 한 공고의 한 열에서 코드 개수와 숙련명 개수가 다르면 그 열은 ①만 적용하고 `n_length_mismatch_rows`에 센다.

- [ ] **1단계: 실패하는 시험을 쓴다**

```python
import pandas as pd, pytest
from aabundle import rawio, skills

@pytest.fixture
def exploded(synth_data):
    cb = skills.load_codebook(synth_data["codebook"])
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025, ["posting_id"] + skills.raw_skill_columns()))
    return cb, *skills.explode_skills(raw, cb)

def codes_of(long, pid):
    return sorted(long.loc[long.posting_id == pid, "skill_code"])

def test_code_in_codebook_is_kept(exploded):
    _, long, _ = exploded
    assert codes_of(long, "p01") == ["IT-01-0001", "IT-05-0398"]

def test_old_format_code_is_mapped_by_name(exploded):
    _, long, _ = exploded
    assert codes_of(long, "p09") == ["SS-02-0009"]
    assert codes_of(long, "p11") == ["IT-01-0001", "XT-01-0001"]
    assert long.loc[(long.posting_id == "p11") & (long.skill_code == "XT-01-0001"), "src"].item() == "qualification"

def test_unknown_skill_is_dropped_and_counted(exploded):
    _, long, stats = exploded
    assert codes_of(long, "p10") == [] and stats["n_unmatched"] == 1

def test_length_mismatch_falls_back_to_codes(exploded):
    _, long, stats = exploded
    assert codes_of(long, "p12") == ["IT-01-0001", "SM-01-0001"]
    assert stats["n_length_mismatch_rows"] == 1

def test_posting_without_skills_has_no_rows(exploded):
    _, long, _ = exploded
    assert codes_of(long, "p06") == []
    assert not long.duplicated(["posting_id", "skill_code"]).any()

def test_same_skill_in_two_columns_is_one_row():
    cb = pd.DataFrame({"skill_code": ["IT-01-0001"], "skill_code_v31": ["x"], "skill_name": ["숙련A"],
                       "domain": ["d"], "sub": ["s"], "detail": ["t"]})
    raw = pd.DataFrame({c: [None] for c in skills.raw_skill_columns()} | {"posting_id": ["q1"]})
    raw["skills_occupation"], raw["skills_occupation_name"] = "IT-01-0001", "숙련A"
    raw["skills_soft"], raw["skills_soft_name"] = "IT-01-0001", "숙련A"
    long, _ = skills.explode_skills(raw, cb)
    assert len(long) == 1 and long["src"].item() == "occupation+soft"

def test_hierarchy_and_drop(exploded):
    cb, long, _ = exploded
    h = skills.attach_hierarchy(long, cb)
    assert h.loc[h.skill_code == "SS-02-0009", "domain"].iloc[0] == "의사소통"
    assert "IT-05-0410" not in set(skills.drop_codes(long, {"IT-05-0410"})["skill_code"])

def test_sparse_filter_thresholds(exploded):
    _, long, _ = exploded
    kept = skills.sparse_filter(long, treated_ids={"p02", "p05"}, n_postings=10, min_rate=0.2, min_treated=1)
    assert kept == ["IT-05-0410", "SM-01-0001"]   # 0410: 2건 모두 처치. SM-01-0001: 4건 중 p02가 처치. IT-01-0001: 4건이지만 처치 공고에 없음
    assert skills.sparse_filter(long, treated_ids=set(), n_postings=10, min_rate=0.0, min_treated=1) == []

def test_codebook_with_duplicate_name_is_rejected(tmp_path, synth_data):
    df = pd.read_csv(synth_data["codebook"], dtype=str, encoding="utf-8-sig")
    pd.concat([df, df.iloc[[0]].assign(skill_code_v4="ZZ-00-0000")]).to_csv(tmp_path / "cb.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError):
        skills.load_codebook(tmp_path / "cb.csv")
```

- [ ] **2단계: 시험이 실패하는지 확인한다** — `./tools/py.ps1 -m pytest experiments/tests/test_skills.py -q`

- [ ] **3단계: `skills.py`를 만든다**

설명 파일은 `utf-8-sig`, 모든 열을 문자열로 읽는다. `explode_skills`는 실제 자료에서 한 번에 1만 7천 건씩 1,200번쯤 불리므로 행 단위 반복문 대신 `str.split`과 `explode`로 만든다.

- [ ] **4단계: 시험이 통과하는지 확인한다** — `./tools/py.ps1 -m pytest -q`, 기대 `27 passed`

- [ ] **5단계: 커밋한다** — 메시지 `숙련 정리 모듈 추가 (여섯 열을 숙련명으로 최종판 코드에 대응)`

---

### 과제 6: 처치 변수

**파일:**
- 생성: `experiments/src/aabundle/treatment.py`, `experiments/tests/test_treatment.py`

**주고받는 것:**
- 쓰는 것: 과제 5의 `explode_skills`가 만든 `long`(`posting_id`, `skill_code`), `load_codebook`의 결과
- 만드는 것:
  - `flag_treatment(text: pd.Series, include: list[str], exclude: Sequence[str] = ()) -> pd.Series` — `proposal.md` 부록 A의 코드를 그대로 옮긴다(아래)
  - `keyword_lists(cfg: dict) -> dict` — `{"strict": [...], "broad": strict + broad_extra, "exclude": [...]}`
  - `pattern_counts(text: pd.Series, patterns: list[str], exclude: Sequence[str] = ()) -> dict[str, int]` — 제외 패턴을 지운 뒤 패턴 하나하나에 걸리는 공고 수. 대소문자를 구분하지 않는다. 한 공고가 여러 패턴에 걸리면 각각 센다. 걸린 공고가 없는 패턴도 0으로 넣는다
  - `agent_skill_codes(codebook: pd.DataFrame, cfg: dict) -> dict[str, set[str]]` — `{"dev": {...}, "use": {...}}`
  - `skill_flags(long: pd.DataFrame, codes: dict[str, set[str]], posting_ids: pd.Series) -> pd.DataFrame` — 열 `posting_id`, `has_dev`, `has_use`(bool). `posting_ids`의 모든 공고가 한 행씩 나온다
  - `build_treat(posting_ids: pd.Series, text: pd.Series, long: pd.DataFrame, codebook: pd.DataFrame, cfg: dict) -> pd.DataFrame` — 열 `posting_id`, `kw_strict`, `kw_broad`, `has_dev`, `has_use`, `treat_code`(`has_dev | has_use`), `treat_all`(`kw_strict | treat_code`). 모두 bool
  - `DEFINITIONS = ("all", "dev", "use", "keyword", "code")`
  - `select_arms(treat: pd.DataFrame, definition: str, keyword_col: str = "kw_strict") -> pd.Series` — `Int8`. 처치 공고 1, 비교 공고 0, 어느 쪽도 아니면 결측. 모르는 정의면 `ValueError`

**부록 A에서 그대로 옮기는 함수:**

```python
def flag_treatment(text, include, exclude=()):
    """제외 패턴을 지운 뒤 포함 사전에 걸리는 공고를 1로 표시"""
    if exclude:
        text = text.str.replace("|".join(exclude), " ", case=False, regex=True)
    return text.str.contains("|".join(include), case=False, regex=True, na=False).astype(int)
```

**AI 에이전트 숙련코드를 고르는 규칙.** 설명 파일에서 `skill_code_v31`이 `v31_prefix`로 시작하는 숙련이 대상이다. 숙련명이 `use_suffix`로 끝나면 `use`, 아니면 `dev`이다. 그 뒤 `move_to_dev`의 코드는 `dev`로, `move_to_use`의 코드는 `use`로 옮긴다. 예외 목록에 대상이 아닌 코드가 있으면 `ValueError`를 낸다.

**처치 정의별 공고 고르기** (설계 문서 6.2절의 표). 비교 공고는 모든 정의에서 `~keyword_col & ~has_dev & ~has_use`이다. 처치 공고는 `all`이 `keyword_col | has_dev | has_use`, `dev`가 `has_dev`, `use`가 `has_use`, `keyword`가 `keyword_col`, `code`가 `has_dev | has_use`이다.

- [ ] **1단계: 실패하는 시험을 쓴다**

```python
import pandas as pd, pytest
from aabundle import rawio, skills, treatment

@pytest.fixture
def treat(synth_data):
    cfg = synth_data["cfg"]
    cb = skills.load_codebook(synth_data["codebook"])
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025,
                                        ["posting_id", "detail"] + skills.raw_skill_columns()))
    long, _ = skills.explode_skills(raw, cb)
    return treatment.build_treat(raw["posting_id"], raw["detail"], long, cb, cfg).set_index("posting_id")

def ids(mask):
    return sorted(mask[mask].index)

def test_agent_codes_split_with_exceptions(synth_data):
    cb = skills.load_codebook(synth_data["codebook"])
    codes = treatment.agent_skill_codes(cb, synth_data["cfg"])
    assert codes["use"] == {"IT-05-0410", "IT-05-0396"}       # 이름 규칙 + 옮긴 것
    assert codes["dev"] == {"IT-05-0398", "IT-05-0413"}       # 나머지 + 옮긴 것
    assert "IT-05-0200" not in codes["use"] | codes["dev"]    # 같은 중분류의 다른 세부 분류

def test_exception_outside_agent_codes_is_rejected(synth_data):
    cfg = synth_data["cfg"]
    cfg["treatment"]["agent_skills"]["move_to_use"] = ["IT-01-0001"]
    with pytest.raises(ValueError):
        treatment.agent_skill_codes(skills.load_codebook(synth_data["codebook"]), cfg)

def test_keyword_flags(treat):
    assert ids(treat.kw_strict) == ["p01", "p07", "p09"]      # p03은 제외 패턴, p06은 본문 결측
    assert ids(treat.kw_broad) == ["p01", "p05", "p07", "p09", "p11"]   # p11은 저자가 더한 표현(Claude)
    s = pd.Series(["Claude Code 사용", "claude와 Gemini", "보험 에이전트", None])
    assert treatment.pattern_counts(s, ["Claude", "Gemini", "에이전트"], ["보험\\s*에이전트"]) == \
        {"Claude": 2, "Gemini": 1, "에이전트": 0}

def test_skill_code_flags(treat):
    assert ids(treat.has_use) == ["p02", "p04", "p05"]
    assert ids(treat.has_dev) == ["p01", "p05"]
    assert ids(treat.treat_code) == ["p01", "p02", "p04", "p05"]
    assert ids(treat.treat_all) == ["p01", "p02", "p04", "p05", "p07", "p09"]
    assert len(treat) == 12

def test_select_arms_use_definition(treat):
    arm = treatment.select_arms(treat.reset_index(), "use")
    arm.index = treat.index
    assert ids(arm == 1) == ["p02", "p04", "p05"]
    assert ids(arm == 0) == ["p03", "p06", "p08", "p10", "p11", "p12"]
    assert ids(arm.isna()) == ["p01", "p07", "p09"]           # 개발 숙련만 있거나 키워드에만 걸린 공고

def test_select_arms_comparison_is_same_for_every_definition(treat):
    t = treat.reset_index()
    comps = [set(t.posting_id[(treatment.select_arms(t, d) == 0).to_numpy()]) for d in treatment.DEFINITIONS]
    assert all(c == comps[0] for c in comps)
    with pytest.raises(ValueError):
        treatment.select_arms(t, "unknown")
```

- [ ] **2단계: 시험이 실패하는지 확인한다** — `./tools/py.ps1 -m pytest experiments/tests/test_treatment.py -q`

- [ ] **3단계: `treatment.py`를 만든다**

`text_columns`가 여러 개면 열을 공백으로 이어 한 문자열로 만든 뒤 `flag_treatment`에 넣는다(이 계획에서는 `detail` 하나).

- [ ] **4단계: 시험이 통과하는지 확인한다** — `./tools/py.ps1 -m pytest -q`, 기대 `33 passed`

- [ ] **5단계: 커밋한다** — 메시지 `처치 변수 모듈 추가 (키워드, AI 에이전트 숙련코드, 개발과 생성형 AI 활용 구분)`

---

### 과제 7: 공고 기본 자료와 제외 규칙

**파일:**
- 생성: `experiments/src/aabundle/sample.py`, `experiments/tests/test_sample.py`

**주고받는 것:**
- 쓰는 것: `classif.normalize_ksic`, `classif.ksic_sub`, `classif.keco_mid`, `ked.extract_ked_firm`의 결과
- 만드는 것:
  - `RAW_COLUMNS = ["posting_id", "source", "post_start", "bzno", "ksic10", "firm_emp", "keco_2025", "salary_annual_min", "emp_type", "edu_level", "career_type", "career_years_min", "region", "dup_count", "ai_peripheral"]`
  - `to_sido(region: pd.Series) -> pd.Series`
  - `salary_band(salary: pd.Series, edges: list[float]) -> pd.Series`
  - `build_posting_base(raw: pd.DataFrame, ked_firm: pd.DataFrame, ksic_valid: set[str], cfg: dict) -> pd.DataFrame` — 입력과 행 수가 같다. 열: `posting_id`, `source`, `year`(`post_start`의 연도), `month`, `quarter`(`"2025Q3"` 형식), `bzno`, `ksic5`, `ksic_sub`, `firm_emp_log`(`log1p`), `keco4`, `keco2`, `career_type`, `career_years_min`, `edu_level`, `salary_annual_min`, `salary_band`, `emp_type`, `sido`, `dup_count`, `ai_peripheral`, `ked_matched`(bool), `firm_age`(`year − ked_estab_year`, 음수면 결측), `ked_size`, `ked_ipo`, `ked_venture`, `in_period`, `in_occ`, `has_bzno`
  - `EXCLUSION_STEPS = ("in_period", "in_occ", "has_bzno", "has_skill")`
  - `apply_exclusions(base: pd.DataFrame, cfg: dict, has_skill: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]` — 순서대로 뺀 뒤 남은 공고와, 열이 `year`, `step`, `n_before`, `n_dropped`, `n_after`인 건수 표를 돌려준다. `cfg["sample"]["require_bzno"]`가 거짓이면 `has_bzno` 단계를 건너뛰고, `has_skill`이 `None`이면 `has_skill` 단계를 건너뛴다

**값을 정하는 규칙.**
- `to_sido`: `region`을 쉼표로 나눈 첫 부분의 첫 낱말을 본다. 끝의 `전체`를 떼고 아래 표로 바꾼다. 표에 없으면 `기타`, 원래 결측이면 결측이다.

  | 결과 | 받아들이는 표기 |
  |---|---|
  | 서울 | 서울, 서울특별시 |
  | 부산, 대구, 인천, 광주, 대전, 울산 | 두 글자 이름과 `○○광역시` |
  | 세종 | 세종, 세종특별자치시 |
  | 경기 | 경기, 경기도 |
  | 강원 | 강원, 강원도, 강원특별자치도 |
  | 충북, 충남, 전남, 경북, 경남 | 두 글자 이름과 `충청북도` 같은 긴 이름 |
  | 전북 | 전북, 전라북도, 전북특별자치도 |
  | 제주 | 제주, 제주도, 제주특별자치도 |

- `salary_band`: 결측이면 `미기재`. `edges`가 비어 있으면 값이 있는 공고는 모두 `기재`. `edges`가 있으면 `[e0, e1)` 식의 구간 이름(`~e0`, `e0~e1`, …, `e_last~`)을 준다. 단위는 원이다.
- `in_period`: `post_start`가 `cfg["period"]`의 `start` 이상 `end` 이하.
- `in_occ`: `keco_2025`가 `cfg["occupations"]["keco4"]`에 있음(결측은 거짓).
- `has_bzno`: `bzno`가 10자리 숫자.
- KED 결합은 `bzno`로 왼쪽 결합하고, 결합 뒤 행 수가 달라지면 `RuntimeError`를 낸다.

- [ ] **1단계: 실패하는 시험을 쓴다**

```python
import pandas as pd, pytest
from aabundle import classif, ked, rawio, sample

@pytest.fixture
def base(synth_data):
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025, sample.RAW_COLUMNS))
    return sample.build_posting_base(raw, ked.extract_ked_firm(synth_data["ked_firm"]),
                                     classif.load_ksic_sub(synth_data["ksic"]), synth_data["cfg"]).set_index("posting_id")

def test_to_sido_variants():
    s = pd.Series(["서울 마포구, 서울전체, 영등포구", "충청남도 천안시", "세종특별자치시", "서울전체,",
                   "광산구", "아시아·중동", "전남광주 어딘가", None])
    out = sample.to_sido(s)
    assert out.tolist()[:7] == ["서울", "충남", "세종", "서울", "기타", "기타", "기타"]
    assert pd.isna(out[7])

def test_salary_band():
    s = pd.Series([None, 25_000_000, 40_000_000])
    assert sample.salary_band(s, []).tolist() == ["미기재", "기재", "기재"]
    out = sample.salary_band(s, [30_000_000])
    assert out[0] == "미기재" and out[1] != out[2]

def test_base_keeps_every_row_and_flags(base):
    assert len(base) == 12
    assert sorted(base.index[~base.in_occ]) == ["p07", "p08"]
    assert sorted(base.index[~base.has_bzno]) == ["p04"]
    assert base.in_period.all()
    assert base.loc["p05", "quarter"] == "2025Q3" and base.loc["p01", "keco2"] == "13"

def test_ked_join(base):
    assert base.loc["p01", "ked_matched"] and base.loc["p01", "firm_age"] == 15   # 2025 - 2010
    assert not base.loc["p09", "ked_matched"]          # 1000000005는 KED 가상 파일에서 버려진 행
    assert pd.isna(base.loc["p05", "firm_age"])         # 결합됐지만 설립일 없음

def test_ksic_columns(base):
    assert set(base["ksic_sub"].dropna()) <= {"620", "862", "012", classif.KSIC_NOT_IN_TABLE}

def test_exclusions_in_order(base, synth_data):
    has_skill = pd.Series(True, index=base.index)
    has_skill[["p06", "p10"]] = False
    kept, counts = sample.apply_exclusions(base.reset_index(), synth_data["cfg"], has_skill.reset_index(drop=True))
    assert sorted(kept.posting_id) == ["p01", "p02", "p03", "p05", "p09", "p11", "p12"]
    c = counts.set_index("step")
    assert c.loc["in_occ", "n_dropped"] == 2 and c.loc["has_bzno", "n_dropped"] == 1 and c.loc["has_skill", "n_dropped"] == 2
    assert c.loc["has_skill", "n_after"] == 7

def test_period_filter(synth_data):
    cfg = synth_data["cfg"]
    cfg["period"] = {"start": "2025-07-01", "end": "2025-12-31", "unit": "quarter"}
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025, sample.RAW_COLUMNS))
    b = sample.build_posting_base(raw, ked.extract_ked_firm(synth_data["ked_firm"]), set(), cfg)
    assert (~b.in_period).sum() == 4                     # p01~p04
```

- [ ] **2단계: 시험이 실패하는지 확인한다** — `./tools/py.ps1 -m pytest experiments/tests/test_sample.py -q`

- [ ] **3단계: `sample.py`를 만든다**

- [ ] **4단계: 시험이 통과하는지 확인한다** — `./tools/py.ps1 -m pytest -q`, 기대 `40 passed`

- [ ] **5단계: 커밋한다** — 메시지 `공고 기본 자료 구성과 제외 규칙 모듈 추가`

---

### 과제 8: 자료 점검 1단계 — 원자료를 훑어 중간 자료 만들기

**파일:**
- 생성: `experiments/scripts/s00_data_audit.py`, `experiments/tests/test_s00_scan.py`

**주고받는 것:**
- 쓰는 것: 과제 2~7의 함수 전부
- 만드는 것: `scan(cfg: dict, root: Path, out_dir: Path) -> dict` — 자료 경로는 `root`를 기준으로 `cfg["data"]`에서 읽는다. `out_dir`에 다음 파일을 쓰고 경로 dict를 돌려준다.

| 파일 | 한 행 | 열 |
|---|---|---|
| `occ_year_all.parquet` | 연도 × 세분류 (모든 공고) | `year`, `keco4`(결측은 `"NA"`), `n`, `n_skill`, `n_use`, `n_dev` |
| `base.parquet` | 공고 (`in_period & in_occ`인 것만) | `build_posting_base`의 열 전부 + `n_skills`(확인된 숙련 수) |
| `treat.parquet` | 공고 (`base.parquet`과 같은 공고) | `build_treat`의 열 전부 |
| `skills_long.parquet` | 공고 × 숙련 (`base.parquet`의 공고만) | `posting_id`, `skill_code`, `src` |
| `scan_stats.parquet` | 연도 | `year`, `n_raw`, `n_in_period`, `n_in_occ`, `n_tokens`, `n_unmatched`, `n_length_mismatch_rows` |
| `kw_pattern_hits.parquet` | 연도 × 키워드 패턴 (`base.parquet`의 공고만) | `year`, `dictionary`(`strict` 또는 `broad_extra`), `pattern`, `n` |

**처리 순서** (행 묶음마다):
1. `sample.RAW_COLUMNS`와 숙련 열 12개를 읽는다. 공고 원문은 아직 읽지 않는다.
2. 모든 공고에 대해 `explode_skills`와 `skill_flags`를 실행하고, 연도 × 세분류 집계에 더한다.
3. `build_posting_base`를 실행해 `in_period & in_occ`인 공고만 남긴다.
4. 남은 공고가 있으면 같은 행 묶음에서 `cfg["treatment"]["text_columns"]`의 열만 다시 읽어 남은 공고의 것만 골라 `build_treat`를 실행한다. 이어서 `kw_broad`가 참인 공고의 원문에만 `pattern_counts`를 실행해 패턴별 공고 수를 연도별로 더한다(넓은 사전에 걸리지 않은 공고는 어느 패턴에도 걸리지 않으므로 다시 읽지 않는다). 원문은 저장하지 않는다.
5. 묶음별 결과를 모아 두었다가 연도가 끝나면 이어 붙인다.

KED 추출(`extract_ked_firm`)과 분류표, 설명 파일 읽기는 시작할 때 한 번만 한다. 진행 상황은 연도가 끝날 때마다 한 줄 출력한다. 모든 연도를 합친 뒤 `posting_id`가 겹치면 `RuntimeError`를 낸다.

명령줄: `python experiments/scripts/s00_data_audit.py scan`은 `paths.load_config()`, `paths.ROOT`, `paths.S0`으로 `scan`을 부른다. 스크립트 맨 위에서 `experiments/` 폴더를 `sys.path`에 넣어 `paths`를 불러온다.

- [ ] **1단계: 실패하는 시험을 쓴다**

```python
import importlib.util, pandas as pd, paths

def load_script():
    spec = importlib.util.spec_from_file_location("s00", paths.ROOT / "experiments" / "scripts" / "s00_data_audit.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_scan_outputs(synth_data, tmp_path):
    out = load_script().scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    base = pd.read_parquet(out["base"]); treat = pd.read_parquet(out["treat"])
    long = pd.read_parquet(out["skills_long"]); occ = pd.read_parquet(out["occ_year_all"])
    stats = pd.read_parquet(out["scan_stats"]).set_index("year")

    assert sorted(base.posting_id) == [f"p{i:02d}" for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12)]
    assert sorted(treat.posting_id) == sorted(base.posting_id)
    assert set(long.posting_id) <= set(base.posting_id)
    assert base.set_index("posting_id").loc["p12", "n_skills"] == 2
    assert base.set_index("posting_id").loc["p10", "n_skills"] == 0

    o = occ.set_index("keco4")
    assert o.loc["0262", "n"] == 5 and o.loc["0262", "n_use"] == 2       # p02, p04
    assert o.loc["1332", "n_use"] == 1 and o.loc["1332", "n_dev"] == 2   # p05 / p01, p05
    assert o.loc["9052", "n"] == 1 and o.loc["NA", "n"] == 1
    assert stats.loc[2025, "n_raw"] == 12 and stats.loc[2025, "n_in_occ"] == 10
    assert stats.loc[2025, "n_unmatched"] == 1

    kw = pd.read_parquet(out["kw_pattern_hits"]).set_index("pattern")
    assert len(kw) == len(synth_data["cfg"]["treatment"]["keywords"]["strict"]) + \
        len(synth_data["cfg"]["treatment"]["keywords"]["broad_extra"])
    assert kw.loc["Claude", "n"] == 1 and kw.loc["Claude", "dictionary"] == "broad_extra"   # p11
    assert kw.loc["AI\\s*agent", "n"] == 0 and kw.loc["Gemini", "n"] == 0                  # p07은 분석 대상 직종이 아니다

def test_scan_never_stores_posting_text(synth_data, tmp_path):
    out = load_script().scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    for p in out.values():
        df = pd.read_parquet(p)
        assert "detail" not in df.columns and "title" not in df.columns
        assert not df.astype(str).apply(lambda c: c.str.contains("에이전트 개발")).any().any()
```

- [ ] **2단계: 시험이 실패하는지 확인한다** — `./tools/py.ps1 -m pytest experiments/tests/test_s00_scan.py -q`

- [ ] **3단계: `s00_data_audit.py`의 `scan`과 명령줄 처리를 만든다**

- [ ] **4단계: 시험이 통과하는지 확인한다** — `./tools/py.ps1 -m pytest -q`, 기대 `42 passed`

- [ ] **5단계: 커밋한다** — 메시지 `자료 점검 스크립트 1단계 추가 (원자료를 훑어 중간 자료 작성)`

---

### 과제 9: 자료 점검 2단계 — 집계표와 보고서

**파일:**
- 수정: `experiments/scripts/s00_data_audit.py`
- 생성: `experiments/tests/test_s00_report.py`

**주고받는 것:**
- 쓰는 것: 과제 8의 중간 자료 6개, `sample.apply_exclusions`, `treatment.select_arms`, `skills.sparse_filter`, `skills.attach_hierarchy`, `classif.load_keco`
- 만드는 것:
  - `build_tables(cfg: dict, root: Path, s0_dir: Path) -> dict[str, pd.DataFrame]` — 아래 표 이름을 키로 하는 dict. 각 표를 `s0_dir/"tables"/"{이름}.csv"`(`utf-8-sig`)로도 쓴다
  - `time_outcome_model(base: pd.DataFrame, treat: pd.DataFrame, long: pd.DataFrame, cfg: dict) -> dict` — 키 `skill_code`, `n_train`, `n_features`, `seconds`, `n_skills_kept`, `est_total_hours`
  - `write_report(tables: dict, timing: dict, cfg: dict, report_path: Path, commit: str) -> Path`
  - 명령줄: `report`(중간 자료로 표와 보고서 작성), `all`(`scan` 다음 `report`)

**집계표.** 번호는 설계 문서 6.6절의 항목 번호이다. 별도 표시가 없으면 대상은 `base.parquet`의 공고(분석 기간 안, 분석 대상 세분류 28개)이다. "처치 정의"는 `kw_strict`, `kw_broad`, `has_dev`, `has_use`, `treat_code`, `treat_all` 여섯 열이다.

| 표 이름 | 내용 |
|---|---|
| `t01_exclusions` | 연도 × 제외 단계별 `n_before`, `n_dropped`, `n_after`. 첫 두 단계는 `scan_stats`에서, 나머지는 `apply_exclusions`에서 |
| `t02a_treated_by_period` | 기간(연, 반기, 분기 각각) × 처치 정의별 공고 수. 제외 규칙을 모두 적용한 공고 기준. 열은 `unit`(`year`, `half`, `quarter`), `period`(`2025`, `2025H2`, `2025Q3` 형식), `n`, 처치 정의 여섯 열. 공고가 있는 기간은 처치 공고가 0건이어도 행이 있다 |
| `t02b_treated_by_occ` | 세분류(이름 포함)와 중분류 × 처치 정의별 공고 수, 전체 공고 수 |
| `t03a_skills_per_posting` | 숙련이 있는 공고의 공고당 숙련 수 분위수(최소, 25%, 50%, 75%, 90%, 99%, 최대)와 평균 |
| `t03b_sparse_filter_grid` | `min_rate` ∈ {0.0005, 0.001, 0.002} × `min_treated` ∈ {10, 30, 50}별로 남는 숙련 수. 처치 공고는 `select_arms(…, "use") == 1`. AI 에이전트 숙련코드는 먼저 뺀다 |
| `t03c_skills_by_source_column` | 숙련 열(`src`를 `+`로 나눈 것)별 공고 × 숙련 건수, 대분류별 건수 |
| `t03d_match_stats` | 연도별 `n_tokens`, `n_unmatched`, `n_length_mismatch_rows` |
| `t04_bzno_ked_by_treatment` | `has_use` 여부별 공고 수, 사업자번호 보유 비율, KED 결합 비율(사업자번호가 있는 공고 가운데). 사업자번호 제외를 적용하기 전 공고 기준 |
| `t05_source` | 연도 × 수집 사이트별 공고 수, `has_use` 비율, 숙련 보유 비율 |
| `t06a_overlap` | `kw_strict` × `treat_code` × `ai_peripheral`의 교차 건수, `kw_strict` × `has_use` × `has_dev`의 교차 건수 |
| `t06b_agent_skill_by_year` | AI 에이전트 숙련코드 각각의 연도별 공고 수. 열은 `skill_code`, `skill_name`, `kind`(`개발` 또는 `생성형 AI 활용`), 연도별 열. 한 건도 없는 숙련도 0으로 넣는다(실제 자료에서는 54행) |
| `t06c_keyword_pattern_hits` | 키워드 패턴별 공고 수. 열은 `dictionary`, `pattern`, 연도별 열, `total`. 사업자번호와 숙련 제외를 적용하기 전의 `base.parquet` 공고 기준. 저자가 키워드 사전을 확정할 때 패턴별로 걸리는 공고가 얼마나 되는지 보는 데 쓴다 |
| `t08_2026_monthly` | 2026년 월별 공고 수와 숙련 보유 비율 |
| `t09a_by_ksic_sub`, `t09b_ksic_not_in_table` | 산업 소분류별 공고 수와 `has_use` 공고 수 / 소분류표에 없는 5자리 코드의 건수(많은 순 30개) |
| `t10a_use_by_occ_year` | 분석 대상 세분류 × 연도별 `has_use` 공고 수와 비율. 제외 규칙을 모두 적용한 공고 기준 |
| `t10b_occ_list_check` | `occ_year_all`에서 `n_use`의 전체 기간 합이 `min_use_postings` 이상인 세분류 목록과 `cfg["occupations"]["keco4"]`의 비교. 열은 `keco4`, `n_use`, `status`(`양쪽에 있음`, `설정에만 있음`, `자료에만 있음`) |
| `t11_salary` | 연도별 연봉 기재 비율과, 기재된 공고의 `salary_annual_min` 분위수(10%, 25%, 50%, 75%, 90%). 연봉 구간(`proposal.md` 10장 12번)을 정하는 데 쓴다 |
| `t12_sido_other` | `sido`별 공고 수 (`기타`와 결측 포함) |


**학습 시간 측정(6.6절 7번).** 제외 규칙을 모두 적용한 공고 가운데 `select_arms(…, cfg의 definition)`이 0 또는 1인 공고를 쓴다. 설명 변수는 `firm_emp_log`, `ked_size`, `ked_ipo`, `ked_venture`, `firm_age`, `ksic_sub`, `keco4`, `career_type`, `career_years_min`, `edu_level`, `salary_band`, `emp_type`, `sido`, `quarter`이며 문자열 열은 `category`로 바꾼다. 결과변수는 희소 필터(설정의 기준)를 통과한 숙련 가운데 가장 많이 나온 것의 보유 여부이다. 희소 필터를 통과한 숙련이 없으면 전체에서 가장 많이 나온 숙련을 쓴다. `GroupKFold(n_folds)`를 `bzno`로 나눈 첫 번째 분할의 학습 쪽에서 비교 공고만 골라 `LGBMClassifier(**cfg["estimation"]["lgbm"], random_state=seed, verbose=-1)`을 한 번 학습하고 걸린 시간을 잰다. `est_total_hours = seconds × n_folds × n_skills_kept / 3600`. 처치 공고가 없거나 비교 공고가 `n_folds`개 기업보다 적으면 측정을 건너뛰고 `seconds`를 결측으로 둔다.

**보고서 `docs/s0-data-audit.md`의 구성.** 맨 위에 실행한 날, 코드 시점(git 커밋), 분석 기간, 분석 대상 세분류 수, 키워드 사전이 임시값이라는 안내를 적는다. 그 아래 `## 1.`부터 `## 10.`까지 설계 문서 6.6절의 항목 순서로 절을 두고(절 제목은 6.6절의 항목 내용을 그대로 쓴다), `## 11. 연봉과 근무지`를 더한다. 각 절은 해당 표를 마크다운 표로 넣되 30행을 넘으면 위 30행만 넣고 전체 표의 CSV 경로를 적는다. 해석 문장은 쓰지 않는다. 숫자만 싣는다.

- [ ] **1단계: 실패하는 시험을 쓴다**

```python
import pandas as pd
from test_s00_scan import load_script

def run(synth_data, tmp_path):
    s00 = load_script()
    s00.scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    return s00, s00.build_tables(synth_data["cfg"], synth_data["root"], tmp_path / "s0")

def test_exclusion_table_matches_fixture(synth_data, tmp_path):
    _, t = run(synth_data, tmp_path)
    c = t["t01_exclusions"].query("year == 2025").set_index("step")
    assert c.loc["in_occ", "n_dropped"] == 2 and c.loc["has_bzno", "n_dropped"] == 1
    assert c.loc["has_skill", "n_dropped"] == 2 and c.loc["has_skill", "n_after"] == 7

def test_treated_counts_by_period(synth_data, tmp_path):
    _, t = run(synth_data, tmp_path)
    q = t["t02a_treated_by_period"].query("unit == 'quarter'").set_index("period")
    assert q.loc["2025Q1", "has_use"] == 1 and q.loc["2025Q3", "has_use"] == 1   # p02 / p05 (p04는 사업자번호 없음)
    assert q.loc["2025Q4", "has_use"] == 0                                        # 처치 공고가 없는 기간도 0으로 나온다

def test_occupation_list_check(synth_data, tmp_path):
    cfg = synth_data["cfg"]; cfg["occupations"]["min_use_postings"] = 2
    s00 = load_script(); s00.scan(cfg, synth_data["root"], tmp_path / "s0")
    chk = s00.build_tables(cfg, synth_data["root"], tmp_path / "s0")["t10b_occ_list_check"].set_index("keco4")
    assert chk.loc["0262", "status"] == "양쪽에 있음" and chk.loc["1332", "status"] == "설정에만 있음"

def test_agent_skill_table_lists_every_code(synth_data, tmp_path):
    _, t = run(synth_data, tmp_path)
    a = t["t06b_agent_skill_by_year"].set_index("skill_code")
    assert len(a) == 4 and a.loc["IT-05-0410", "kind"] == "생성형 AI 활용" and a.loc["IT-05-0413", "kind"] == "개발"
    k = t["t06c_keyword_pattern_hits"].set_index("pattern")
    assert k.loc["Claude", "total"] == 1 and k.loc["Gemini", "total"] == 0

def test_timing_skips_when_too_small(synth_data, tmp_path):
    s00, _ = run(synth_data, tmp_path)
    d = tmp_path / "s0"
    timing = s00.time_outcome_model(pd.read_parquet(d / "base.parquet"), pd.read_parquet(d / "treat.parquet"),
                                    pd.read_parquet(d / "skills_long.parquet"), synth_data["cfg"])
    assert set(timing) == {"skill_code", "n_train", "n_features", "seconds", "n_skills_kept", "est_total_hours"}

def test_report_has_all_sections_and_no_text(synth_data, tmp_path):
    s00, t = run(synth_data, tmp_path)
    timing = {"skill_code": None, "n_train": 0, "n_features": 14, "seconds": None, "n_skills_kept": 0, "est_total_hours": None}
    p = s00.write_report(t, timing, synth_data["cfg"], tmp_path / "report.md", commit="testcommit")
    text = p.read_text(encoding="utf-8")
    assert all(f"## {i}." in text for i in range(1, 12))
    assert "testcommit" in text and "임시" in text and "에이전트 개발" not in text
```

- [ ] **2단계: 시험이 실패하는지 확인한다** — `./tools/py.ps1 -m pytest experiments/tests/test_s00_report.py -q`

- [ ] **3단계: `build_tables`, `time_outcome_model`, `write_report`와 명령줄의 `report`, `all`을 만든다**

- [ ] **4단계: 시험이 통과하는지 확인한다** — `./tools/py.ps1 -m pytest -q`, 기대 `48 passed`

- [ ] **5단계: 커밋한다** — 메시지 `자료 점검 스크립트 2단계 추가 (집계표, 학습 시간 측정, 보고서)`

---

### 과제 10: 실제 자료 확인과 자료 점검 실행

**파일:**
- 생성: `experiments/tests/test_realdata.py`, `docs/s0-data-audit.md`(스크립트가 씀)
- 수정: `README.md`(실행 시간과 산출물 위치), `docs/decisions.md`(분석 대상 세분류 확인 결과)

**주고받는 것:**
- 쓰는 것: 과제 1~9 전부. 이 과제만 `data/`의 실제 자료를 읽는다.

- [ ] **1단계: 실제 자료 확인 시험을 쓴다.** 모든 시험에 `@pytest.mark.realdata`를 붙이고 `paths.DATA`가 없으면 건너뛴다.

```python
def test_codebook_size():
    cb = skills.load_codebook(paths.ROOT / cfg["data"]["codebook"])
    assert len(cb) == 8550 and cb["domain"].nunique() == 17 and cb["sub"].nunique() == 65

def test_agent_skill_codes_match_decisions():
    codes = treatment.agent_skill_codes(cb, cfg)
    assert len(codes["use"]) == 11 and len(codes["dev"]) == 43
    assert codes["use"] | codes["dev"] == {f"IT-05-{i:04d}" for i in range(376, 430)}
    assert {"IT-05-0410", "IT-05-0396"} <= codes["use"] and "IT-05-0413" in codes["dev"]

def test_classification_tables():
    assert len(classif.load_ksic_sub(paths.ROOT / cfg["data"]["ksic"])) == 232
    keco = classif.load_keco(paths.ROOT / cfg["data"]["keco"])
    assert len(keco) == 495 and set(cfg["occupations"]["keco4"]) <= set(keco["keco4"])

def test_first_row_group_of_each_year_is_readable():
    # 연도마다 첫 행 묶음을 읽어 build_posting_base와 explode_skills가 예외 없이 끝나는지 확인
    ...
```

마지막 시험의 본문: 연도마다 `next(rawio.iter_postings(paths.POSTINGS, year, sample.RAW_COLUMNS + skills.raw_skill_columns()))`으로 첫 묶음을 받아 `explode_skills`를 실행하고 `stats["n_unmatched"] / stats["n_tokens"] < 0.01`을 확인한다. `build_posting_base`는 빈 KED 표(열만 있는 것)로 실행해 행 수가 같은지 확인한다.

- [ ] **2단계: 실제 자료 확인 시험을 돌린다**

실행: `./tools/py.ps1 -m pytest -m realdata -q`
기대: `4 passed`. 실패하면 설계 문서 3장의 수치와 어긋난 것이므로 원인을 찾아 저자에게 알리고 멈춘다. 시험의 기대값을 자료에 맞춰 바꾸지 않는다.

- [ ] **3단계: 자료 점검을 실제 자료로 실행한다**

실행: `./tools/py.ps1 experiments/scripts/s00_data_audit.py all` (백그라운드로 실행. 공고 2,000만 건의 원문을 읽으므로 한 시간 넘게 걸릴 수 있다)
기대: `experiments/outputs/s0/`에 중간 자료 6개와 `tables/`가 생기고 `docs/s0-data-audit.md`가 써진다.

- [ ] **4단계: 결과를 설계 문서의 값과 대조한다**

| 확인할 것 | 설계 문서의 값 |
|---|---|
| `scan_stats`의 `n_raw` 합 | 20,006,136 |
| `occ_year_all`의 `n_use` 합 | 10,827 (2021년부터 46, 30, 418, 1,042, 2,813, 6,478) |
| `occ_year_all`의 `n_dev` 합 | 18,346 |
| `t10b_occ_list_check` | 28개 모두 "양쪽에 있음", 다른 상태의 행 없음 |
| `base.parquet`에서 숙련과 사업자번호가 있는 공고 | 약 395만 |
| 그중 `has_use`인 공고 | 8,572 |

어긋나는 값이 있으면 보고서를 고치지 말고, 어긋난 값과 원인을 저자에게 알린다. 설계 문서의 값은 코드 열에서 정규식으로 센 것이고 이 스크립트는 숙련명으로도 찾으므로 `n_use`, `n_dev`가 조금 늘 수 있다. 그 경우 늘어난 건수와 숙련 열을 함께 알린다.

- [ ] **5단계: `docs/s0-data-audit.md`에 공고 원문이나 사업자번호가 들어가지 않았는지 확인한다**

실행: 보고서에서 10자리 숫자와 `detail` 낱말을 찾는다(`Grep`으로 `\b\d{10}\b`).
기대: 일치하는 줄 없음.

- [ ] **6단계: 문서를 고치고 커밋한다**

`README.md`에 자료 점검의 실제 실행 시간과 산출물 위치를 적는다. `docs/decisions.md`의 "분석 대상 세분류" 행에 목록 확인 결과와 날짜를 적는다.

```bash
git add experiments/tests/test_realdata.py docs/s0-data-audit.md docs/decisions.md README.md
git commit -m "자료 점검 보고서 추가"
```

- [ ] **7단계: 저자에게 보고한다**

보고서 경로와, 보고서를 보고 정해야 하는 항목을 알린다: 설계 문서 6.7절의 13개 항목과 `proposal.md` 10장의 항목(분석 기간과 기간 단위, 키워드 사전, 희소 필터, 인접 숙련 목록, placebo 숙련 목록, 연봉 구간, 직종군). 여기서 멈추고 두 번째 계획은 저자가 확정 사항을 정한 뒤 작성한다.
