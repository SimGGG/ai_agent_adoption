# 생성형 AI 활용 숙련과 숙련 묶음 연구

한국 채용 공고 자료로, 생성형 AI 활용 숙련이 들어 있는 공고(처치 공고)와 그렇지 않은 공고(비교 공고)의 숙련 묶음 차이를 DML(이중 머신러닝)로 추정하는 연구의 작업 폴더이다. 분석 방법은 `proposal.md`에, 작업 환경의 설계는 `docs/superpowers/specs/2026-11-12-research-infra-design.md`에 있다. 작업 규칙은 `CLAUDE.md`에 있다.

## 폴더 구조

```text
14_ai-agent-adoption-skill-bundle/
├── CLAUDE.md                 # 공통 규칙
├── README.md                 # 폴더 설명과 실행 순서
├── proposal.md               # 연구계획서
├── environment.yml           # conda 가상환경의 패키지 목록
├── environment.lock.yml      # 설치된 패키지 버전
├── pyproject.toml            # aabundle 모듈 설치 정보, 시험 설정
├── .gitignore
├── data/                     # 원자료. 읽기 전용, git 제외
├── docs/
│   ├── superpowers/specs/    # 설계 문서
│   ├── superpowers/plans/    # 실행 계획
│   ├── decisions.md          # 확정 사항 기록
│   └── s0-data-audit.md      # 자료 점검 보고서
├── experiments/              # 실험 공간
│   ├── CLAUDE.md
│   ├── config.yaml           # 확정 사항을 담은 설정
│   ├── paths.py              # 자료와 산출 경로
│   ├── src/aabundle/         # 공용 모듈
│   ├── scripts/              # 단계별 실행 스크립트
│   ├── notebooks/            # 결과 열람용
│   ├── tests/
│   └── outputs/              # git 제외
│       ├── interim/          # 중간 자료
│       ├── s0/               # 자료 점검의 중간 자료와 집계표(tables/)
│       └── runs/             # 실행별 결과
├── paper/                    # 논문 공간
│   ├── CLAUDE.md
│   ├── exhibits/             # 실험 공간에서 내보낸 확정 표와 그림, 목록 파일
│   ├── manuscript/           # LaTeX 원고
│   ├── references/           # 참고문헌 (.bib)
│   ├── reviews/              # 원고 검토 기록
│   └── submissions/          # 투고본
└── study-notes/              # 학습노트
```

`docs/s0-data-audit.md`, `docs/superpowers/plans/`, `experiments/outputs/`는 해당 작업을 하면 생긴다.

## 환경 만들기

conda 가상환경 이름은 `aabundle`이다. 저장소 최상위 폴더에서 PowerShell로 다음 네 줄(환경 변수 설정 한 줄과 명령 세 줄)을 순서대로 실행한다. 이 형태(`conda run ... python -m pip`)는 PowerShell에서만 확인했고, Git Bash에서는 `conda run`이 가상환경이 아닌 다른 파이썬을 찾는 경우가 있어 쓰지 않는다. `aabundle`이 이미 있으면 새로 만들지 말고 연구자에게 알린다.

```powershell
$env:PYTHONNOUSERSITE = "1"
conda env create -f environment.yml
conda run -n aabundle python -m pip install --no-deps --no-build-isolation -e .
conda env export -n aabundle --no-builds > environment.lock.yml
```

첫 줄은 사용자 폴더(`AppData\Roaming\Python`)에 설치된 패키지가 가상환경의 패키지를 가려 설치와 시험이 실패하는 것을 막는다(`tools/py.ps1`에는 이미 들어 있다). 마지막 줄(`conda env export`)은 PowerShell 5.1에서 파일을 UTF-16으로 저장하므로, 파일을 다시 쓸 때는 UTF-8(BOM 없음)로 저장한다.

마지막 줄로 만든 `environment.lock.yml`은 다음과 같이 고친다.

- `prefix:`로 시작하는 줄을 지운다(사용자 폴더 경로가 들어 있다).
- `channels:`의 `defaults` 줄을 지운다(conda가 암묵적으로 더한 것이며 `environment.yml`에 없다).
- `pip:` 블록을 지운다. 이 블록에는 `aabundle`(편집 가능 설치한 이 프로젝트)이 들어 있어 PyPI에서 찾으면 실패한다. `scipy`와 `pyarrow`도 이 블록에 나오지만, 실제로는 conda-forge에서 설치된 패키지이다(conda-forge가 이 두 패키지에 pip 형식의 정보 파일을 함께 넣어서 conda가 pip 설치로 잘못 표시한다). 따라서 지운 뒤 `scipy=버전`, `pyarrow=버전`을 일반 패키지 줄로 옮겨 적는다.
- 파일은 UTF-8로 저장한다.

가상환경의 위치는 다음 명령으로 확인한다. 위치는 conda의 기본 위치(`anaconda3\envs\aabundle`)이므로 구글 드라이브 밖이다.

```bash
conda env list
```

## 파이썬 실행

모든 파이썬 명령은 저장소 최상위 폴더에서 `./tools/py.ps1`로 실행한다. 이 스크립트는 `conda run -n aabundle --no-capture-output python -X utf8` 뒤에 받은 인자를 붙여 실행한다.

```powershell
./tools/py.ps1 -m pytest -q
```

`tools/py.ps1`은 `PYTHONUTF8=1`과 `PYTHONNOUSERSITE=1`을 설정한 뒤 실행한다. 두 번째 설정은 사용자 폴더에 설치된 같은 이름의 패키지가 가상환경의 패키지를 가리는 것을 막는다.

PowerShell이 스크립트 실행을 막거나 Git Bash에서 실행할 때는 가상환경의 `python.exe`를 직접 부른다. Git Bash에서는 `conda run`이 가상환경이 아닌 다른 파이썬을 부르는 일이 있어 `conda run` 형태를 쓰지 않는다.

```bash
PYTHONNOUSERSITE=1 PYTHONUTF8=1 "$(conda info --base)/envs/aabundle/python.exe" -X utf8 -m pytest -q
```

## 시험 실행

```powershell
./tools/py.ps1 -m pytest -q
```

`data/`의 실제 자료를 읽는 시험(`realdata` 표시)은 기본으로 돌지 않는다. 실행하려면 `-m realdata`를 준다.

## 자료 점검 실행

자료 점검(S0)은 원자료로 공고 수, 처치 공고 수, 숙련 출현율 등을 집계해 `docs/s0-data-audit.md`에 적는 단계이다. 저장소 최상위 폴더에서 다음 명령으로 실행한다.

```powershell
./tools/py.ps1 experiments/scripts/s00_data_audit.py all
```

- `scan`: 연도별 공고 파일 여섯 개, 숙련 설명 파일, 산업 분류표(`KSIC10.xlsx`), KED 기업 자료를 한 번 읽어 중간 자료 여덟 파일(`base`, `treat`, `skills_long`, `occ_year_all`, `scan_stats`, `kw_pattern_hits`, `post_start_missing`, `sido_other_words`, 모두 parquet)과 1단계 기록 `scan_meta.json`(실행한 시각, git 커밋, 커밋하지 않은 변경이 있었는지, 아래 표의 `scan`을 다시 해야 하는 설정 항목의 값)을 `experiments/outputs/s0/`에 쓴다.
- `report`: 중간 자료와 함께 숙련 설명 파일, 직종 분류표(`KECO2025.xlsx`), KED 기업 자료(`ked.extract_ked_firm`으로 줄 수와 기업 수를 센다)를 읽어 집계표(CSV)를 `experiments/outputs/s0/tables/`에 쓰고, 결과모형(공고의 특성으로 그 숙련이 공고에 들어 있을 확률을 예측하는 모형) 한 번의 학습 시간을 잰 뒤 `docs/s0-data-audit.md`를 쓴다. 먼저 `scan_meta.json`의 설정 값과 지금 `config.yaml`을 비교해, 다른 항목이 있으면 그 항목을 모두 적고 멈춘다.
- `all`: `scan` 다음에 `report`를 실행한다.

두 명령 모두 일을 시작하기 전에 `config.yaml`의 값이 쓸 수 있는 형태인지 확인한다(`aabundle/validate.py`). 예를 들어 `occupations.keco4`의 코드를 따옴표 없이 적으면 멈춘다.

`config.yaml`을 고친 뒤 다시 실행할 명령은 고친 항목에 따라 다르다.

| 다시 실행할 명령 | 설정 항목 |
|---|---|
| `scan`부터(`all`) | `data.years`, `data.codebook`, `data.ksic`, `data.ked_firm`, `period.start`, `period.end`, `occupations.keco4`, `treatment.text_columns`, `treatment.keywords`, `treatment.agent_skills`, `salary_band_edges` |
| `report`만 | `data.keco`, `period.unit`, `occupations.min_use_postings`, `sample.require_bzno`, `treatment.definition`, `treatment.keyword_dictionary`, `skills`, `estimation` |

나머지 항목(`occ_group` 등)은 자료 점검에서 쓰지 않는다.

2026-11-12에 실제 자료로 `all`을 실행한 시간은 약 87분이었다(`scan` 약 83분, `report` 약 4분). `scan`에서 연도별 공고 파일을 읽는 데 걸린 시간은 2021년 819초, 2022년 1,021초, 2023년 819초, 2024년 867초, 2025년 772초, 2026년 691초이다. 집계표만 다시 만들 때는 `report`만 실행한다(중간 자료가 있어야 한다).

## 실행 순서

1. 환경 만들기와 시험 실행
2. 자료 점검(S0)을 실행하고 보고서를 읽는다.
3. 보고서를 보고 확정 사항을 `docs/decisions.md`에 적고 `experiments/config.yaml`에 반영한다.
4. 단계별 스크립트(S1~S12)를 `experiments/scripts/`에서 순서대로 실행한다(두 번째 계획에서 추가. 지금은 `s00_data_audit.py`만 있다).
5. 확정된 표와 그림을 `experiments/scripts/export_to_paper.py`로 `paper/exhibits/`에 내보낸다(두 번째 계획에서 추가).
