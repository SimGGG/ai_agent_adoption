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

conda 가상환경 이름은 `aabundle`이다. 저장소 최상위 폴더에서 다음 세 명령을 순서대로 실행한다. `aabundle`이 이미 있으면 새로 만들지 말고 연구자에게 알린다.

```bash
conda env create -f environment.yml
conda run -n aabundle python -m pip install --no-deps --no-build-isolation -e .
conda env export -n aabundle --no-builds > environment.lock.yml
```

명령은 PowerShell에서 실행한다. Git Bash에서는 `conda run`이 가상환경이 아닌 다른 파이썬을 찾는 경우가 있었다. 사용자 폴더(`AppData\Roaming\Python`)에 설치된 패키지가 가상환경의 패키지를 가려 시험이 실패하는 것을 막으려면 설치 명령 앞에 `$env:PYTHONNOUSERSITE = "1"`을 실행한다(`tools/py.ps1`에는 이미 들어 있다).

세 번째 명령으로 만든 `environment.lock.yml`에서 `prefix:`로 시작하는 줄(사용자 폴더 경로가 들어 있다)은 지우고, 파일은 UTF-8로 저장한다.

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

PowerShell이 스크립트 실행을 막거나 Bash에서 실행할 때는 풀어서 쓴다.

```bash
conda run -n aabundle --no-capture-output python -X utf8 -m pytest -q
```

Git Bash에서는 `conda run`이 가상환경이 아닌 다른 파이썬을 부르는 일이 있었으므로, PowerShell에서 실행하는 것을 기본으로 한다. Git Bash에서 꼭 실행해야 하면 가상환경의 `python.exe`를 직접 부른다.

```bash
PYTHONNOUSERSITE=1 PYTHONUTF8=1 "$(conda info --base)/envs/aabundle/python.exe" -X utf8 -m pytest -q
```

## 시험 실행

```powershell
./tools/py.ps1 -m pytest -q
```

`data/`의 실제 자료를 읽는 시험(`realdata` 표시)은 기본으로 돌지 않는다. 실행하려면 `-m realdata`를 준다.

## 자료 점검 실행

자료 점검(S0)은 원자료로 공고 수, 처치 공고 수, 숙련 출현율 등을 집계해 `docs/s0-data-audit.md`에 적는 단계이다. 실행 명령은 과제 8~9에서 스크립트를 만든 뒤 이 절에 적는다.

## 실행 순서

1. 환경 만들기와 시험 실행
2. 자료 점검(S0)을 실행하고 보고서를 읽는다.
3. 보고서를 보고 확정 사항을 `docs/decisions.md`에 적고 `experiments/config.yaml`에 반영한다.
4. 단계별 스크립트(S1~S12)를 `experiments/scripts/`에서 순서대로 실행한다.
5. 확정된 표와 그림을 `experiments/scripts/export_to_paper.py`로 `paper/exhibits/`에 내보낸다.
