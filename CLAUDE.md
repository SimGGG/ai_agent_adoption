# 작업 규칙 (공통)

## 연구 개요

생성형 AI 활용 숙련이 들어 있거나 본문이 생성형 AI 활용 키워드 사전에 걸린 채용 공고(처치 공고)와 그렇지 않은 공고(비교 공고)가 요구하는 숙련 묶음이 얼마나 다른지를 DML(이중 머신러닝, 처치 변수와 결과 변수 각각을 기계학습으로 예측해 교란변수의 영향을 걷어낸 뒤 처치의 효과를 추정하는 방법)로 추정한다. 교란변수는 처치 여부와 결과 양쪽에 영향을 주어, 통제하지 않으면 처치의 효과로 잘못 읽히는 변수이다. 연구의 목적은 생성형 AI 도입에 따라 숙련 조합에 유의미한 변화가 있었는지를 같은 기업 안에서 보는 것이므로, 주 분석은 같은 기업 안의 비교이고 결과는 직종별, 산업별로 나누어 낸다(`docs/decisions.md` 결정 47, 48). 분석 방법은 `proposal.md`에, 작업 환경의 설계는 `docs/superpowers/specs/2026-11-12-research-infra-design.md`(설계 문서)에 있다. 분석 기간은 2023년 1월부터 2026년 8월까지이고 분석 대상은 직종 세분류 20개이다.

## 규칙

1. 두 공간의 경계: 논문 공간(`paper/`)은 `paper/exhibits/`만 읽는다. `paper/exhibits/`에는 `experiments/scripts/export_to_paper.py`만 쓴다. 논문 공간에서 `experiments/outputs/`를 직접 참조하지 않는다.
2. 원자료(`data/`)는 읽기 전용이다. 어떤 코드도 `data/` 아래에 쓰지 않는다. 외부에서 받은 공개 자료는 사람이 직접 내려받아 `data/external/`에 둔다.
3. `data/ked/KED5002.txt`의 19번째 항목(0부터 세면 18번)은 개인사업자의 주민등록번호이다. 이 항목은 읽지 않는다. 화면 출력, 로그, 시험 자료에도 실제 값을 넣지 않는다.
4. 확정 사항은 `docs/decisions.md`에 먼저 적고 `experiments/config.yaml`에 반영한다.
5. 설계 문서의 결정 표와 `proposal.md`가 다르면 설계 문서를 따른다.
6. 파이썬 명령은 저장소 최상위 폴더에서 PowerShell로 `./tools/py.ps1`을 실행한다(conda 가상환경 `aabundle`). 예: `./tools/py.ps1 -m pytest -q`. PowerShell이 스크립트 실행을 막거나 Git Bash에서 실행할 때는 가상환경의 `python.exe`를 직접 부른다: `PYTHONNOUSERSITE=1 PYTHONUTF8=1 "$(conda info --base)/envs/aabundle/python.exe" -X utf8 -m pytest -q`. Git Bash에서 `conda run`은 가상환경이 아닌 다른 파이썬을 부르므로 쓰지 않는다.
7. 공고 원문 열(`detail`, `title`)은 처치 판별과 본문 글자 수 계산에서만 읽고, 중간 자료와 보고서에 원문을 저장하지 않는다. 글자 수는 저장한다.
8. 시험과 가상 자료에서 나온 수치는 연구 결과로 적지 않는다.
9. 커밋은 한국어로 쓴다. `data/`, `experiments/outputs/`, `test.ipynb`는 커밋하지 않는다.

## 문서 작성 원칙

- 영어 개념을 줄이거나 직역해서 새 용어를 만들지 않는다.
- 은유와 과장된 표현을 쓰지 않는다. 개념은 그 기능을 풀어서 적는다.
- 표준 학술·기술 용어는 쓸 수 있되, 문서에서 처음 나올 때 한 문장으로 정의한다.
- 설계 문서의 용어(처치 공고, 비교 공고, 숙련 묶음, 가상 자료)를 그대로 쓴다.
