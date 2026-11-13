# 연구 수행 환경 2차 구축: 확정 사항 반영과 자료 점검 재실행 — 실행 계획

> **작업자(에이전트)에게:** 이 계획은 superpowers:subagent-driven-development(권장) 또는 superpowers:executing-plans 스킬로 과제 단위로 실행한다. 각 단계는 체크박스(`- [ ]`)로 진행 상황을 표시한다.

**목표:** 저자가 2026-11-13에 정한 확정 사항(`docs/decisions.md`)을 설계 문서, 설정, 자료 처리 모듈, 자료 점검 스크립트에 반영하고, 실제 자료로 자료 점검을 다시 실행해 새 보고서와 두 개의 후보 표(인접 숙련, placebo 숙련)를 낸다.

**구조:** 확정 사항은 모두 `experiments/config.yaml`의 값으로 들어가고, 모듈은 그 값을 읽어 동작한다. 게시일 규칙, 기간 이름, 시도 합치기, 산업코드 처리는 `sample.py`와 `classif.py`의 작은 함수로 두고 `build_posting_base`가 부른다. 산업별 AI 노출 지수는 새 모듈 `confounders.py`가 외부 자료 세 개를 이어 산업 소분류별 값으로 만든다. 자료 점검 스크립트는 지금의 두 단계(원자료를 훑는 `scan`, 표와 보고서를 만드는 `report`) 구조를 유지한다.

**기술:** Python 3.12, conda 가상환경 `aabundle`, pandas, pyarrow, LightGBM, PyYAML, openpyxl, pytest. 새 패키지는 없다.

**설계 문서:** `docs/superpowers/specs/2026-11-12-research-infra-design.md` (이하 "설계 문서"). 과제 1에서 아래 "이번에 반영하는 확정 사항"대로 고친다. 작업자는 이 계획과 설계 문서를 함께 읽는다.

**범위:** 확정 사항의 반영과 자료 점검 재실행까지이다. 설계 문서 10장의 구축 순서 3~5번(추정 모듈 `dml`·`network`·`runlog`, 단계 스크립트 `s01`~`s12`와 내보내기, 논문 원고 틀)은 저자가 이 계획의 결과(인접 숙련 후보 표, placebo 숙련 후보 표, 산업별 AI 노출 지수의 결합 비율)를 보고 남은 두 목록을 정한 뒤 세 번째 계획으로 작성한다.

## 이번에 반영하는 확정 사항

근거는 `docs/decisions.md`(2026-11-13 기록)이다. 번호는 설계 문서 2장의 결정 표에 새로 넣을 번호이다.

| # | 결정 |
|---|---|
| 23 | 분석 기간은 2023-01-01부터 2026-08-31까지이다. 2021~2022년 공고 파일은 읽지 않는다. |
| 24 | 기간은 연 단위의 세 기간으로 나눈다: 초기(2023~2024년), 1 도입기(2025년), 2 도입기(2026년). |
| 25 | 게시일은 `post_start`를 쓰고, 없으면 `post_end`를 쓴다. 둘 다 없는 공고는 제외한다. |
| 26 | 결정 20의 28개 세분류 가운데 관리직(`0122`)과 영업·판매직(`6121`, `6154`)을 분석 대상에서 뺀다. 분석 대상 세분류는 25개, 그 세분류가 속한 중분류는 4개(`02`, `13`, `21`, `41`)이다. |
| 27 | 직종군별 추정의 직종군은 중분류 4개이고, 직종군별 추정에 필요한 처치 공고 수의 기준은 100건이다. |
| 28 | 소분류표에 없는 산업코드는 결측으로 둔다. 공고는 빼지 않는다. |
| 29 | 근무지의 `광주`, `전남`, `전남광주`는 한 범주 `광주·전남`으로 본다. |
| 30 | `proposal.md` 3.4절의 "산업별 평균 AI 도입률" 대신 Felten, Raj, Seamans의 산업별 AI 노출 지수(AIIE) 가운데 언어 모형 판을 쓴다(자료: https://github.com/AIOE-Data/AIOE 의 `Language Modeling AIOE and AIIE.xlsx`, 미국 산업분류 NAICS 4자리 250개 산업). 직전 기간으로 계산하는 변수는 없고, 변수 산정용 기간도 두지 않는다. |
| 31 | 연봉 구간의 경계는 2,600만, 3,000만, 3,500만, 4,000만 원이다(5개 구간과 결측 범주). |
| 32 | 희소 숙련 필터는 출현율 0.1% 이상, 처치 공고 30건 이상이다. 키워드 사전은 `config.yaml`의 엄격한 사전 7개 표현, 넓은 사전에 더한 18개 표현, 제외 표현 5개로 확정한다. |
| 33 | 설계 문서 6.7절의 1, 2, 4, 5, 6, 11, 12, 14, 17번은 기본값으로 확정한다. 3번(인접 숙련 목록)과 placebo 숙련 목록은 이 계획이 만드는 후보 표를 본 뒤 정한다. |
| 34 | 수집 사이트를 교란변수에 넣는다(결정 14를 바꿈). |
| 35 | 공고 본문의 글자 수를 교란변수에 넣는다. 본문은 저장하지 않고 글자 수만 저장한다. 글자 수를 뺀 추정도 함께 낸다. |
| 36 | 생성형 AI 활용 표현의 키워드 사전(`treatment.keywords.use`)을 둔다. 이 사전에 걸린 공고는 비교 공고에서 뺀다. 이 사전만으로 정한 처치(`use_keyword`)와 숙련코드와 합친 처치(`use_any`)를 점검용 처치 정의로 더한다. 주 분석의 처치 정의는 그대로 생성형 AI 활용 숙련(`use`)이다. |
| 37 | 세 번째 계획에 다음 분석을 넣는다: 같은 기업·같은 연도 안의 비교, 처치 공고를 처음 내기 전 시기의 공고로 하는 점검, 직종·기간별 보급 정도에 따른 효과, 변수 역할 표, 단순 차이와 추정치의 병렬 제시, 교란변수 묶음별 제외 비교, 기간 비교 시 직종 구성 고정. 이 계획은 그 가운데 앞의 두 분석이 가능한지 보여 주는 건수 표만 만든다. |

## 실행 전에 필요한 것 (저자 확인)

1. **외부 자료를 두는 폴더.** 과제 6에서 공개 자료 두 개를 `data/external/` 아래에 내려받아 둔다. 기존 원자료는 건드리지 않는다. 코드는 지금처럼 `data/` 아래에 쓰지 않고, 내려받기는 작업을 지휘하는 쪽이 명령으로 한 번 한다.
   - `data/external/aioe/Language Modeling AIOE and AIIE.xlsx` (https://github.com/AIOE-Data/AIOE)
   - `data/external/census/2017_NAICS_to_ISIC_4.xlsx` (미국 통계국의 NAICS 2017과 국제표준산업분류 ISIC Rev.4 연계표, https://www.census.gov/naics/concordances/2017_NAICS_to_ISIC_4.xlsx)
2. **한국표준산업분류 10차와 ISIC Rev.4를 잇는 표.** 따로 구할 수 있는 연계표가 없어(저자 확인, 2026-11-13) 분류 이름을 보고 판단해 만들었다: `experiments/crosswalks/ksic10_sub_to_isic4.csv`. 소분류 232개마다 한 줄이고, `isic` 열에는 그 소분류에 해당한다고 본 ISIC 코드의 앞자리(2~4자리)를 `;`로 이어 적었다(그 글자로 시작하는 ISIC 4자리 코드 모두에 잇는다는 뜻). 판단이 갈릴 만한 줄은 `note` 열에 이유를 적었다. 2026-11-13에 실제 파일로 확인한 결과 232개 가운데 221개 소분류에 지수 값이 붙는다. 값이 붙지 않는 11개는 농업 3개와 수렵, 양식어업, 외무 및 국방 행정, 사회보장 행정, 가구 내 고용활동, 자가 소비 생산 활동 2개, 국제 및 외국기관이다(미국 쪽 자료에 해당 산업이 없다). 이 표는 git으로 관리하며 저자가 줄 단위로 고칠 수 있다.

## 공통 제약

- 원자료(`data/`)는 읽기만 한다. 어떤 코드도 `data/` 아래에 쓰지 않는다. `data/external/`의 파일은 사람이 직접 내려받아 두는 것이며 코드가 만들지 않는다.
- `data/ked/KED5002.txt`의 19번째 항목(0부터 세면 18번)은 개인사업자의 주민등록번호이다. 어떤 코드도 이 항목을 읽지 않는다. 화면 출력, 로그, 시험 자료에도 실제 값을 넣지 않는다. 이 계획은 `ked.py`를 고치지 않는다.
- 공고 원문 열(`detail`, `title`)은 처치 판별과 본문 글자 수 계산에서만 읽고, 중간 자료와 보고서에 원문을 저장하지 않는다. 글자 수는 저장한다(결정 35).
- 기준값은 `experiments/config.yaml`에만 둔다. 모듈과 스크립트에 기준값을 직접 적지 않는다.
- 시험과 가상 자료에서 나온 수치는 연구 결과로 적지 않는다. `docs/s0-data-audit.md`에는 실제 자료로 실행한 수치만 들어간다.
- 실제 자료는 과제 7에서만 읽는다. 과제 1~6(3A 포함)의 작업자는 `data/`와 `experiments/outputs/`를 읽지 않고 가상 자료(`experiments/tests/synth.py`)로만 시험한다. 과제 6에서 외부 자료 세 파일의 형식을 확인하는 단계만 예외이다.
- 파이썬 명령은 저장소 최상위 폴더에서 실행한다. PowerShell: `./tools/py.ps1 -m pytest -q`. Git Bash: `PYTHONNOUSERSITE=1 PYTHONUTF8=1 "$(conda info --base)/envs/aabundle/python.exe" -X utf8 -m pytest -q`. Git Bash에서 `conda run`은 쓰지 않는다. 시험은 `-W error`를 붙여 경고 없이 통과해야 한다.
- 작업은 `master`에서 만든 `infra-2` 브랜치로 한다. 별도 작업 폴더(git worktree)는 만들지 않는다. 푸시는 하지 않는다.
- 과제가 끝날 때마다 커밋한다. `data/`, `experiments/outputs/`, `test.ipynb`는 커밋하지 않는다. 커밋 메시지는 한국어로 쓰고 끝에 `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`를 붙인다.
- 사용자에게 보이는 한국어 문서(설계 문서, `README.md`, `CLAUDE.md`, 보고서의 표 설명, 주석)는 사용자의 용어 원칙을 따른다: 영어 개념을 줄여 새 용어를 만들지 않는다. 은유를 쓰지 않는다. 개념은 기능을 풀어서 적는다. 표준 학술 용어는 문서에서 처음 나올 때 한 문장으로 정의한다.
- 이미 있는 시험의 기대값을 바꿀 때는 바꾼 단언과 이유를 과제 보고에 하나씩 적는다. 설정의 기본값이 바뀌어서 달라지는 기대값만 바꾼다.
- 이 계획의 시험 코드에 나오는 고정물 이름(`base`, `scan_out`, `tables`, `synth_scan`)은 해당 시험 파일에 이미 있는 고정물을 가리킨다. 이름이 다르면 그 파일의 이름을 쓴다. 단언의 내용은 바꾸지 않는다.
- 입력 파일이나 필요한 열이 없으면 무엇이 없는지 적은 예외를 내고 바로 멈춘다. 결합 전후로 행 수가 달라지면 멈춘다.

## 검토 때 특히 볼 곳

설계 문서와 확정 사항이 직접 말하지 않지만 실제 자료에서 만날 입력이다. 각 줄의 시험은 해당 과제에 넣었다.

1. **날짜로 읽을 수 없거나 아주 먼 미래인 마감일.** `post_end`에 `9999-12-31` 같은 값이나 날짜가 아닌 글자가 있을 수 있다. 읽을 수 없으면 결측으로 보고 멈추지 않는다. 읽히지만 분석 기간 뒤인 날짜는 분석 기간 단계에서 빠진다. (과제 3)
2. **게시일은 없고 마감일이 분석 기간 밖인 공고.** 마감일로 게시일을 대신했는데 그 날짜가 2026-08-31 뒤이거나 2023-01-01 앞이면 빠진다. 몇 건이 살아나고 몇 건이 빠지는지를 표에서 따로 센다. (과제 4, 5)
3. **형식이 고르지 않은 외부 자료의 산업 코드.** 미국 통계국 연계표의 ISIC 코드는 `111`처럼 적혀 있다(`0111`의 뜻). AIIE의 산업 코드 250개는 대부분 4자리 산업(`611100`)이지만 3자리 수준 6개(`531000`)와 여러 4자리 산업을 묶은 14개(`3250A1`, 설명의 괄호 안에 `3251, 3252, 3253, and 3259 only`처럼 구성 산업이 적혀 있음)가 섞여 있다. 앞 네 글자만 자르면 부동산업, 화물 운송업, 화학·금속 제조업 등이 이어지지 않는다(2026-11-13 확인: 이렇게 하면 값이 붙는 소분류가 221개에서 207개로 준다). (과제 6)
4. **산업별 AI 노출 지수가 붙지 않는 공고.** 산업코드가 없는 공고, 소분류표에 없는 코드, 연계표로 이어지지 않는 소분류는 값이 결측이어야 하고 멈추면 안 된다. 붙은 비율을 표로 낸다. (과제 6)
5. **설정에 적은 세부 분류나 대분류 이름이 숙련코드 설명 파일에 없는 경우.** 이름을 잘못 적으면 후보 표가 조용히 비게 된다. 빈 표를 내지 말고 없는 이름을 적은 예외를 낸다. (과제 5)

---

## 파일 구성

| 파일 | 바뀌는 내용 | 과제 |
|---|---|---|
| `docs/superpowers/specs/2026-11-12-research-infra-design.md`, `CLAUDE.md`, `experiments/CLAUDE.md` | 확정 사항 23~33 반영 | 1 |
| `experiments/config.yaml`, `experiments/src/aabundle/validate.py` | 새 설정 값과 설정 확인 규칙 | 2 |
| `experiments/src/aabundle/sample.py`, `experiments/src/aabundle/classif.py`, `experiments/tests/synth.py` | 게시일 규칙, 기간 이름, 시도 합치기, 산업코드 처리 | 3 |
| `experiments/src/aabundle/treatment.py` | 생성형 AI 활용 키워드 표시, 비교 공고의 정의, 처치 정의 두 개 추가 | 3A |
| `experiments/scripts/s00_data_audit.py` (`scan`) | 마감일 읽기, 게시일 관련 집계, 1단계 기록 항목 | 4 |
| `experiments/scripts/s00_data_audit.py` (`report`) | 바뀐 표와 새 표(기간, 직종군, 인접 숙련 후보, placebo 숙련 후보) | 5 |
| `experiments/src/aabundle/confounders.py` (새 파일) | 산업 소분류별 AI 노출 지수 | 6 |
| `docs/s0-data-audit.md`, `README.md`, `docs/decisions.md` | 실제 자료 실행 결과 | 7 |

---

### 과제 1: 설계 문서와 작업 규칙에 확정 사항 반영

**파일:**
- 수정: `docs/superpowers/specs/2026-11-12-research-infra-design.md`, `CLAUDE.md`, `experiments/CLAUDE.md`

코드는 바꾸지 않는다. 시험은 기존 시험이 그대로 통과하는지만 본다.

- [ ] **1단계: 브랜치를 만든다**

```bash
git checkout master && git checkout -b infra-2
```

`git status --short`에 `M docs/decisions.md`와 `?? test.ipynb`만 보여야 한다(`docs/decisions.md`의 변경은 저자가 정한 내용을 적은 것이며 이 과제의 커밋에 함께 넣는다).

- [ ] **2단계: 설계 문서를 고친다**

  - 머리말의 상태를 `저자 승인 (2026-11-12), 확정 사항 반영 개정 (2026-11-13)`으로 바꾼다.
  - 2장 결정 표에 위 "이번에 반영하는 확정 사항"의 23~33번을 표의 문장 그대로 넣고 정한 날을 `2026-11-13`으로 적는다. 결정 20의 문장 끝에 "(결정 26으로 세 개를 뺌)"을 붙인다.
  - 3.4절 끝에 한 문단을 더한다: 결정 26에 따라 `0122`, `6121`, `6154`를 뺀 25개가 분석 대상이라는 것, 자료 점검 보고서(2026-11-12 실행)의 2023~2026년 처치 공고 수가 관리직 52건, 영업·판매직 89건이었다는 것.
  - 6.1절: `confounders.parquet` 행의 내용을 "산업 소분류별 AI 노출 지수(AIIE)"로 바꾼다. "제외 규칙의 적용 순서" 문단의 ①을 "게시일(결정 25)이 분석 기간 밖이거나 게시일을 정할 수 없는 공고"로, ②의 "28개"를 "25개"로 바꾼다.
  - 6.2절 변수 대응 표의 행을 고친다: 산업("분류표에 없는 코드는 결측으로 둠(결정 28)"), 직종("분석 대상 세분류 25개 범주(결정 20, 26)"), 직종군("중분류 4개(결정 27)"), 근무지("`region`의 시도. 광주·전남·전남광주는 한 범주(결정 29)"), 게시 시점("`post_start`, 없으면 `post_end`의 연-분기(결정 25). 기간별 분석의 기간은 결정 24의 세 기간"), 산업별 AI 도입률(행 이름을 "산업별 AI 노출 정도"로, 자료의 열을 "Felten 등의 AIIE 언어 모형 판을 산업 소분류로 이은 값(결정 30)"으로, 비고에 잇는 순서 "KSIC 소분류 → ISIC Rev.4(판단으로 만든 표 `experiments/crosswalks/ksic10_sub_to_isic4.csv`) → NAICS 6자리(미국 통계국 연계표) → NAICS 4자리의 AIIE, 소분류에 이어진 서로 다른 NAICS 4자리 산업의 단순 평균"을 적는다).
  - 6.3절 `confounders.py` 행: 역할을 "산업 소분류별 AI 노출 지수 만들기와 붙이기"로, 주요 함수를 `aiie_by_ksic_sub`, `attach_aiie`로 바꾼다. `add_industry_ai_rate`는 쓰지 않는다고 한 문장 적는다.
  - 6.5절 설정 항목 목록에 더한다: 게시일로 쓰는 열의 순서, 기간 이름과 연도, 분석에서 빼는 중분류, 직종군별 추정의 처치 공고 수 기준, 분류표에 없는 산업코드의 처리, 합치는 시도, 인접 숙련 후보의 세부 분류, placebo 숙련 후보의 대분류, 외부 자료 두 개와 산업 연계표의 경로.
  - 6.6절 목록의 1번 끝에 "게시일이 없는 공고 가운데 마감일로 게시일을 대신한 공고 수와 둘 다 없는 공고 수, 게시일과 마감일이 모두 있는 공고의 두 날짜 간격 분포"를 더한다. 목록에 11~13번을 더한다: "11. 연봉과 근무지"(보고서에 이미 있는 절이다), "12. 인접 숙련 후보(6.7절 3번)와 placebo 숙련 후보의 숙련별 공고 수와 출현율", "13. 산업별 AI 노출 지수가 붙은 공고의 비율과 값의 분포". 1번과 10번의 "28개"는 "분석 대상 세분류"로 바꾼다.
  - 6.7절 표 아래에 한 문단을 더한다: 2026-11-13에 3번을 뺀 모든 항목이 확정되었고 값은 `docs/decisions.md`에 있다는 것, 7·8·9·10·13·15·16번은 기본값과 다르게 정해졌으며 결정 23~30이 그 내용이라는 것.
  - 6.8절의 `confounders.py` 줄을 "연계표 세 개를 이어 산업 소분류별 값을 만드는지, 앞자리 0이 빠진 코드를 맞추는지, 이어지지 않는 산업이 결측으로 남는지"로 바꾼다.
  - 2장 결정 14의 문장 끝에 "(결정 34로 바꿈)"을 붙인다. 6.1절 `posting_base.parquet` 행의 내용에 "본문 글자 수"를 더하고, 그 아래 "수집 사이트는 교란변수가 아니다(결정 14)…" 문단을 "수집 사이트는 교란변수이다(결정 34)"로 고친다. `treat.parquet` 행에 "생성형 AI 활용 키워드 여부"를 더한다.
  - 6.2절 변수 대응 표: 수집 사이트 행을 "`source` | 교란변수(결정 34)"로 고치고, "공고의 상세한 정도 | 본문(`detail`)의 글자 수 | 로그 변환. 길게 쓴 공고일수록 숙련이 많이 추출되므로 통제함(결정 35)" 행을 더한다. 처치 정의 표에 두 행을 더한다: "생성형 AI 활용 키워드만 | 생성형 AI 활용 키워드에 걸린 공고", "생성형 AI 활용(숙련코드 또는 키워드) | 생성형 AI 활용 숙련이 있거나 생성형 AI 활용 키워드에 걸린 공고". 표 아래 문장 "비교 공고는 모든 정의에서 같다"의 뒤에 "비교 공고는 AI 에이전트 키워드, 생성형 AI 활용 키워드, 54개 숙련 어느 것에도 해당하지 않는 공고이다(결정 36)"를 더한다.
  - 6.6절 목록에 14번을 더한다: "14. 처치 여부별 공고당 숙련 수와 본문 글자 수, 처치 공고가 있는 기업 수와 기업별 집중도, 같은 기업·같은 연도에 비교 공고도 있는 처치 공고의 수, 처치 공고를 처음 낸 연도별 기업 수와 그 전에 낸 비교 공고 수".
  - 10장 끝에 한 문장을 더한다: 2번과 3번 사이에 확정 사항을 반영하고 자료 점검을 다시 실행하는 단계가 있으며 계획은 `docs/superpowers/plans/2026-11-13-infra-2-decisions-audit.md`라는 것.

- [ ] **3단계: 작업 규칙 파일을 고친다**

  - `CLAUDE.md` 규칙 2: "원자료(`data/`)는 읽기 전용이다. 어떤 코드도 `data/` 아래에 쓰지 않는다. 외부에서 받은 공개 자료는 사람이 직접 내려받아 `data/external/`에 둔다."
  - `CLAUDE.md` 연구 개요의 끝에 한 문장: 분석 기간은 2023년 1월부터 2026년 8월까지이고 분석 대상은 직종 세분류 25개라는 것.
  - `experiments/CLAUDE.md` 셋째 줄: "placebo 숙련(생성형 AI 활용과 관계가 없어야 하는 숙련으로, 같은 방법으로 추정했을 때 효과가 0으로 나오는지 확인하는 데 쓴다. `proposal.md` 7장 2번)의 목록은 결과를 보기 전에 정한다." (저자가 "위약 검증"이라는 표현의 뜻을 되물었으므로 그 표현은 쓰지 않는다.)
  - `CLAUDE.md` 규칙 7: "공고 원문 열(`detail`, `title`)은 처치 판별과 본문 글자 수 계산에서만 읽고, 중간 자료와 보고서에 원문을 저장하지 않는다. 글자 수는 저장한다." `experiments/CLAUDE.md` 다섯째 줄의 "공고 원문 열은 처치 판별에서만 읽는다"도 같은 뜻으로 고친다.
  - `experiments/CLAUDE.md` 넷째 줄: "교란변수(처치와 결과 모두에 영향을 주는 변수)는 같은 기간의 처치 변수로 만들지 않는다. 산업별 AI 노출 정도는 외부 지수(Felten 등의 AIIE)를 쓴다."

- [ ] **4단계: 확인하고 커밋한다**

실행: `-m pytest -q -W error` → `137 passed, 4 deselected`.
설계 문서에서 `28개`를 찾아 남은 곳이 결정 20의 문장, 3.4절의 계산 설명, 6.7절 표의 옛 기본값 설명뿐인지 확인한다.

```bash
git add docs/superpowers/specs/2026-11-12-research-infra-design.md CLAUDE.md experiments/CLAUDE.md docs/decisions.md docs/superpowers/plans/2026-11-13-infra-2-decisions-audit.md
git commit -m "설계 문서와 작업 규칙에 2026-11-13 확정 사항 반영"
```

---

### 과제 2: 설정 값과 설정 확인

**파일:**
- 수정: `experiments/config.yaml`, `experiments/src/aabundle/validate.py`, `experiments/tests/test_validate.py`
- 수정(기대값만): 설정의 기본값이 바뀌어 달라지는 기존 시험

**주고받는 것:**
- 만드는 것: 아래 설정 항목. 뒤 과제는 이 이름으로 읽는다.
- 만드는 것: `validate.INDUSTRY_NOT_IN_TABLE = ("category", "missing")`

- [ ] **1단계: 실패하는 시험을 쓴다** (`test_validate.py`)

기존 `test_each_rule_names_the_key`의 매개변수 목록에 다음 경우를 더한다(항목, 넣을 값, 오류 문장에 들어 있어야 하는 글자).

```python
("period.date_columns", [], "period.date_columns"),
("period.groups", [{"name": "초기", "years": [2023, 2024]}, {"name": "1 도입기", "years": [2025]}], "2026"),          # 기간에 드는 연도가 빠짐
("period.groups", [{"name": "초기", "years": [2023, 2024, 2025]}, {"name": "1 도입기", "years": [2025]},
                   {"name": "2 도입기", "years": [2026]}], "2025"),                                                   # 한 연도가 두 기간에 듦
("period.groups", [{"name": "초기", "years": [2022, 2023, 2024]}, {"name": "1 도입기", "years": [2025]},
                   {"name": "2 도입기", "years": [2026]}], "2022"),                                                   # 분석 기간 밖의 연도
("period.groups", [{"name": "초기", "years": [2023, 2024]}, {"name": "초기", "years": [2025]},
                   {"name": "2 도입기", "years": [2026]}], "초기"),                                                    # 이름이 겹침
("occupations.exclude_keco2", ["13"], "1332"),                                                                      # 뺀 중분류의 세분류가 목록에 남음
("occupations.exclude_keco2", [1], "occupations.exclude_keco2"),
("occupations.group_min_treated", 0, "occupations.group_min_treated"),
("industry.not_in_table", "drop", "industry.not_in_table"),
("region.merge_sido", {"광주·전남": ["광주", "전남"], "호남": ["전남", "전북"]}, "전남"),                              # 한 시도가 두 묶음에 듦
("skills.adjacent_candidate_details", "MLOps및AI인프라", "skills.adjacent_candidate_details"),                       # 목록이 아님
("skills.placebo_candidate_domains", [""], "skills.placebo_candidate_domains"),
```

새 시험 두 개:

```python
def test_use_keywords_shipped():
    kw = paths.load_config()["treatment"]["keywords"]["use"]
    assert len(kw) == 34 and len(set(kw)) == 34
    joined = re.compile("|".join(kw), re.IGNORECASE)
    for hit in ["ChatGPT 활용 경험", "챗GPT", "생성형 AI 툴", "프롬프트 엔지니어링", "LLM 기반", "LLMs", "AI 활용 능력", "GenAI"]:
        assert joined.search(hit), hit
    for miss in ["Fulfillment 센터 운영", "풀필먼트", "명령 프롬프트", "KAI 활용", "달리 정한 바"]:
        assert not joined.search(miss), miss


def test_groups_require_year_unit():
    cfg = paths.load_config()
    cfg["period"]["unit"] = "quarter"
    with pytest.raises(ValueError, match="period.unit"):
        validate.validate_config(cfg)


def test_shipped_values_of_2026_11_13():
    cfg = paths.load_config()
    assert cfg["data"]["years"] == [2023, 2024, 2025, 2026]
    assert (cfg["period"]["start"], cfg["period"]["end"], cfg["period"]["unit"]) == ("2023-01-01", "2026-08-31", "year")
    assert [(g["name"], g["years"]) for g in cfg["period"]["groups"]] == [
        ("초기", [2023, 2024]), ("1 도입기", [2025]), ("2 도입기", [2026])]
    assert cfg["period"]["date_columns"] == ["post_start", "post_end"]
    assert len(cfg["occupations"]["keco4"]) == 25
    assert not {"0122", "6121", "6154"} & set(cfg["occupations"]["keco4"])
    assert cfg["occupations"]["exclude_keco2"] == ["01", "61"] and cfg["occupations"]["group_min_treated"] == 100
    assert cfg["industry"]["not_in_table"] == "missing"
    assert cfg["region"]["merge_sido"] == {"광주·전남": ["광주", "전남", "전남광주"]}
    assert cfg["salary_band_edges"] == [26000000, 30000000, 35000000, 40000000]
    assert cfg["skills"]["sparse_filter"] == {"min_rate": 0.001, "min_treated": 30}
```

- [ ] **2단계: 시험이 실패하는지 확인한다**

실행: `-m pytest experiments/tests/test_validate.py -q` → 새 경우가 `KeyError` 또는 "DID NOT RAISE"로 실패.

- [ ] **3단계: `config.yaml`을 고친다**

바뀌는 블록만 적는다. 나머지(`treatment`, `sample`, `estimation`, `occ_group`)는 그대로 두고, "임시값", "자료 점검 뒤 확정" 주석은 확정 근거(결정 번호)로 바꾼다.

```yaml
data:
  years: [2023, 2024, 2025, 2026]          # 결정 23
  codebook: data/postings/ojp_codebook_skill.csv
  ksic: data/postings/KSIC10.xlsx
  keco: data/postings/KECO2025.xlsx
  ked_firm: data/ked/KED5002.txt
  aiie: "data/external/aioe/Language Modeling AIOE and AIIE.xlsx"   # 결정 30
  naics_isic: data/external/census/2017_NAICS_to_ISIC_4.xlsx
  ksic_isic: experiments/crosswalks/ksic10_sub_to_isic4.csv   # 판단으로 만든 표. git으로 관리

period:                                    # 결정 23, 24, 25
  start: "2023-01-01"
  end: "2026-08-31"
  unit: year
  date_columns: [post_start, post_end]     # 앞의 열이 비어 있으면 다음 열을 게시일로 쓴다
  groups:
    - {name: "초기", years: [2023, 2024]}
    - {name: "1 도입기", years: [2025]}
    - {name: "2 도입기", years: [2026]}

occupations:                               # 결정 20, 26, 27
  min_use_postings: 30
  exclude_keco2: ["01", "61"]
  group_min_treated: 100
  keco4: ["0241", "0243", "0261", "0262", "0265", "0271", "0272", "0284", "0292", "0295",
          "0299", "1320", "1331", "1332", "1333", "1339", "1342", "1351", "1352", "1361",
          "2143", "2151", "4154", "4155", "4166"]

industry:
  not_in_table: missing                    # 결정 28. category | missing

region:
  merge_sido:                              # 결정 29
    "광주·전남": ["광주", "전남", "전남광주"]

skills:
  adjacent: []                             # 후보 표를 본 뒤 정한다
  adjacent_candidate_details: ["MLOps및AI인프라", "로봇및자율시스템"]   # 설계 문서 6.7절 3번의 후보
  placebo: []                              # 결과를 보기 전에 정한다
  placebo_candidate_domains: ["건설·현장·이동 작업", "생산·제조·정비", "돌봄·서비스·공공", "자격·인증"]
  sparse_filter:                           # 결정 32
    min_rate: 0.001
    min_treated: 30

salary_band_edges: [26000000, 30000000, 35000000, 40000000]   # 결정 31. 단위는 원
```

`treatment` 블록의 `keywords`에 생성형 AI 활용 표현의 목록을 더하고(결정 36), 주석의 "임시값이며 확정 필요"를 결정 32로 바꾼다. `strict`, `broad_extra`, `exclude`는 그대로 둔다.

```yaml
    use: ['Chat\s*GPT', '챗\s*GPT', '챗\s*지피티', 'GPT', '생성형\s*AI', '생성형\s*인공지능', '생성\s*AI', 'Generative\s*AI',
          '(?<![A-Za-z])Gen\s?AI', '프롬프트\s*엔지니어', '프롬프트\s*(?:작성|설계|활용|디자인)', 'prompt\s*engineer',
          '(?<![A-Za-z])LLMs?(?![A-Za-z])', '대규모\s*언어\s*모델', '거대\s*언어\s*모델', '초거대\s*AI',
          'Copilot', '코파일럿', 'Midjourney', '미드저니', 'Stable\s*Diffusion', '스테이블\s*디퓨전',
          'Claude', 'Gemini', '제미나이', 'Perplexity', '퍼플렉시티', '뤼튼', 'HyperCLOVA', '하이퍼클로바', '클로바\s*X',
          'Notion\s*AI|노션\s*AI', '(?<![A-Za-z])AI\s*(?:툴|도구)', '(?<![A-Za-z])AI\s*활용']
```

표현을 고른 기준: 생성형 AI를 가리키는 일반 표현, 프롬프트 작성 표현, 널리 쓰이는 생성형 AI 도구 이름이다. 다른 낱말 안에 들어가 잘못 걸릴 수 있는 표현은 앞뒤 글자를 제한했다(`LLM`은 `Fulfillment` 안에 들어 있다). 뜻이 여럿인 표현(`달리`, `Sora`, `Bard`, `Cursor`, `클로드`)은 넣지 않았다. 이 사전의 주된 쓰임은 비교 공고에서 생성형 AI를 언급한 공고를 빼는 것이므로 범위를 넓게 잡았고, 개발 맥락의 표현(`LLM`, `Copilot`)도 넣었다. 표현별 공고 수는 자료 점검 보고서(`t06c_keyword_pattern_hits`)에 나온다.

`treatment.definition`의 주석에 고를 수 있는 값을 `all | dev | use | keyword | code | use_keyword | use_any`로 적는다. 값은 `use` 그대로이다.

`placebo_candidate_domains`의 네 대분류는 이 계획이 정한 후보 범위이다(생성형 AI 활용과 관계가 멀다고 본 대분류). 저자가 계획 검토에서 바꿀 수 있다.

- [ ] **4단계: `validate.validate_config`에 규칙을 더한다**

오류 문장은 지금처럼 항목 이름과 문제를 한국어로 적는다. `data`의 세 외부 자료 경로를 뺀 새 항목은 모두 있어야 하며, 없으면 지금의 다른 항목처럼 "항목이 없습니다"로 알린다.

  - `period.date_columns`: 빈 문자열이 아닌 문자열이 하나 이상 든 목록.
  - `period.groups`: 목록. 원소마다 `name`(빈 문자열이 아닌 문자열)과 `years`(정수가 하나 이상 든 목록). 이름이 겹치면 안 된다. `period.start`의 연도부터 `period.end`의 연도까지 모든 연도가 정확히 한 기간에 들어야 하고, 그 밖의 연도가 있으면 안 된다. 오류 문장에는 문제가 된 연도나 이름을 적는다. `period.groups`가 있으면 `period.unit`은 `year`여야 한다.
  - `occupations.exclude_keco2`: 숫자 2자리 문자열의 목록(비어도 됨). `occupations.keco4`에 이 목록의 글자로 시작하는 코드가 있으면 그 코드를 적어 오류.
  - `occupations.group_min_treated`: 1 이상의 정수.
  - `industry.not_in_table`: `INDUSTRY_NOT_IN_TABLE` 가운데 하나.
  - `region.merge_sido`: 사전. 값은 빈 문자열이 아닌 문자열이 하나 이상 든 목록. 한 시도 이름이 두 묶음에 들면 그 이름을 적어 오류.
  - `skills.adjacent_candidate_details`, `skills.placebo_candidate_domains`: 빈 문자열이 아닌 문자열의 목록(비어도 됨).
  - `treatment.keywords.use`: 지금의 `strict`와 같은 확인(하나 이상, 표현마다 정규식으로 읽힘, `|`로 이어 붙인 것도 읽힘).
  - `data.aiie`, `data.naics_isic`, `data.ksic_isic`: 있으면 빈 문자열이 아닌 문자열. 파일이 있는지는 읽는 함수가 확인한다(과제 6).

- [ ] **5단계: 시험을 통과시킨다**

실행: `-m pytest -q -W error` → 모두 통과.
`synth.write_all`은 실제 설정을 읽어 일부 항목만 바꾸므로, 기본값이 바뀌어 기대값이 달라지는 기존 시험이 있다(연봉 구간 이름, 분석 기간의 처음과 끝, 분석 기간을 바꿔 보는 시험에서 `period.groups`와 맞지 않게 되는 값). 가상 자료 쪽(`synth.write_all`이 바꾸는 설정 항목, 가상 공고의 날짜)을 맞추는 것을 먼저 하고, 그래도 달라지는 단언만 고친다. `synth.write_all`은 가상 공고의 연도가 모두 한 기간에 들도록 `period.start`, `period.end`, `period.groups`를 가상 자료에 맞게 정한다.

- [ ] **6단계: 커밋한다**

```bash
git add experiments/config.yaml experiments/src/aabundle/validate.py experiments/tests
git commit -m "설정에 2026-11-13 확정값을 넣고 새 항목의 확인 규칙 추가"
```

---

### 과제 3: 게시일 규칙, 기간 이름, 시도 합치기, 산업코드 처리

**파일:**
- 수정: `experiments/src/aabundle/sample.py`, `experiments/src/aabundle/classif.py`, `experiments/tests/synth.py`
- 시험: `experiments/tests/test_sample.py`, `experiments/tests/test_classif.py`

**주고받는 것:**
- 쓰는 것: 과제 2의 설정 항목 `period.date_columns`, `period.groups`, `region.merge_sido`, `industry.not_in_table`
- 만드는 것:
  - `sample.posting_date(raw: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]` — (날짜, 날짜를 가져온 열 이름). 날짜는 `datetime64[ns]`, 열 이름은 `string`이고 날짜를 정할 수 없으면 둘 다 결측.
  - `sample.period_name(year: pd.Series, groups: list[dict]) -> pd.Series` — 기간 이름(`string`). 어느 기간에도 들지 않으면 결측.
  - `sample.merge_sido(sido: pd.Series, merge: dict[str, list[str]]) -> pd.Series`
  - `classif.ksic_sub(s, valid, not_in_table: str = "category")` — `"missing"`이면 분류표에 없는 소분류를 결측으로 돌려준다.
  - `sample.RAW_COLUMNS`에 `"post_end"` 추가. `sample.BASE_COLUMNS`에서 `"quarter"` 뒤에 `"period"`, `"date_source"` 추가.
  - `base`의 `year`, `month`, `quarter`, `in_period`는 `posting_date`의 날짜로 계산한다.

- [ ] **1단계: 실패하는 시험을 쓴다**

`test_sample.py`:

```python
def test_posting_date_uses_columns_in_order():
    raw = pd.DataFrame({"post_start": ["2025-03-02", None, None, "날짜아님"],
                        "post_end": ["2025-04-01", "2025-05-10", None, "9999-12-31"]})
    date, source = sample.posting_date(raw, ["post_start", "post_end"])
    assert date.dt.strftime("%Y-%m-%d").tolist()[:2] == ["2025-03-02", "2025-05-10"]
    assert date.isna().tolist() == [False, False, True, True]          # 읽을 수 없는 값과 범위 밖의 날짜는 결측
    assert source.tolist()[:2] == ["post_start", "post_end"]
    assert source.isna().tolist() == [False, False, True, True]


def test_posting_date_accepts_date_objects():
    import datetime as dt
    raw = pd.DataFrame({"post_start": [dt.date(2025, 3, 2), None], "post_end": [None, dt.date(9999, 12, 31)]})
    date, source = sample.posting_date(raw, ["post_start", "post_end"])
    assert date.isna().tolist() == [False, True] and source.isna().tolist() == [False, True]


def test_period_name():
    groups = [{"name": "초기", "years": [2023, 2024]}, {"name": "1 도입기", "years": [2025]}]
    out = sample.period_name(pd.Series([2023, 2024, 2025, 2026, pd.NA], dtype="Int64"), groups)
    assert out.tolist()[:3] == ["초기", "초기", "1 도입기"]
    assert out.isna().tolist() == [False, False, False, True, True]


def test_merge_sido():
    s = pd.Series(["광주", "전남", "전남광주", "서울", "기타", pd.NA], dtype="string")
    out = sample.merge_sido(s, {"광주·전남": ["광주", "전남", "전남광주"]})
    assert out.tolist()[:5] == ["광주·전남", "광주·전남", "광주·전남", "서울", "기타"]
    assert pd.isna(out.iloc[5])


def test_base_uses_post_end_when_post_start_is_missing(base):       # base: 이 파일의 기존 고정물
    """가상 공고 가운데 게시일이 없고 마감일만 있는 공고는 마감일의 연도·월·기간을 갖고 분석 기간 안에 든다.
    둘 다 없는 공고는 연도가 결측이고 분석 기간 밖이다."""
    by_id = base.set_index("posting_id")
    assert by_id.loc[synth.ID_END_ONLY, "date_source"] == "post_end"
    assert by_id.loc[synth.ID_END_ONLY, "in_period"] and by_id.loc[synth.ID_END_ONLY, "period"] == synth.PERIOD_NAME
    assert pd.isna(by_id.loc[synth.ID_NO_DATE, "date_source"]) and pd.isna(by_id.loc[synth.ID_NO_DATE, "year"])
    assert not by_id.loc[synth.ID_NO_DATE, "in_period"]
    assert base.loc[base["in_period"], "period"].notna().all()


def test_base_merges_sido_and_blanks_ksic_not_in_table(base):
    assert not base["sido"].isin(["광주", "전남", "전남광주"]).any()
    assert not (base["ksic_sub"] == classif.KSIC_NOT_IN_TABLE).any()
    # 분류표에 없는 코드는 5자리 코드는 남고 소분류만 결측이다(보고서가 건수를 셀 수 있게)
    assert (base["ksic5"].notna() & base["ksic_sub"].isna()).any()
```

`test_classif.py`:

```python
def test_ksic_sub_not_in_table_as_missing():
    s = pd.Series(["62010", "41999", None])
    assert classif.ksic_sub(s, {"620"}, not_in_table="missing").isna().tolist() == [False, True, True]
    assert classif.ksic_sub(s, {"620"}).tolist()[1] == classif.KSIC_NOT_IN_TABLE
    with pytest.raises(ValueError, match="not_in_table"):
        classif.ksic_sub(s, {"620"}, not_in_table="drop")
```

- [ ] **2단계: 가상 자료를 넓힌다** (`synth.py`)

  - 공고 표에 `post_end` 열(`pa.date32()`)을 더한다. 기존 공고는 게시일 30일 뒤를 마감일로 준다.
  - 공고 두 건을 더한다: 게시일이 없고 마감일만 있는 공고(`ID_END_ONLY`)와 둘 다 없는 공고(`ID_NO_DATE`). 둘 다 분석 대상 세분류, 사업자번호 있음, 숙련 있음으로 만든다. 상수 `ID_END_ONLY`, `ID_NO_DATE`, `PERIOD_NAME`(가상 공고의 연도가 드는 기간 이름)을 모듈에 둔다.
  - 근무지 순환 목록에 `"광주 북구"`와 `"전남광주통합특별시 광산구"`를 더한다. 산업코드 순환 목록에 가상 소분류표에 없는 5자리 코드가 없으면 하나 더한다.
  - 공고 수가 바뀌므로 공고 수나 행 묶음 수를 직접 적은 기존 단언은 새 수로 고치고 보고에 적는다.

- [ ] **3단계: 시험이 실패하는지 확인한다**

실행: `-m pytest experiments/tests/test_sample.py experiments/tests/test_classif.py -q` → 새 시험이 `AttributeError`(함수 없음) 또는 `TypeError`(인자 없음)로 실패.

- [ ] **4단계: 구현한다**

  - `posting_date`: 열마다 `pd.to_datetime(…, errors="coerce")`로 읽고 `columns`의 순서대로 처음으로 값이 있는 열을 쓴다. 시각은 버리고 날짜만 남긴다(`dt.normalize()`). `columns`의 열이 `raw`에 없으면 없는 열 이름을 적은 `KeyError`.
  - `build_posting_base`: `dts`를 `posting_date(raw, cfg["period"]["date_columns"])`의 날짜로 바꾸고 `date_source`를 함께 싣는다. `period`는 `period_name(year, cfg["period"]["groups"])`. `sido`는 `merge_sido(to_sido(raw["region"]), cfg["region"]["merge_sido"])`. `ksic_sub`는 `classif.ksic_sub(raw["ksic10"], ksic_valid, cfg["industry"]["not_in_table"])`. `_SIDO`의 주석("저자가 정할 수 있도록 따로 둔다")은 결정 29에 맞게 고친다. `to_sido` 자체는 그대로 `전남광주`를 돌려주고 합치기는 `merge_sido`가 한다.
  - `apply_exclusions`의 설명 글에서 "`post_start`가 결측인 공고"를 "게시일을 정할 수 없는 공고"로 고친다.

- [ ] **5단계: 시험을 통과시키고 커밋한다**

실행: `-m pytest -q -W error` → 모두 통과.

```bash
git add experiments/src/aabundle/sample.py experiments/src/aabundle/classif.py experiments/tests
git commit -m "게시일이 없으면 마감일을 쓰고, 기간 이름·시도 합치기·분류표에 없는 산업코드의 결측 처리 추가"
```

---

### 과제 3A: 생성형 AI 활용 키워드와 비교 공고

**파일:**
- 수정: `experiments/src/aabundle/treatment.py`, `experiments/tests/synth.py`
- 시험: `experiments/tests/test_treatment.py`

**주고받는 것:**
- 쓰는 것: 과제 2의 `treatment.keywords.use`
- 만드는 것:
  - `treatment.keyword_lists(cfg)`가 돌려주는 사전에 `"use"` 추가
  - `treatment.build_treat(...)`의 결과에 `kw_broad` 다음 열로 `kw_use`(bool) 추가. 제외 표현을 지운 뒤 `use` 사전에 걸리는지이다.
  - `treatment.DEFINITIONS = ("all", "dev", "use", "keyword", "code", "use_keyword", "use_any")`
  - `treatment.select_arms(treat, definition, keyword_col="kw_strict")`: 처치 공고는 `use_keyword`이면 `kw_use`, `use_any`이면 `has_use | kw_use`. 비교 공고는 모든 정의에서 `~keyword_col & ~has_dev & ~has_use & ~kw_use`이다. `treat`에 `kw_use` 열이 없으면 `KeyError`.

- [ ] **1단계: 실패하는 시험을 쓴다** (`test_treatment.py`)

```python
def _treat(rows):
    cols = ["kw_strict", "kw_broad", "kw_use", "has_dev", "has_use"]
    return pd.DataFrame(rows, columns=cols).astype(bool)


def test_use_keyword_postings_leave_the_comparison_group():
    #            strict broad  use   dev   use(code)
    t = _treat([(0, 0, 0, 0, 0),      # 아무 것도 아님 → 비교 공고
                (0, 0, 1, 0, 0),      # 생성형 AI 활용 키워드만 → 어느 쪽도 아님
                (0, 0, 0, 0, 1),      # 생성형 AI 활용 숙련만 → 처치 공고
                (0, 0, 1, 0, 1)])     # 둘 다 → 처치 공고
    assert treatment.select_arms(t, "use").tolist() == [0, pd.NA, 1, 1]
    assert treatment.select_arms(t, "use_keyword").tolist() == [0, 1, pd.NA, 1]
    assert treatment.select_arms(t, "use_any").tolist() == [0, 1, 1, 1]
    assert treatment.select_arms(t, "dev").tolist() == [0, pd.NA, pd.NA, pd.NA]      # 비교 공고는 어떤 정의에서도 같다


def test_select_arms_requires_kw_use():
    t = _treat([(0, 0, 0, 0, 0)]).drop(columns="kw_use")
    with pytest.raises(KeyError, match="kw_use"):
        treatment.select_arms(t, "use")


def test_build_treat_flags_use_keywords(synth_data):
    """가상 공고 본문에 `ChatGPT 활용`이 있고 숙련코드는 없는 공고(`synth.ID_USE_KEYWORD_ONLY`)는 `kw_use`만 참이다.
    본문에 `Fulfillment`만 있는 공고(`synth.ID_FULFILLMENT`)는 `kw_use`가 거짓이다."""
    tr = _build(synth_data).set_index("posting_id")          # 이 파일에서 build_treat를 부르는 기존 도우미나 고정물
    assert tr.loc[synth.ID_USE_KEYWORD_ONLY, ["kw_use", "kw_strict", "has_use"]].tolist() == [True, False, False]
    assert not tr.loc[synth.ID_FULFILLMENT, "kw_use"]
    assert list(tr.columns)[:3] == ["kw_strict", "kw_broad", "kw_use"]
```

기존 시험 가운데 처치 정의 다섯 개의 처치 공고와 비교 공고를 가상 공고로 확인하는 매개변수 시험에 새 정의 두 개를 더하고, 비교 공고의 기대값에서 `kw_use`인 공고를 뺀다.

- [ ] **2단계: 가상 자료를 넓힌다** (`synth.py`): 공고 두 건(`ID_USE_KEYWORD_ONLY`, `ID_FULFILLMENT`)을 더한다. 둘 다 분석 대상 세분류, 사업자번호 있음, AI 에이전트 숙련코드가 아닌 숙련 있음.

- [ ] **3단계: 시험이 실패하는지 확인하고 구현한다**

실행: `-m pytest experiments/tests/test_treatment.py -q` → 새 시험이 `KeyError` 또는 `ValueError`(알 수 없는 처치 정의)로 실패. 구현한 뒤 `-m pytest -q -W error` 모두 통과. `treatment.py`의 함수 설명 글에서 비교 공고의 정의를 고친다. `validate.py`는 `treatment.DEFINITIONS`를 읽으므로 고칠 것이 없다.

- [ ] **4단계: 커밋한다**

```bash
git add experiments/src/aabundle/treatment.py experiments/tests
git commit -m "생성형 AI 활용 키워드 표시를 더하고, 걸린 공고를 비교 공고에서 제외"
```

---

### 과제 4: 자료 점검 1단계(`scan`)에 반영

**파일:**
- 수정: `experiments/scripts/s00_data_audit.py` (`scan`과 그 위의 상수, `_post_start_counts`)
- 시험: `experiments/tests/test_s00_scan.py`, `experiments/tests/test_s00_scan_meta.py`

**주고받는 것:**
- 쓰는 것: 과제 3의 `sample.posting_date`, `base`의 새 열 `period`, `date_source`
- 만드는 것:
  - `SCAN_CONFIG_KEYS`에 `"period.date_columns"`, `"period.groups"`, `"industry.not_in_table"`, `"region.merge_sido"` 추가
  - `post_start_missing.parquet`의 열(파일 연도 × 수집 사이트별): `year`, `source`, `n_raw`, `n_no_post_start`, `n_no_date`, `n_in_occ`, `n_in_occ_no_post_start`, `n_in_occ_no_date`, `n_in_occ_from_end_in_period`, `n_in_occ_use`, `n_in_occ_use_no_post_start`, `n_in_occ_use_no_date`, `n_in_occ_use_from_end_in_period`. `no_date`는 `period.date_columns`의 모든 열이 비었거나 읽을 수 없는 공고, `from_end_in_period`는 첫 열이 없어 뒤 열로 게시일을 정했고 그 날짜가 분석 기간 안인 공고이다.
  - 새 결과 파일 `post_end_gap.parquet`(`OUTPUT_NAMES`에 추가): 열 `year`(파일 연도), `bin`, `n`. 분석 대상 세분류의 공고 가운데 `period.date_columns`의 첫 열과 둘째 열이 모두 읽히는 공고를, 둘째 날짜에서 첫 날짜를 뺀 일수의 구간으로 센다. 구간은 스크립트의 상수 `POST_END_GAP_BINS = (("음수", None, -1), ("0~7일", 0, 7), ("8~30일", 8, 30), ("31~60일", 31, 60), ("61~90일", 61, 90), ("91~180일", 91, 180), ("181일 이상", 181, None))`로 두고, 공고가 없는 구간도 0으로 싣는다. 열이 하나뿐이면 빈 표를 쓴다.
  - `occ_year_all.parquet`의 `year`는 `posting_date`의 연도(없으면 파일 연도)로 센다.
  - `base.parquet`에 열 `text_len`(정수) 추가: `treatment.text_columns`의 열을 이어 붙인 본문의 글자 수. 본문이 없으면 0. 본문 자체는 지금처럼 어디에도 쓰지 않는다.
  - `treat.parquet`에 과제 3A의 `kw_use` 열이 들어간다. `kw_pattern_hits.parquet`에 `dictionary`가 `use`인 행을 더한다: `kw_use`가 참인 공고 안에서 `use` 사전의 표현마다 걸린 공고 수(제외 표현을 지운 뒤, 게시 연도별).

- [ ] **1단계: 실패하는 시험을 쓴다**

`test_s00_scan.py`에 다음을 더한다(가상 자료로 `scan`을 실행한 결과에 대한 단언).

```python
def test_scan_counts_dates_taken_from_post_end(scan_out):          # scan_out: 이 파일의 기존 고정물(scan 결과 경로 사전)
    psm = pd.read_parquet(scan_out["post_start_missing"])
    total = psm.drop(columns=["year", "source"]).sum()
    assert total["n_no_post_start"] == 2 and total["n_no_date"] == 1          # 가상 공고 ID_END_ONLY, ID_NO_DATE
    assert total["n_in_occ_from_end_in_period"] == 1
    assert total["n_in_occ_no_post_start"] - total["n_in_occ_no_date"] >= total["n_in_occ_from_end_in_period"]
    base = pd.read_parquet(scan_out["base"])
    assert synth.ID_END_ONLY in set(base["posting_id"]) and synth.ID_NO_DATE not in set(base["posting_id"])
    assert set(base["date_source"]) == {"post_start", "post_end"} and base["period"].notna().all()


def test_scan_writes_post_end_gap(scan_out):
    gap = pd.read_parquet(scan_out["post_end_gap"])
    assert gap["bin"].nunique() == len(s00.POST_END_GAP_BINS)                 # 공고가 없는 구간도 0으로 있다
    assert gap.loc[gap["bin"] == "8~30일", "n"].sum() >= 1                      # 가상 공고의 마감일은 게시일 30일 뒤
    assert gap["n"].sum() == synth.N_IN_OCC_BOTH_DATES      # 분석 대상 세분류이고 두 날짜가 모두 있는 가상 공고 수
```

`synth.N_IN_OCC_BOTH_DATES`는 가상 공고 목록에서 계산해 두는 상수이다(직접 숫자를 적지 않는다).

`test_s00_scan.py`에 더한다: `base`의 `text_len`이 0 이상의 정수이고 본문이 있는 가상 공고에서 그 본문의 글자 수와 같다. `base`, `treat`, `skills_long` 어느 열에도 가상 본문의 글자열(`synth`의 본문 가운데 하나)이 들어 있지 않다(기존에 같은 단언이 있으면 새 열까지 포함되는지만 확인한다). `kw_pattern_hits`에 `dictionary == "use"`인 행이 표현 수만큼 연도마다 있고, `Chat\s*GPT` 표현의 합이 1 이상이다.

`test_s00_scan_meta.py`: 기존 "설정을 바꾸면 `build_tables`가 거부한다" 시험과 같은 방식으로, `scan` 뒤에 `region.merge_sido`를 `{}`로 바꾸면 `build_tables`가 `region.merge_sido`를 적은 `RuntimeError`를 내고, `occupations.group_min_treated`를 바꾸면 내지 않는지 단언한다. 기존 시험이 `period.start`를 바꾸는 값은 `period.groups`와 어긋나지 않게(같은 연도 안의 다른 날짜로) 고친다.

- [ ] **2단계: 시험이 실패하는지 확인한다**

실행: `-m pytest experiments/tests/test_s00_scan.py experiments/tests/test_s00_scan_meta.py -q` → 새 시험이 열이나 파일이 없어 실패.

- [ ] **3단계: 구현한다**

  - `_post_start_counts`를 새 열에 맞게 고친다. 필요한 값(`posting_date`의 날짜와 열 이름, `in_period`)은 `scan`이 이미 만든 `b`에서 넘긴다.
  - `post_end_gap`은 다른 작은 결과 파일과 같은 방식으로 모아 `.part`로 쓰고 이름을 바꾼다. 시작할 때 지우고 실패하면 지우는 파일 목록에도 들어간다.
  - 원자료의 `post_end` 열은 `sample.RAW_COLUMNS`에 들어 있으므로 `read_cols`에 저절로 들어간다. 열이 없는 연도 파일이 있으면 지금처럼 `rawio.open_postings`가 없는 열을 적어 멈춘다.
  - 결과 파일 수를 적은 설명 글과 `README.md`의 중간 자료 목록을 고친다(아홉 → 열 개에 1단계 기록).

- [ ] **4단계: 시험을 통과시키고 커밋한다**

실행: `-m pytest -q -W error` → 모두 통과.

```bash
git add experiments/scripts/s00_data_audit.py experiments/tests README.md
git commit -m "자료 점검 1단계가 마감일을 읽고 게시일 관련 건수와 두 날짜의 간격을 집계"
```

---

### 과제 5: 자료 점검 2단계(`report`)의 표

**파일:**
- 수정: `experiments/scripts/s00_data_audit.py` (`build_tables`와 표 함수, 보고서 설명 글), `experiments/src/aabundle/skills.py`(`load_codebook`이 세부 분류 열을 함께 읽게 할 때만)
- 시험: `experiments/tests/test_s00_report.py`

**주고받는 것:**
- 쓰는 것: 과제 4의 `post_start_missing`, `post_end_gap`, `base`의 `period`
- 만드는 것: 아래 표. `build_tables`가 돌려주는 사전의 이름이다.

| 표 | 내용 |
|---|---|
| `t01b_post_start_missing` | 열을 과제 4의 새 열로 바꾼다. 비율 열은 `no_date_rate = n_no_date / n_raw`, `in_occ_use_no_date_rate = n_in_occ_use_no_date / n_in_occ_use`. 설명: 게시일이 없으면 마감일을 쓰고, 둘 다 없는 공고와 마감일이 분석 기간 밖인 공고가 빠진다는 것. |
| `t01c_post_end_gap` | 파일 연도 × 구간별 공고 수와 그 연도 안의 비율(`share_in_year`). 설명: 게시일과 마감일이 모두 있는 공고에서 두 날짜가 얼마나 떨어져 있는지를 보여 주며, 마감일로 게시일을 대신한 공고의 날짜가 실제 게시일보다 그만큼 늦을 수 있다는 것. |
| `t02a_treated_by_period` | `unit`에 설정의 기간(`cfg["period"]["groups"]`)을 하나 더한다. 값은 `period`, 표시 이름은 "설정의 기간". 기간의 순서는 설정의 순서. 연·반기·분기는 그대로 둔다. |
| `t02c_occ_group_treated` | 제외 규칙 네 단계를 적용한 공고. 직종군(`keco2`)별 공고 수 `n`, 설정의 처치 정의로 고른 처치 공고 수 `n_treated`, 비교 공고 수 `n_control`, `meets_min`(`n_treated >= occupations.group_min_treated`). 직종군 이름(`name`)을 함께 싣는다. |
| `t09b_ksic_not_in_table` | `ksic5`가 있고 앞 3자리가 소분류표에 없는 공고를 `ksic5`별로 센다(소분류 열이 결측으로 바뀌었으므로 `ksic_sub`로는 고를 수 없다). 설명에 이 공고들의 `ksic_sub`가 설정에 따라 결측이라는 것을 적는다. |
| `t10b_occ_list_check` | 자료 쪽 목록에서 `occupations.exclude_keco2`의 중분류에 속한 세분류를 뺀 뒤 설정의 목록과 비교한다. 뺀 세분류는 `status` = "설정에서 뺀 중분류"로 따로 싣는다. 설명에 건수가 `data.years`의 연도만 센 것임을 적는다. |
| `t12_sido_other` | 설명에서 `전남광주`를 따로 센다는 문장을 `광주·전남`으로 합쳤다는 문장으로 바꾼다. |
| `t13_adjacent_candidates` | 제외 규칙 네 단계를 적용한 공고. 숙련코드 설명 파일에서 세부 분류(`detail_v4`)가 `skills.adjacent_candidate_details`에 든 숙련 가운데 AI 에이전트 숙련코드가 아닌 것마다 한 행: `skill_code`, `skill_name`, `detail`, `n`(그 숙련이 있는 공고 수), `rate`(`n` / 전체 공고 수), `n_treated`(생성형 AI 활용 처치 공고 가운데 그 숙련이 있는 수), `treated_rate`, `passes_sparse_filter`. 공고가 0건인 후보도 싣는다. `n`이 많은 순. 보고서에서 행 수를 자르지 않는다. |
| `t14_placebo_candidates` | 제외 규칙 네 단계를 적용한 공고. 설정의 희소 필터를 통과한 숙련(3절 `t03b`와 같은 공고와 같은 처치 공고) 가운데 대분류(`domain_v4`)가 `skills.placebo_candidate_domains`에 든 것마다 한 행: `skill_code`, `skill_name`, `domain`, `sub`, `n`, `rate`. `n`이 많은 순. 보고서에서 행 수를 자르지 않는다. |

| `t16_detail_by_arm` | 제외 규칙 네 단계를 적용한 공고를 설정의 처치 정의로 나눈 처치 공고와 비교 공고. 팔마다 공고 수, AI 에이전트 숙련코드를 뺀 숙련 수의 평균·중앙값·75%·90% 분위수, 본문 글자 수(`text_len`)의 평균·중앙값. 마지막 행 `같은 세분류·연도 안의 차이`: 세분류 × 연도마다 처치 공고 평균에서 비교 공고 평균을 뺀 값을 처치 공고 수로 가중 평균한 것(숙련 수와 글자 수 각각). |
| `t17_firm_treated` | 제외 규칙 네 단계를 적용한 공고. 처치 공고가 있는 기업 수, 기업당 처치 공고 수의 중앙값과 최댓값, 처치 공고가 많은 상위 10개 기업의 비중, 같은 기업·같은 연도에 비교 공고도 있는 처치 공고의 수와 비율과 그런 기업 수, 같은 기업·연도·세분류에 비교 공고도 있는 처치 공고의 수와 비율. `key`, `설명`, `값` 세 열의 긴 표. 기업 식별자는 싣지 않는다. |
| `t18_first_treated_firms` | 처치 공고를 처음 낸 연도(`first_year`)별 기업 수 `n_firms`, 그 가운데 그 전 연도에 비교 공고를 낸 기업 수 `n_firms_with_earlier`, 그 비교 공고 수 `n_earlier_postings`. |

`t02a_treated_by_period`, `t02b_treated_by_occ`의 처치 표시 열에 `kw_use`를 더한다. `t06a_overlap`에 `cross` = `kw_use × has_use × has_dev`를 더한다. `t06c_keyword_pattern_hits`는 `use` 사전의 행도 싣고, 보고서에서 행 수를 자르지 않는다. 7절 학습 시간 측정의 설명 변수(`TIMING_FEATURES`)에 `source`와 `text_len_log`(`log1p(text_len)`)를 더한다(결정 34, 35). `t16`~`t18`은 새 14절(제목은 과제 1에서 고친 설계 문서 6.6절 14번의 문장)에 싣는다.

2026-11-12의 중간 자료(옛 규칙)로 미리 계산한 참고값: 처치 공고의 숙련 수 평균 8.2, 비교 공고 4.6, 같은 세분류·연도 안의 차이 2.8, 처치 공고가 있는 기업 2,632개, 같은 기업·연도에 비교 공고도 있는 처치 공고 92%.

`t14`에는 처치 공고와 비교 공고의 출현율을 싣지 않는다. 설명에 이유를 적는다: placebo 숙련은 추정 결과를 보기 전에 골라야 하고, 두 집단의 출현율 차이는 추정 결과와 같은 방향의 정보이기 때문이다. placebo 숙련(`proposal.md` 7장 2번)은 생성형 AI 활용과 관계가 없어야 하는 숙련이며, 이 숙련에서 효과가 0으로 나오는지를 보아 분석 방법에 문제가 없는지 점검하는 데 쓴다는 설명도 한 문장으로 적는다.

- [ ] **1단계: 실패하는 시험을 쓴다** (`test_s00_report.py`)

가상 자료로 `scan`과 `build_tables`를 실행한 결과에 대한 단언이다. 가상 숙련코드 설명 파일(`synth._CODEBOOK`)에 세부 분류가 후보 이름인 숙련 두 개(하나는 가상 공고에 나오고 하나는 나오지 않음)와 대분류가 후보 이름인 숙련 하나를 더하고, `synth.write_all`의 설정에서 희소 필터 기준을 가상 자료에서 그 숙련이 통과하는 값으로 둔다.

```python
def test_t01b_and_t01c(tables):
    t = tables["t01b_post_start_missing"]
    assert {"n_no_date", "n_in_occ_from_end_in_period", "no_date_rate"} <= set(t.columns)
    year = t[t["level"] == "연도"]
    assert year["n_no_date"].sum() == 1 and year["n_in_occ_from_end_in_period"].sum() == 1
    gap = tables["t01c_post_end_gap"]
    assert gap.groupby("year")["share_in_year"].sum().round(6).eq(1).all()


def test_t02a_has_configured_periods(tables, synth_data):
    t = tables["t02a_treated_by_period"]
    names = [g["name"] for g in synth_data["cfg"]["period"]["groups"]]
    assert t.loc[t["unit"] == "period", "period"].tolist() == [n for n in names if n in set(t["period"])]


def test_t02c_marks_groups_below_the_minimum(tables, synth_data):
    t = tables["t02c_occ_group_treated"]
    assert (t["meets_min"] == (t["n_treated"] >= synth_data["cfg"]["occupations"]["group_min_treated"])).all()
    assert (t["n_treated"] + t["n_control"] <= t["n"]).all()


def test_t09b_counts_codes_missing_from_the_table(tables):
    assert tables["t09b_ksic_not_in_table"]["n"].sum() >= 1


def test_t13_lists_every_candidate_including_absent_ones(tables):
    t = tables["t13_adjacent_candidates"]
    assert set(t["skill_code"]) == {synth.ADJ_PRESENT, synth.ADJ_ABSENT}
    assert t.set_index("skill_code").loc[synth.ADJ_ABSENT, "n"] == 0
    assert not set(t["skill_code"]) & synth.AGENT_CODES            # AI 에이전트 숙련코드는 후보가 아니다


def test_t14_has_no_treated_columns(tables):
    t = tables["t14_placebo_candidates"]
    assert list(t.columns) == ["skill_code", "skill_name", "domain", "sub", "n", "rate"]
    assert synth.PLACEBO_CANDIDATE in set(t["skill_code"])


@pytest.mark.parametrize("key", ["adjacent_candidate_details", "placebo_candidate_domains"])
def test_unknown_candidate_name_is_an_error(synth_scan, key):        # synth_scan: scan까지 끝낸 고정물
    cfg = copy.deepcopy(synth_scan["cfg"])
    cfg["skills"][key] = ["없는이름"]
    with pytest.raises(ValueError, match="없는이름"):
        s00.build_tables(cfg, synth_scan["root"], synth_scan["s0_dir"])
```

```python
def test_t16_t17_t18(tables):
    t16 = tables["t16_detail_by_arm"].set_index("arm")
    assert {"처치 공고", "비교 공고", "같은 세분류·연도 안의 차이"} <= set(t16.index)
    assert t16.loc["처치 공고", "n"] + t16.loc["비교 공고", "n"] <= tables["t03b_sparse_filter_grid"]["n_postings"].iloc[0]
    t17 = tables["t17_firm_treated"].set_index("key")["값"]
    assert t17["n_treated_same_firm_year"] <= t17["n_treated"] and t17["n_treated_same_firm_year_occ"] <= t17["n_treated_same_firm_year"]
    assert not tables["t17_firm_treated"].astype(str).apply(lambda c: c.str.fullmatch(r"\d{10}")).any().any()   # 사업자번호가 없다
    t18 = tables["t18_first_treated_firms"]
    assert (t18["n_firms_with_earlier"] <= t18["n_firms"]).all()
```

보고서 글에 대한 단언 하나를 더한다: `write_report`가 쓴 글에 `28개`가 없고(분석 대상 세분류 수는 설정에서 센다), 새 표 이름 네 개(`t01c_post_end_gap`, `t02c_occ_group_treated`, `t13_adjacent_candidates`, `t14_placebo_candidates`)가 있다.

- [ ] **2단계: 시험이 실패하는지 확인한다**

실행: `-m pytest experiments/tests/test_s00_report.py -q` → 새 시험이 `KeyError`(표 없음)로 실패.

- [ ] **3단계: 구현한다**

  - 표 함수는 기존 방식(`_tNN_이름`)으로 더하고 `build_tables`에서 부른다. `REPORT_SECTIONS`에 새 표를 넣는다: `t01c`는 1절, `t02c`는 2절, `t13`과 `t14`는 새 12절(제목은 과제 1에서 고친 설계 문서 6.6절 12번의 문장). 기존 절은 옮기지 않는다. 1절의 제목은 설계 문서 6.6절 1번의 고친 문장으로 바꾼다.
  - 가상 자료의 상수를 `synth.py`에 둔다: `ADJ_PRESENT`, `ADJ_ABSENT`(세부 분류가 후보 이름인 가상 숙련 코드. 앞의 것만 가상 공고에 나온다), `PLACEBO_CANDIDATE`(대분류가 후보 이름이고 가상 자료의 희소 필터를 통과하는 숙련 코드), `AGENT_CODES`(가상 숙련코드 설명 파일의 AI 에이전트 숙련코드 집합).
  - `REPORT_MAX_ROWS_BY_TABLE`에 `t13`, `t14`를 `None`으로 넣는다.
  - 보고서의 절 제목과 설명 글에서 `28개`를 직접 적은 곳은 설정에서 센 수로 바꾼다.
  - `_POST_YEAR`("연도와 기간은 공고 게시일(`post_start`) 기준이다")를 "연도와 기간은 게시일(`post_start`, 없으면 `post_end`) 기준이다"로 고친다.
  - 후보 이름이 숙련코드 설명 파일에 하나도 없으면 없는 이름을 모두 적은 `ValueError`를 낸다.
  - 7절(학습 시간)의 설명 변수 목록 `TIMING_FEATURES`는 그대로 둔다(게시 시점 변수는 그대로 `quarter`이다).

- [ ] **4단계: 시험을 통과시키고 커밋한다**

실행: `-m pytest -q -W error` → 모두 통과.

```bash
git add experiments/scripts/s00_data_audit.py experiments/src/aabundle/skills.py experiments/tests
git commit -m "자료 점검 보고서에 게시일 집계, 설정의 기간, 직종군, 인접 숙련 후보, placebo 숙련 후보 표 추가"
```

---

### 과제 6: 산업 소분류별 AI 노출 지수

**미리 필요한 것:** `data/external/` 아래의 공개 자료 두 파일과 `experiments/crosswalks/ksic10_sub_to_isic4.csv`("실행 전에 필요한 것").

**파일:**
- 생성: `experiments/src/aabundle/confounders.py`, `experiments/tests/test_confounders.py`
- 커밋에 포함(이미 있음): `experiments/crosswalks/ksic10_sub_to_isic4.csv`
- 수정: `experiments/scripts/s00_data_audit.py`(`t15_aiie_coverage`), `experiments/tests/synth.py`, `README.md`

**주고받는 것:**
- 쓰는 것: 설정의 `data.aiie`, `data.naics_isic`, `data.ksic_isic`
- 만드는 것:
  - `confounders.load_naics_isic(path: Path) -> pd.DataFrame` — 열 `naics6`, `isic4`(둘 다 문자열). 첫 시트에서 머리글에 `NAICS`가 든 첫 열과 `ISIC 4.0`인 열을 쓴다. `naics6`이 숫자 6자리인 행만 남기고, `isic4`는 4자리가 되게 앞에 0을 채운다.
  - `confounders.load_aiie(path: Path, naics4_all: set[str]) -> pd.DataFrame` — 열 `naics4`(숫자 4자리 문자열, 겹치지 않음), `aiie`(실수). 시트 `LM AIIE`의 `NAICS`, `NAICS Description`, `Language Modeling AIIE` 열을 읽어 4자리 산업별 값으로 편다. 코드가 숫자이고 `000`으로 끝나면 3자리 수준의 항목이며 `naics4_all`(연계표에 나오는 모든 4자리 산업) 가운데 앞 3자리가 같은 산업 모두에 그 값을 준다. 그 밖의 숫자 코드는 앞 네 글자가 4자리 산업이다. 숫자가 아닌 글자가 든 코드는 여러 산업을 묶은 항목이며 설명에 나오는 4자리 숫자 모두에 그 값을 준다. 한 4자리 산업이 4자리 항목(또는 묶은 항목)과 3자리 항목 양쪽에서 값을 받으면 앞의 것을 쓴다. 4자리 항목이나 묶은 항목끼리 겹치면 겹친 코드를 적은 `ValueError`를 낸다.
  - `confounders.load_ksic_isic(path: Path) -> pd.DataFrame` — 열 `ksic_sub`(3자리 문자열), `isic_prefix`(문자열). `ksic_sub,ksic_name,isic,note` 머리글의 CSV를 문자열로 읽어 `isic` 열을 `;`로 나눠 한 줄에 앞자리 하나가 되게 편다. `isic`이 빈 소분류는 줄이 없다. `ksic_sub`가 겹치거나 3자리 숫자가 아니면 `ValueError`.
  - `confounders.aiie_by_ksic_sub(aiie, naics_isic, ksic_isic) -> pd.DataFrame` — 열 `ksic_sub`, `aiie`, `n_naics4`. 소분류마다, 그 소분류의 앞자리로 시작하는 ISIC 4자리 코드, 그 ISIC 코드에 이어진 NAICS 6자리의 앞 4자리 가운데 AIIE 값이 있는 서로 다른 산업을 모아 단순 평균한다. 이어진 산업이 없는 소분류는 행이 없다. 연계표의 어느 ISIC 코드와도 맞지 않는 앞자리가 있으면 그 소분류와 앞자리를 적은 `ValueError`를 낸다(표를 잘못 적은 것이다).
  - `confounders.attach_aiie(ksic_sub: pd.Series, table: pd.DataFrame) -> pd.Series` — 소분류로 값을 찾는다. 없으면 결측. 입력과 길이와 순서가 같다.
  - 표 `t15_aiie_coverage`: 제외 규칙 네 단계를 적용한 공고를 생성형 AI 활용 숙련 보유 여부(`has_use`)와 전체로 나눠, 공고 수 `n`, 값이 붙은 공고 수 `n_with_aiie`, 비율, 값의 분위수(10, 25, 50, 75, 90%)를 싣는다. 값이 붙은 소분류 수와 전체 소분류 수를 설명에 적는다.

- [ ] **1단계: 외부 자료를 받는다** (작업을 지휘하는 쪽이 한다)

```bash
mkdir -p data/external/aioe data/external/census
curl -sL -o "data/external/aioe/Language Modeling AIOE and AIIE.xlsx" "https://github.com/AIOE-Data/AIOE/raw/main/Language%20Modeling%20AIOE%20and%20AIIE.xlsx"
curl -sL -o data/external/census/2017_NAICS_to_ISIC_4.xlsx "https://www.census.gov/naics/concordances/2017_NAICS_to_ISIC_4.xlsx"
```

두 파일이 엑셀 파일로 열리고, 첫 파일의 `LM AIIE` 시트가 머리글을 빼고 250행인지 확인한다(2026-11-13에 확인한 값).

- [ ] **2단계: 판단으로 만든 표가 있는지 확인한다**

`experiments/crosswalks/ksic10_sub_to_isic4.csv`가 있고 머리글이 `ksic_sub,ksic_name,isic,note`, 줄 수가 머리글을 빼고 232인지 확인한다(2026-11-13에 만든 파일이다). 이 과제의 작업자는 표의 내용을 고치지 않는다. 고칠 곳이 보이면 보고에 적는다.

- [ ] **3단계: 실패하는 시험을 쓴다** (`test_confounders.py`)

```python
def test_aiie_by_ksic_sub_links_three_tables():
    aiie = pd.DataFrame({"naics4": ["5112", "5415", "6111", "5182", "5191"], "aiie": [1.0, 2.0, 3.0, 4.0, 6.0]})
    naics_isic = pd.DataFrame({
        "naics6": ["511210", "541511", "541512", "611110", "999999", "518210", "519130"],
        "isic4": ["5820", "6201", "6201", "8510", "6201", "6311", "6312"]})
    ksic_isic = pd.DataFrame({
        "ksic_sub": ["582", "620", "851", "631", "011"],
        "isic_prefix": ["582", "620", "851", "631", "011"]})      # 631은 앞자리 하나가 ISIC 6311과 6312 둘에 이어진다
    naics_isic = pd.concat([naics_isic, pd.DataFrame({"naics6": ["111110"], "isic4": ["0111"]})], ignore_index=True)
    out = confounders.aiie_by_ksic_sub(aiie, naics_isic, ksic_isic).set_index("ksic_sub")
    assert out.loc["582", "aiie"] == 1.0
    assert out.loc["620", "aiie"] == 2.0 and out.loc["620", "n_naics4"] == 1     # 같은 산업은 한 번만, AIIE에 없는 9999는 빠짐
    assert out.loc["631", "aiie"] == 5.0 and out.loc["631", "n_naics4"] == 2     # 두 산업의 단순 평균
    assert "011" not in out.index                                               # ISIC에는 이어지지만 AIIE 값이 없는 소분류


def test_prefix_that_matches_no_isic_code_is_an_error():
    aiie = pd.DataFrame({"naics4": ["5415"], "aiie": [2.0]})
    naics_isic = pd.DataFrame({"naics6": ["541511"], "isic4": ["6201"]})
    ksic_isic = pd.DataFrame({"ksic_sub": ["620", "999"], "isic_prefix": ["620", "777"]})
    with pytest.raises(ValueError, match="777"):
        confounders.aiie_by_ksic_sub(aiie, naics_isic, ksic_isic)


def test_load_aiie_expands_three_digit_and_combined_entries(tmp_path):
    """가상 AIIE 시트: `611100`(4자리) 1.0, `531000`(3자리 수준) 2.0, `3250A1`(설명에 `(3251 and 3259 only)`) 3.0,
    `531100`(4자리) 9.0. 3자리 항목은 연계표의 5311·5312·5313에 퍼지되, 4자리 항목이 따로 있는 5311은 9.0이다."""
    path = synth.write_external(tmp_path)["aiie"]
    a = confounders.load_aiie(path, {"6111", "5311", "5312", "5313", "3251", "3259", "3252"}).set_index("naics4")["aiie"]
    assert a["6111"] == 1.0 and a["5312"] == 2.0 and a["5313"] == 2.0 and a["5311"] == 9.0
    assert a["3251"] == 3.0 and a["3259"] == 3.0 and "3252" not in a.index
    assert a.index.is_unique


def test_attach_aiie_keeps_rows_and_leaves_missing():
    table = pd.DataFrame({"ksic_sub": ["620"], "aiie": [2.0], "n_naics4": [1]})
    out = confounders.attach_aiie(pd.Series(["620", "999", pd.NA, "620"], dtype="string"), table)
    assert out.tolist()[0] == 2.0 and out.tolist()[3] == 2.0 and out.isna().tolist() == [False, True, True, False]


def test_loaders_pad_codes_and_drop_unusable_rows(tmp_path):
    """가상 엑셀과 CSV로 읽기 함수를 시험한다.
    연계표: 숫자로 저장된 ISIC `111` → `0111`, NAICS가 `0`인 행은 버림.
    CSV: `isic`이 `011;012`인 줄은 두 줄로 펴지고, `isic`이 빈 소분류는 줄이 없다."""
    paths_ = synth.write_external(tmp_path)          # {"aiie": …, "naics_isic": …, "ksic_isic": …}
    n = confounders.load_naics_isic(paths_["naics_isic"])
    assert n["naics6"].str.fullmatch(r"\d{6}").all() and n["isic4"].str.fullmatch(r"\d{4}").all()
    assert "0111" in set(n["isic4"])
    k = confounders.load_ksic_isic(paths_["ksic_isic"])
    assert k[k["ksic_sub"] == "011"]["isic_prefix"].tolist() == ["011", "012"]
    assert "981" not in set(k["ksic_sub"])


def test_shipped_table_covers_every_sub_class_once():
    k = pd.read_csv(paths.ROOT / "experiments" / "crosswalks" / "ksic10_sub_to_isic4.csv", dtype=str, keep_default_na=False)
    assert list(k.columns) == ["ksic_sub", "ksic_name", "isic", "note"]
    assert len(k) == 232 and k["ksic_sub"].is_unique and k["ksic_sub"].str.fullmatch(r"\d{3}").all()
    assert k.loc[k["isic"] != "", "isic"].str.fullmatch(r"\d{2,4}(;\d{2,4})*").all()


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="ksic10_sub_to_isic4.csv"):
        confounders.load_ksic_isic(tmp_path / "ksic10_sub_to_isic4.csv")
```

`synth.write_external(root) -> dict`를 더한다: 실제 파일과 같은 시트 이름·머리글(미국 통계국 연계표의 머리글 `'2017\nNAICS\nUS  '`, `'ISIC 4.0'` 포함)을 가진 가상 파일 세 개를 쓴다(AIIE 시트에는 4자리 항목, 3자리 수준 항목, 묶은 항목이 모두 있고, 산업 연계표는 실제와 같은 네 열이다). `synth.write_all`은 이 함수를 불러 설정의 세 경로를 가상 파일로 돌린다.

`test_s00_report.py`에 더한다: `t15_aiie_coverage`의 `n_with_aiie <= n`, 전체 행의 `n`이 제외 규칙 네 단계를 적용한 공고 수와 같음, 산업코드가 없는 가상 공고가 값이 붙지 않은 쪽에 세어짐. 과제 5의 보고서 글 단언에 `t15_aiie_coverage`를 더한다.

- [ ] **4단계: 시험이 실패하는지 확인하고 구현한다**

실행: `-m pytest experiments/tests/test_confounders.py -q` → `ModuleNotFoundError`로 실패. 구현한 뒤 통과.
`AIIE`의 `naics4`가 겹치면(서로 다른 여섯 글자 코드가 같은 네 글자로 줄면) 겹친 코드를 적은 `ValueError`를 낸다.
`t15_aiie_coverage`는 `build_tables`에서 세 파일을 읽어 만든다(`load_aiie`의 `naics4_all`은 연계표의 `naics6` 앞 4자리 집합이다). 파일 경로 항목(`data.aiie` 등)이 설정에 없거나 파일이 없으면 이 표만 건너뛰고, 보고서의 해당 절에 "외부 자료가 없어 만들지 않았다"와 없는 파일의 경로를 적는다(다른 표는 그대로 만든다). 새 절 "13. 산업별 AI 노출 지수의 결합"에 싣는다.

- [ ] **5단계: 실제 파일로 형식을 확인한다** (작업을 지휘하는 쪽이 한다. 건수만 출력)

세 읽기 함수와 `aiie_by_ksic_sub`를 실제 파일에 적용해 다음을 출력한다: 4자리 산업으로 편 AIIE의 산업 수, 연계표의 줄 수, 값이 붙은 소분류 수, 값의 최소·중앙·최대. 2026-11-13에 같은 방법으로 미리 계산한 값은 4자리 산업 289개, 값이 붙은 소분류 221개, 최소 −1.740, 중앙 −0.227, 최대 2.053이다. 이 값과 다르면 멈추고 원인을 확인한다.

- [ ] **6단계: 커밋한다**

`README.md`에 "외부 자료" 항목(공개 자료 두 파일의 출처, 두는 위치, 받은 날, 판단으로 만든 산업 연계표의 위치와 만든 방법, 잇는 순서)을 더한다. `data/external/`은 `/data/`가 git 제외 목록에 있으므로 커밋되지 않는다.

```bash
git add experiments/src/aabundle/confounders.py experiments/crosswalks experiments/scripts/s00_data_audit.py experiments/tests README.md
git commit -m "산업 소분류별 AI 노출 지수(Felten 등의 AIIE 언어 모형 판)를 만드는 모듈과 결합 비율 표 추가"
```

---

### 과제 7: 실제 자료로 자료 점검을 다시 실행하고 보고

작업을 지휘하는 쪽이 한다(실행이 길어 작업자에게 맡기지 않는다).

**파일:**
- 수정: `docs/s0-data-audit.md`(스크립트가 다시 씀), `README.md`(실행 시간), `docs/decisions.md`(실행 결과로 채울 건수), `experiments/tests/test_realdata.py`(연도 목록이나 건수를 직접 적은 곳이 있으면)

- [ ] **1단계: 전체 시험을 실행한다**

실행: `-m pytest -q -W error` → 모두 통과. `git status --short`에 `?? test.ipynb`만 보인다.

- [ ] **2단계: `all`을 실행한다** (공고 파일 네 개, 약 60분으로 예상. 그동안 컴퓨터를 끄지 않는다)

```bash
PYTHONNOUSERSITE=1 PYTHONUTF8=1 "$(conda info --base)/envs/aabundle/python.exe" -X utf8 experiments/scripts/s00_data_audit.py all
```

끝난 코드가 0이고 마지막 줄이 "보고서를 썼습니다"여야 한다.

- [ ] **3단계: 결과를 대조한다** (집계 값만 본다)

| 확인할 것 | 기대값 |
|---|---|
| 연도 파일별 읽은 행 수 | 2023년 3,218,121, 2024년 3,212,594, 2025년 3,570,045, 2026년 2,930,306 (2026-11-12 실행과 같음) |
| `t01b`의 `n_no_post_start` (파일 연도별) | 123,380 / 132,987 / 789,979 / 261,302 (2026-11-12 실행과 같음) |
| `t01b`의 `n_no_date` | `n_no_post_start`보다 작거나 같음. 2025년은 약 33만 건(2026-11-12에 센 값: 게시일이 없는 789,979건 가운데 마감일이 있는 것 456,970건) |
| `t04b_ked_lines` | 9,271,690줄, 버린 줄 0, 비운 항목 0, 기업 8,679,456개 |
| `t10b_occ_list_check` | "양쪽에 있음" 25개, "설정에서 뺀 중분류" 3개. "자료에만 있음"이나 "설정에만 있음"이 있으면 저자에게 알릴 항목으로 적는다(2021~2022년을 빼고 세어 30건에 못 미치게 된 세분류가 있을 수 있다) |
| `t12_sido_other` | `광주`, `전남`, `전남광주` 행이 없고 `광주·전남` 행이 있음 |
| `t09a_by_ksic_sub` | `분류표에 없음` 행이 없음. `t09b`의 합계가 0보다 큼 |
| `t02a`의 설정의 기간 | 세 기간이 설정의 순서로 있음. 처치 공고 수가 연 단위 행의 합과 같음(초기 = 2023년 + 2024년) |
| `t03d_match_stats` | 찾지 못한 코드 0, 조각 수가 다른 칸 0 |
| `t13_adjacent_candidates` | 56행 |
| `t16_detail_by_arm` | 처치 공고의 숙련 수 평균이 비교 공고보다 큼(2026-11-12 참고값: 8.2와 4.6) |
| `t06c_keyword_pattern_hits` | `use` 사전의 표현 34개가 모두 있음. `LLM` 표현의 공고 수가 `Chat\s*GPT`보다 크게 많으면 잘못 걸리는 낱말이 있는지 의심한다 |
| `base.parquet` | `text_len`의 최솟값 0 이상. 열 이름에 `detail`, `title`이 없음 |
| `scan_meta.json` | 커밋이 실행 시점의 `infra-2` 커밋, 커밋하지 않은 변경 없음 |
| 보고서와 표 | 숫자 10자리 이상의 정수 없음, `\d{6}-\d{7}` 형태 없음, `detail`·`title` 없음 |

- [ ] **4단계: 문서를 고치고 커밋한다**

`README.md`의 실행 시간을 이번 실행 값으로 바꾼다. `docs/decisions.md`의 6.7-15, 6.7-16 근거 칸에 이번 실행의 건수(마감일로 게시일을 대신해 살아난 공고 수와 그 가운데 생성형 AI 활용 숙련 공고 수, `광주·전남` 공고 수)를 더한다.

```bash
git add docs/s0-data-audit.md README.md docs/decisions.md experiments/tests
git commit -m "확정 사항을 반영한 자료 점검 보고서"
```

- [ ] **5단계: 저자에게 보고한다**

보고서 경로와 다음을 알린다.

  - 제외 규칙 네 단계를 적용한 공고 수와 처치 공고 수(기간별, 직종군별), 2026-11-12 보고서와 달라진 크기
  - 마감일로 게시일을 대신해 살아난 공고 수, 두 날짜의 간격 분포
  - 희소 필터를 통과한 숙련 수와 새 학습 시간 추정
  - 정할 것 두 가지: 인접 숙련 목록(`t13_adjacent_candidates`), placebo 숙련 목록(`t14_placebo_candidates`)
  - 생성형 AI 활용 키워드 사전의 표현별 공고 수(`t06c`)와, 이 사전 때문에 비교 공고에서 빠진 공고 수. 뺄 표현이 있는지 물어본다
  - 처치 공고와 비교 공고의 숙련 수·글자 수 차이(`t16`), 같은 기업 안의 비교와 처음 내기 전 시기의 점검이 가능한 표본 크기(`t17`, `t18`)
  - 산업별 AI 노출 지수가 붙은 공고의 비율(`t15_aiie_coverage`)과, 판단으로 만든 산업 연계표를 검토해 달라는 요청
  - 3단계의 대조에서 기대값과 달랐던 것

여기서 멈춘다. 세 번째 계획(추정 모듈, 단계 스크립트, 논문 원고 틀)은 저자가 두 목록을 정한 뒤 작성한다.
