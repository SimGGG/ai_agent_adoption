"""시험용 가상 자료. 뒤 과제의 시험이 아래 공고 17건의 내용에 기대므로 값을 바꾸지 않는다.

p13, p14는 게시일 규칙(설계 문서 결정 25)을 시험하려고 나중에 더한 공고이다. p13은 게시일이 없고
마감일만 있으며, p14는 둘 다 없다. 둘 다 분석 대상 세분류이고 사업자번호와 숙련이 있다.
같은 때 산업코드와 근무지 순환 목록에 값을 더해(분류표에 없는 산업코드, 광주와 전남광주 근무지),
p06 이후 공고의 산업코드와 p05 이후 공고의 근무지가 바뀌었다.

p15, p16은 생성형 AI 활용 키워드 사전(설계 문서 결정 36)을 시험하려고 더한 공고이다. p15는 본문에
`ChatGPT 활용`이 있고 AI 에이전트 숙련코드는 없다. p16은 본문의 `Fulfillment`에 `llm`이라는 글자가
들어 있지만 낱말의 일부이므로 사전에 걸리지 않아야 한다. 둘 다 분석 대상 세분류이고 사업자번호,
게시일, AI 에이전트 숙련이 아닌 숙련이 있다.

p17은 자료 점검 1단계가 날짜로 쓰지 못하는 마감일을 세는지 시험하려고 과제 4에서 더한 공고이다. 게시일이
없고 마감일이 `9999-12-31`(날짜로 쓸 수 있는 범위 밖)이어서 게시일을 정할 수 없다. 분석 대상 세분류이고
사업자번호와 AI 에이전트 숙련이 아닌 숙련이 있다.

여기서 나온 수치는 연구 결과가 아니다.
"""
import datetime as dt
from pathlib import Path

import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import paths

SKILL_KINDS = ["occupation", "transversal", "qualification", "competency", "soft", "physical"]

# (posting_id, keco_2025, bzno, post_start, detail, {열: [(코드, 숙련명), ...]})
_POSTINGS = [
    ("p01", "1332", "1000000001", dt.date(2025, 1, 10), "AI 에이전트 개발",
     {"occupation": [("IT-05-0398", "RAG 파이프라인 구축하기"), ("IT-01-0001", "숙련A")]}),
    ("p02", "0262", "1000000001", dt.date(2025, 2, 10), "마케팅 담당",
     {"occupation": [("IT-05-0410", "ChatGPT 활용하기"), ("SM-01-0001", "숙련B")]}),
    ("p03", "0262", "1000000002", dt.date(2025, 4, 10), "보험 에이전트 모집",
     {"occupation": [("SM-01-0001", "숙련B")]}),
    ("p04", "0262", None, dt.date(2025, 4, 11), "사무 보조",
     {"occupation": [("IT-05-0396", "프롬프트 엔지니어링하기")]}),
    ("p05", "1332", "1000000003", dt.date(2025, 7, 1), "LangChain 경험자",
     {"occupation": [("IT-05-0413", "LangChain 프레임워크 활용하기"), ("IT-05-0410", "ChatGPT 활용하기")]}),
    ("p06", "1332", "1000000003", dt.date(2025, 7, 2), None, {}),
    ("p07", "9052", "1000000004", dt.date(2025, 7, 3), "AI agent",
     {"occupation": [("IT-01-0001", "숙련A")]}),
    ("p08", None, "1000000004", dt.date(2025, 7, 4), "일반",
     {"occupation": [("SM-01-0001", "숙련B")]}),
    ("p09", "0262", "1000000005", dt.date(2025, 10, 1), "multi-agent 시스템",
     {"transversal": [("SS-01-02-0009", "러시아어 구사하기")]}),
    ("p10", "0262", "1000000005", dt.date(2025, 10, 2), "일반",
     {"soft": [("SS-99-99-9999", "설명 파일에 없는 숙련")]}),
    ("p11", "1332", "1000000006", dt.date(2025, 11, 1), "Claude Code 사용 경험",
     {"occupation": [("IT-01-0001", "숙련A")], "qualification": [("XT-02-009-0000", "학사 학위")]}),
    # 코드는 둘, 숙련명은 하나: 개수가 맞지 않는 공고
    ("p12", "1332", "1000000006", dt.date(2025, 11, 2), "일반",
     {"occupation": [("IT-01-0001", "숙련A"), ("SM-01-0001", None)]}),
    # 게시일이 없고 마감일만 있는 공고
    ("p13", "1332", "1000000001", None, "일반",
     {"occupation": [("IT-01-0001", "숙련A")]}),
    # 게시일과 마감일이 모두 없는 공고
    ("p14", "0262", "1000000002", None, "일반",
     {"occupation": [("SM-01-0001", "숙련B")]}),
    # 생성형 AI 활용 키워드에만 걸리는 공고(숙련코드는 AI 에이전트 숙련이 아님)
    ("p15", "0262", "1000000003", dt.date(2025, 8, 1), "ChatGPT 활용 문서 작성",
     {"occupation": [("SM-01-0001", "숙련B")]}),
    # 낱말 안에 `llm`이 들어 있지만 어느 사전에도 걸리지 않는 공고
    ("p16", "1332", "1000000005", dt.date(2025, 9, 1), "Fulfillment 센터 물류 관리",
     {"occupation": [("IT-01-0001", "숙련A")]}),
    # 게시일이 없고 마감일이 날짜로 쓸 수 없는 값(9999-12-31)인 공고
    ("p17", "1332", "1000000006", None, "일반",
     {"occupation": [("IT-01-0001", "숙련A")]}),
]

ID_END_ONLY = "p13"
ID_NO_DATE = "p14"
ID_USE_KEYWORD_ONLY = "p15"
ID_FULFILLMENT = "p16"
ID_END_UNUSABLE = "p17"
# 마감일은 게시일 30일 뒤로 준다. 게시일이 없는 공고의 마감일은 아래 표에서 정한다(없으면 결측).
_POST_END_DAYS = 30
_POST_END_WITHOUT_START = {ID_END_ONLY: dt.date(2025, 12, 20), ID_END_UNUSABLE: dt.date(9999, 12, 31)}
# 가상 공고의 연도(2025)가 드는 기간의 이름. write_all이 설정의 period.groups에 넣는다.
PERIOD_NAME = "가상 기간"
# write_all이 설정의 occupations.keco4에 넣는 분석 대상 세분류
OCCUPATIONS = ["1332", "0262"]

_KSIC_CYCLE = ["62010", "Q86201", "1220", "C30", None, "47121"]   # 47121: 가상 소분류표에 없는 소분류(471)
_REGIONS = ["서울 마포구", "경기도 성남시", "부산전체", None, "광주 북구", "전남광주통합특별시 광산구"]
_SALARIES = [30_000_000.0, None, 45_000_000.0, 60_000_000.0]
_FIRM_EMP = [10, 250, None, 1200]
_EMP_TYPES = ["regular", "fixed_term", "part_time"]
_EDU = ["IRRELEVANT", "ASSOCIATE", "BACHELOR", "HIGH_SCHOOL"]
_CAREER = ["IRRELEVANT", "ENTRY", "EXPERIENCED_SPECIFIED"]
_SOURCES = ["잡사이트A", "잡사이트B", "잡사이트C"]


def _post_end(p) -> dt.date | None:
    if p[3] is None:
        return _POST_END_WITHOUT_START.get(p[0])
    return p[3] + dt.timedelta(days=_POST_END_DAYS)


def _usable(d: dt.date | None) -> bool:
    """날짜가 있고 게시일로 쓸 수 있는 범위(`datetime64[ns]`, 1677~2262년) 안인지."""
    return d is not None and pd.Timestamp.min.date() <= d <= pd.Timestamp.max.date()


# 가상 공고 목록에서 센 공고 수. 시험이 숫자를 직접 적지 않고 이 값과 비교한다.
# 게시일로 쓸 날짜가 첫 날짜 열(post_start)에 없는 공고
N_NO_POST_START = sum(not _usable(p[3]) for p in _POSTINGS)
# 두 날짜 열 어디에도 게시일로 쓸 날짜가 없는 공고
N_NO_DATE = sum(not _usable(p[3]) and not _usable(_post_end(p)) for p in _POSTINGS)
# 첫 날짜 열에 날짜가 없고 마감일에 값이 있지만 날짜로 쓸 수 없는 공고
N_END_UNUSABLE = sum(not _usable(p[3]) and _post_end(p) is not None and not _usable(_post_end(p)) for p in _POSTINGS)
# 분석 대상 세분류이고 게시일과 마감일이 모두 날짜로 읽히는 공고
N_IN_OCC_BOTH_DATES = sum(p[1] in OCCUPATIONS and _usable(p[3]) and _usable(_post_end(p)) for p in _POSTINGS)


def _postings_table() -> pa.Table:
    n = len(_POSTINGS)
    cols = {
        "posting_id": [p[0] for p in _POSTINGS],
        "source": [_SOURCES[i % 3] for i in range(n)],
        "post_start": [p[3] for p in _POSTINGS],
        "post_end": [_post_end(p) for p in _POSTINGS],
        "bzno": [p[2] for p in _POSTINGS],
        "ksic10": [_KSIC_CYCLE[i % len(_KSIC_CYCLE)] for i in range(n)],
        "firm_emp": [_FIRM_EMP[i % 4] for i in range(n)],
        "keco_2025": [p[1] for p in _POSTINGS],
        "salary_annual_min": [_SALARIES[i % 4] for i in range(n)],
        "emp_type": [_EMP_TYPES[i % 3] for i in range(n)],
        "edu_level": [_EDU[i % 4] for i in range(n)],
        "career_type": [_CAREER[i % 3] for i in range(n)],
        "career_years_min": [float(i % 4) for i in range(n)],
        "region": [_REGIONS[i % len(_REGIONS)] for i in range(n)],
        "dup_count": [1 + i % 3 for i in range(n)],
        "ai_peripheral": [i % 5 == 0 for i in range(n)],
        "detail": [p[4] for p in _POSTINGS],
    }
    types = {
        "posting_id": pa.string(), "source": pa.string(), "post_start": pa.date32(), "post_end": pa.date32(),
        "bzno": pa.string(), "ksic10": pa.string(), "firm_emp": pa.int64(),
        "keco_2025": pa.string(), "salary_annual_min": pa.float64(), "emp_type": pa.string(),
        "edu_level": pa.string(), "career_type": pa.string(), "career_years_min": pa.float64(),
        "region": pa.string(), "dup_count": pa.int64(), "ai_peripheral": pa.bool_(),
        "detail": pa.string(),
    }
    for kind in SKILL_KINDS:
        codes, names = [], []
        for p in _POSTINGS:
            items = p[5].get(kind)
            if not items:
                codes.append(None)
                names.append(None)
                continue
            codes.append("|".join(c for c, _ in items))
            names.append("|".join(nm for _, nm in items if nm is not None))
        cols[f"skills_{kind}"] = codes
        cols[f"skills_{kind}_name"] = names
        types[f"skills_{kind}"] = pa.string()
        types[f"skills_{kind}_name"] = pa.string()
    return pa.table({k: pa.array(v, type=types[k]) for k, v in cols.items()})


_CODEBOOK = [
    ("IT-01-0001", "IT-01-001-0001", "숙련A", "IT·디지털·컴퓨팅", "소프트웨어 개발"),
    ("SM-01-0001", "SM-01-001-0001", "숙련B", "영업·마케팅·대외관계", "영업"),
    ("IT-05-0396", "IT-06-012-0028", "프롬프트 엔지니어링하기", "IT·디지털·컴퓨팅", "AI·머신러닝 시스템"),
    ("IT-05-0398", "IT-06-012-0030", "RAG 파이프라인 구축하기", "IT·디지털·컴퓨팅", "AI·머신러닝 시스템"),
    ("IT-05-0410", "IT-06-012-0044", "ChatGPT 활용하기", "IT·디지털·컴퓨팅", "AI·머신러닝 시스템"),
    ("IT-05-0413", "IT-06-012-0047", "LangChain 프레임워크 활용하기", "IT·디지털·컴퓨팅", "AI·머신러닝 시스템"),
    ("IT-05-0200", "IT-06-011-0001", "MLOps 구축하기", "IT·디지털·컴퓨팅", "AI·머신러닝 시스템"),
    ("SS-02-0009", "SS-01-02-0009", "러시아어 구사하기", "의사소통", "외국어"),
    ("XT-01-0001", "XT-02-009-0000", "학사 학위", "자격·인증", "학위"),
]


def _write_codebook(path: Path) -> None:
    rows = []
    for i, (v4, v31, name, dom, sub) in enumerate(_CODEBOOK, start=1):
        rows.append({
            "skill_id": f"S{i:04d}", "skill_code_v4": v4, "skill_code_v31": v31,
            "skill_name": name, "domain_v4": dom, "sub_v4": sub,
            "detail_v4": f"{name} 세부", "domain_v31": dom, "sub_v31": sub,
            "detail_v31": f"{name} 세부", "description": f"{name}에 대한 설명",
        })
    pd.DataFrame(rows, dtype=str).to_csv(path, index=False, encoding="utf-8-sig")


def _text_row(ws, values) -> None:
    """모든 칸을 문자열로 쓴다. 앞자리 0이 사라지지 않게 하기 위해서다."""
    ws.append(values)
    for cell in ws[ws.max_row]:
        if cell.value is not None:
            cell.value = str(cell.value)
            cell.data_type = "s"


def _write_ksic(path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["한국표준산업분류 (가상)"])
    ws.append([])
    ws.append(["대분류", "대분류명", "중분류", "중분류명", "소분류", "소분류명", "세분류", "세분류명", "세세분류", "세세분류명"])
    # 병합된 칸은 묶음의 첫 줄에만 값이 있다. 소분류마다 한 줄이므로 각 줄에 소분류가 있다.
    _text_row(ws, ["J", "정보통신업", "62", "컴퓨터 프로그래밍업", "620", "컴퓨터 프로그래밍업", "6201", "컴퓨터 프로그래밍", "62010", "컴퓨터 프로그래밍업"])
    _text_row(ws, ["Q", "보건업", "86", "보건업", "862", "병원", "8620", "병원", "86201", "종합병원"])
    _text_row(ws, ["A", "농업", "01", "작물 재배업", "012", "다년생 작물 재배업", "0121", "과실작물 재배업", "01210", "포도 재배업"])
    wb.save(path)


def _write_keco(path: Path) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["한국고용직업분류 (가상)"])
    ws.append([])
    ws.append(["대분류코드", "대분류명", "중분류코드", "중분류명", "소분류코드", "소분류명", "세분류코드", "세분류명"])
    _text_row(ws, ["1", "연구직 및 공학 기술직", "13", "정보통신 연구개발직 및 공학기술직", "133", "소프트웨어 개발자", "1332", "응용 소프트웨어 개발자"])
    _text_row(ws, ["0", "경영·사무·금융·보험직", "02", "경영·행정·사무직", "026", "경영지원 사무원", "0262", "영업 및 마케팅 사무원"])
    _text_row(ws, ["9", "농림어업직", "90", "농림어업직", "905", "농림어업 단순 종사원", "9052", "임업 관련 단순 종사원"])
    _text_row(ws, [None, None, None, None, None, None, "3개", None])
    wb.save(path)


# KED 가상 파일: 사업자번호 1000000002는 옛 행과 새 행이 있고, 새 행(정보일자 20230501)의 값이 남아야 한다.
KED_SIZE_LATEST = "L2"
KED_SIZE_OLD = "S1"


def _ked_row(bzno, estab, info_date, size="M1", ipo="N", venture="Y", extra_at=None,
             extra_n=1, overrides=None) -> bytes:
    """48개 항목을 `|`로 이은 한 줄을 CP949 바이트로 돌려준다. 18번 항목은 항상 `SECRET18`이다.

    `extra_at`을 주면 그 번호의 항목 안에 `|`를 `extra_n`개 더 넣는다. `overrides`는 {항목 번호: 값}이다.
    """
    f = [f"항목{i:02d}" if i % 2 else f"v{i:02d}" for i in range(48)]
    f[5], f[12], f[18], f[19] = size, venture, "SECRET18", estab
    f[20], f[31], f[32] = ipo, info_date, bzno
    for i, v in (overrides or {}).items():
        f[i] = v
    if extra_at is not None:
        f[extra_at] = f[extra_at] + "|추가" * extra_n
    return ("|".join(f) + "\r\n").encode("cp949")


def _write_ked(path: Path) -> None:
    rows = [
        _ked_row("1000000001", "20100315", "20230501"),
        _ked_row("1000000002", "19991201", "20230501", size=KED_SIZE_LATEST),
        _ked_row("1000000002", "19991201", "20220101", size=KED_SIZE_OLD),
        _ked_row("1000000003", "", "20230501", size="", ipo="", venture=""),
        _ked_row("12345", "20000101", "20230501"),
        _ked_row("1000000005", "20150101", "20230501", extra_at=5),
        _ked_row("1000000006", "SECRET18", "20230501"),
        _ked_row("1000000008", "20120101", "20230501", extra_at=40),
        # 행 9: 18번 앞에 `|`가 둘 더 있어 항목이 두 칸 밀린다. 18번 값이 ked_ipo 자리(20)로 오고,
        # 30번 항목의 값이 사업자번호 자리(32)로 와서 10자리 검사를 통과한다.
        _ked_row("1", "20160101", "20230501", extra_at=5, extra_n=2,
                 overrides={29: "20230501", 30: "1000000009"}),
        # 행 10: 구분자는 47개이지만 6번 항목 안에 줄 끝이 아닌 캐리지 리턴 문자(CR)가 하나 있다.
        # CR를 줄 끝으로 읽는 도구라면 줄이 둘로 갈라져 둘째 줄의 12번 자리에 18번 값이,
        # 32번 자리에 38번 값이 온다.
        _ked_row("1000000010", "20170101", "20230501",
                 overrides={6: "a\rb", 38: "9999999999"}),
    ]
    path.write_bytes(b"".join(rows))


def write_all(root: Path) -> dict:
    """`root` 아래에 가상 자료 한 벌을 쓰고 경로와 설정을 담은 사전을 돌려준다.

    폴더 배치는 실제와 같다(`root/data/postings/…`, `root/data/ked/…`).
    KED 파일은 8행짜리 가상 파일로 쓴다(CP949, 18번 항목은 모두 `SECRET18`).
    """
    root = Path(root)
    postings_dir = root / "data" / "postings"
    ked_dir = root / "data" / "ked"
    postings_dir.mkdir(parents=True, exist_ok=True)
    ked_dir.mkdir(parents=True, exist_ok=True)

    pq.write_table(_postings_table(), postings_dir / "pfile_2025.parquet", row_group_size=5)
    codebook = postings_dir / "ojp_codebook_skill.csv"
    ksic = postings_dir / "KSIC10.xlsx"
    keco = postings_dir / "KECO2025.xlsx"
    _write_codebook(codebook)
    _write_ksic(ksic)
    _write_keco(keco)
    _write_ked(ked_dir / "KED5002.txt")

    cfg = paths.load_config()
    cfg["data"]["years"] = [2025]
    cfg["data"]["codebook"] = "data/postings/ojp_codebook_skill.csv"
    cfg["data"]["ksic"] = "data/postings/KSIC10.xlsx"
    cfg["data"]["keco"] = "data/postings/KECO2025.xlsx"
    cfg["data"]["ked_firm"] = "data/ked/KED5002.txt"
    cfg["occupations"]["keco4"] = list(OCCUPATIONS)
    cfg["occupations"]["exclude_keco2"] = ["01", "61"]
    # 가상 공고의 게시일(게시일이 없으면 마감일)은 모두 2025년이다
    cfg["period"]["start"], cfg["period"]["end"] = "2025-01-01", "2025-12-31"
    cfg["period"]["groups"] = [{"name": PERIOD_NAME, "years": [2025]}]
    cfg["treatment"]["agent_skills"]["move_to_dev"] = ["IT-05-0413"]
    return {
        "root": root,
        "postings_dir": postings_dir,
        "codebook": codebook,
        "ksic": ksic,
        "keco": keco,
        "ked_firm": ked_dir / "KED5002.txt",
        "cfg": cfg,
    }
