"""
HR Insight Dashboard 더미 데이터 생성기
- 1,500명 샘플 × 2개 파일 (다른 시드)
- 기존 100명 샘플과 동일한 컬럼 구조/값 도메인
- 학습 가능한 강한 시그널 부여 (목표 ROC AUC ~0.80)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ----- 카테고리 후보 (기존 샘플 값 도메인) -----
GENDER = ["M", "F"]
POSITION = ["사원", "주임", "대리", "과장", "차장", "부장"]
TITLE = ["팀원", "파트장", "팀장", "센터장"]
JOB = ["개발", "연구", "영업", "마케팅", "재무", "인사", "생산", "품질", "구매", "전략"]
ORG = ["IT본부", "경영본부", "R&D본부", "생산본부", "영업본부", "마케팅본부"]
TEAM = ["A팀", "B팀", "C팀", "D팀", "E팀"]
HIRE_TYPE = ["정규직", "계약직"]
REGION = ["서울", "경기", "인천", "부산", "대전", "광주", "대구", "울산"]
NCT = ["Y", "N"]
EDU = ["고졸", "학사", "석사", "박사"]
MAJOR = ["기계", "전자", "컴퓨터", "화학", "디자인", "경영", "기타"]
ROLE = ["연구", "기획", "관리", "분석", "운영"]
MARRIAGE = ["미혼", "기혼"]
YN = ["Y", "N"]
GRADE = ["SS", "EE", "AA", "BB", "CC", "UN"]
RESIGN_REASON = ["이직", "창업", "학업", "개인사유", "급여불만", "업무과중", "조직문화", "건강"]
DEST = ["에버모터스", "브라이트랩", "센트로테크", "노바케미컬", "퓨처AI", "비전소프트",
        "한울제약", "프라임컨설팅", "스카이로지스", "미상"]

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허남심노")
GIVEN_1 = list("민서지예준현우윤도하수은혜재태성영진주아연")
GIVEN_2 = list("준호영서윤아진희석민수경원빈호철국식찬빈")


def generate(N, seed, start_emp_id, out_csv, out_xlsx):
    rng = np.random.default_rng(seed=seed)

    def make_name():
        return rng.choice(SURNAMES) + rng.choice(GIVEN_1) + rng.choice(GIVEN_2)

    def pick(choices, p=None):
        return rng.choice(choices, p=p)

    # 기본 인적
    emp_ids = [start_emp_id + i for i in range(N)]
    names = [make_name() for _ in range(N)]
    gender = [pick(GENDER, p=[0.62, 0.38]) for _ in range(N)]
    age = rng.integers(25, 60, size=N)
    marriage = [pick(MARRIAGE, p=[0.45, 0.55]) for _ in range(N)]
    edu = [pick(EDU, p=[0.10, 0.55, 0.28, 0.07]) for _ in range(N)]
    major = [pick(MAJOR, p=[0.18, 0.15, 0.20, 0.12, 0.10, 0.15, 0.10]) for _ in range(N)]

    # 직무/조직
    position = [pick(POSITION, p=[0.20, 0.18, 0.22, 0.22, 0.12, 0.06]) for _ in range(N)]
    job = [pick(JOB) for _ in range(N)]
    title = [pick(TITLE, p=[0.72, 0.17, 0.09, 0.02]) for _ in range(N)]
    org = [pick(ORG) for _ in range(N)]
    team = [pick(TEAM) for _ in range(N)]
    hire_type = [pick(HIRE_TYPE, p=[0.78, 0.22]) for _ in range(N)]
    region = [pick(REGION, p=[0.42, 0.18, 0.07, 0.12, 0.08, 0.05, 0.05, 0.03]) for _ in range(N)]
    nct = [pick(NCT, p=[0.30, 0.70]) for _ in range(N)]
    role = [pick(ROLE) for _ in range(N)]

    # 근속/평가/보상
    work_years = np.clip(rng.normal(8, 5, size=N), 0.0, 30).round(1)
    promo_years = np.clip(rng.normal(3, 2, size=N), 0.0, 12).round(1)
    pre_jobs = rng.integers(0, 5, size=N)
    base_salary = (rng.normal(5200, 1400, size=N)
                   + (np.array([POSITION.index(p) for p in position]) * 900)
                  ).clip(2800, 12000).round(0).astype(int)
    competency = rng.integers(1, 6, size=N)

    grade = [pick(GRADE, p=[0.05, 0.22, 0.35, 0.22, 0.10, 0.06]) for _ in range(N)]
    core = [pick(YN, p=[0.30, 0.70]) for _ in range(N)]
    incentive = [pick(YN, p=[0.65, 0.35]) for _ in range(N)]
    overtime = [pick(YN, p=[0.35, 0.65]) for _ in range(N)]
    wfh = [pick(YN, p=[0.50, 0.50]) for _ in range(N)]
    pre_hired = [pick(YN, p=[0.40, 0.60]) for _ in range(N)]

    # 강한 시그널 + 상호작용 (목표 ROC AUC ~0.80)
    logit = np.full(N, -5.2)
    logit += np.array([1.2 if r == "서울" else 0.0 for r in region])
    logit += np.array([1.5 if o == "Y" else 0.0 for o in overtime])
    logit += np.array([1.4 if i == "N" else 0.0 for i in incentive])
    logit += np.array([1.0 if c == "N" else 0.0 for c in core])
    logit += np.array([{"SS": -1.5, "EE": -1.0, "AA": 0.0,
                        "BB": 0.8, "CC": 1.8, "UN": 0.5}[g] for g in grade])
    logit += np.where(work_years < 3, 1.3, 0.0)
    logit += np.where(work_years > 15, -1.2, 0.0)
    logit += np.where(promo_years > 6, 1.0, 0.0)
    logit += np.where(pre_jobs >= 3, 1.1, 0.0)
    logit += np.where(base_salary < 4000, 1.0, 0.0)
    logit += np.array([0.8 if t == "계약직" else 0.0 for t in hire_type])
    logit += np.array([0.4 if m == "미혼" else 0.0 for m in marriage])
    logit += np.where(age < 30, 0.4, 0.0)
    logit += np.array([0.7 if (r == "서울" and o == "Y") else 0.0
                       for r, o in zip(region, overtime)])
    logit += rng.normal(0, 0.15, size=N)

    prob_quit = 1 / (1 + np.exp(-logit))
    status = (rng.random(N) < prob_quit).astype(int)

    재직_str = ["퇴직" if s == 1 else "재직" for s in status]

    # 퇴직일/사유/이직처
    today = datetime(2025, 12, 31)
    resign_date, resign_reason, resign_dest = [], [], []
    for s in status:
        if s == 1:
            d = today - timedelta(days=int(rng.integers(30, 900)))
            resign_date.append(d.strftime("%Y-%m-%d"))
            resign_reason.append(pick(RESIGN_REASON,
                                      p=[0.28, 0.10, 0.05, 0.18, 0.13, 0.12, 0.10, 0.04]))
            resign_dest.append(pick(DEST))
        else:
            resign_date.append("")
            resign_reason.append("")
            resign_dest.append("")

    df = pd.DataFrame({
        "사원번호": emp_ids,
        "재직": 재직_str,
        "퇴직일": resign_date,
        "이름": names,
        "나이": age,
        "성별": gender,
        "직위": position,
        "직책": title,
        "직무": job,
        "승진후경과연수": promo_years,
        "소속조직": org,
        "팀": team,
        "근무연수": work_years,
        "채용유형": hire_type,
        "근무지역": region,
        "국가핵심기술관리": nct,
        "최종교육수준": edu,
        "전공": major,
        "보유역량": competency,
        "직무역할": role,
        "결혼": marriage,
        "기본급": base_salary,
        "경력입사여부": pre_hired,
        "경력이직횟수": pre_jobs,
        "연장근무": overtime,
        "재택근무": wfh,
        "평가등급": grade,
        "핵심인재": core,
        "인센티브": incentive,
        "퇴직사유": resign_reason,
        "퇴직후이직처": resign_dest,
    })

    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    try:
        df.to_excel(out_xlsx, index=False)
        xlsx_ok = True
    except Exception as e:
        xlsx_ok = False
        print(f"  (xlsx 스킵: {e})")

    n_quit = (df["재직"] == "퇴직").sum()
    print(f"  CSV : {out_csv}")
    if xlsx_ok:
        print(f"  XLSX: {out_xlsx}")
    print(f"  인원 {len(df):,}명  |  재직 {len(df)-n_quit:,}명  /  퇴직 {n_quit:,}명 ({n_quit/len(df)*100:.1f}%)  |  핵심인재 {(df['핵심인재']=='Y').sum():,}명")


if __name__ == "__main__":
    base = "/Users/minju/Desktop/phyton"
    print("[A] 시드 42 / 사원번호 10001~")
    generate(
        N=1500,
        seed=42,
        start_emp_id=10001,
        out_csv=f"{base}/HR_퇴직예측_샘플데이터_1500명_A.csv",
        out_xlsx=f"{base}/HR_퇴직예측_샘플데이터_1500명_A.xlsx",
    )
    print("\n[B] 시드 2026 / 사원번호 20001~")
    generate(
        N=1500,
        seed=2026,
        start_emp_id=20001,
        out_csv=f"{base}/HR_퇴직예측_샘플데이터_1500명_B.csv",
        out_xlsx=f"{base}/HR_퇴직예측_샘플데이터_1500명_B.xlsx",
    )
