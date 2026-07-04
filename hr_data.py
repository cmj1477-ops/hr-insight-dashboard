# =========================
# 데이터 로딩 / 전처리 / 인코딩
# =========================
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# 결측/미기재를 의미하는 토큰 (통계 집계 시 제외 대상) — 전 페이지 공통
MISSING_TOKENS = {'', 'nan', 'NaN', 'NAN', 'None', 'NONE', 'NULL', 'null', '미기재', '미입력', '-'}

# 핵심인재 여부 판정 시 '핵심인재 아님'으로 간주하는 값 (대문자 비교)
CORE_NEG_SET = {'미입력', '미기재', 'NAN', 'NONE', 'NULL', '', 'N', 'NO', 'FALSE', '0', '-'}

# 업로드 컬럼 표준화(과거 명칭 → 현재 표준명)
# - 타깃: '재직' → '상태'(재직=0, 퇴직=1)
# - 과거 이직 지표: '이직' → '경력입사여부', '이직횟수' → '입사전이직횟수'
CANON_MAP = {
    "재직": "상태",
    "이직": "경력입사여부",
    "이직횟수": "입사전이직횟수",
    "경력이직횟수": "입사전이직횟수",
    # 최근 스키마 변경
    "직급": "직책"
}

# 상태(타깃) 변환 맵
TARGET_MAP = {
    'Y': 1, 'YES': 1, 'Yes': 1, 'yes': 1, '퇴직': 1, 1: 1, '1': 1, True: 1,
    'N': 0, 'NO': 0, 'No': 0, 'no': 0, '재직': 0, 0: 0, '0': 0, False: 0
}

# 식별자/보고용 드롭
DROP_COLS_BASE = ['사원번호', '이름']

# 예측에 쓰면 안 되는(누수/사후 정보/날짜/자유텍스트) 컬럼
LEAKAGE_DROP = ['퇴직일', '퇴직사유', '퇴직후이직처']

# ---------- 스키마(최신) ----------
NUM_COLS_HINT = [
    '나이', '승진후경과연수', '근무연수', '기본급', '입사전이직횟수', '보유역량'
]
CAT_COLS = [
    '성별', '직위', '직무', '직책', '소속조직', '팀', '채용유형', '근무지역', '국가핵심기술관리',
    '최종교육수준', '전공', '직무역할', '결혼',
    '경력입사여부', '연장근무', '재택근무', '평가등급', '핵심인재', '인센티브'
]


def load_any(uploaded_file):
    if uploaded_file is None:
        return None, "파일을 업로드하세요."
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding="cp949")
        elif name.endswith(".xlsx"):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif name.endswith(".xls"):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, engine="xlrd")
        else:
            return None, "지원하지 않는 포맷입니다. csv/xlsx만 업로드해주세요."
        return df, None
    except Exception as e:
        return None, f"파일 로딩 중 오류: {str(e)}"


def sanitize_df(df: pd.DataFrame, fill_cat="미입력", fill_num=0):
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace(['nan', 'NaN', 'NULL', 'None'], np.nan)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(fill_num)
        else:
            df[col] = df[col].fillna(fill_cat).astype(str).str.strip()
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 이름 매핑
    for old, new in CANON_MAP.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    # 상태(타깃) 정규화
    if '상태' in df.columns:
        df['상태'] = df['상태'].map(lambda v: TARGET_MAP.get(v, TARGET_MAP.get(str(v), np.nan)))
        unrecognized = df['상태'].isna().sum()
        if unrecognized > 0:
            st.warning(f"'상태' 컬럼에서 인식되지 않은 값 {unrecognized}개가 있습니다. 해당 행은 재직(0)으로 처리됩니다. 데이터를 확인하세요.")
        df['상태'] = df['상태'].fillna(0).astype(int)
    return df


def clean_text_series(s: pd.Series) -> pd.Series:
    """문자열 시리즈에서 결측 토큰(미입력/미기재/nan 등)을 제거하고 반환"""
    s = s.astype(str).str.strip()
    return s[~s.isin(MISSING_TOKENS)]


def get_core_mask(df: pd.DataFrame, label_encoders: dict, core_col: str = '핵심인재') -> pd.Series:
    """핵심인재 여부 마스크 (부정/빈 값 키워드 외에는 모두 핵심인재로 간주)"""
    if core_col not in df.columns:
        return pd.Series(False, index=df.index)
    if core_col in label_encoders:
        classes = list(label_encoders[core_col].classes_)
        pos_idx = [i for i, v in enumerate(classes) if str(v).strip().upper() not in CORE_NEG_SET]
        return df[core_col].isin(pos_idx)
    return ~df[core_col].astype(str).str.strip().str.upper().isin(CORE_NEG_SET)


def _build_validation(df_raw: pd.DataFrame) -> dict:
    """업로드 직후 데이터 품질 리포트용 요약 정보 생성"""
    expected = set(NUM_COLS_HINT + CAT_COLS + ['상태', '사원번호', '이름'])
    canon_cols = set()
    for c in df_raw.columns:
        canon_cols.add(CANON_MAP.get(c, c))
    missing_rate = (df_raw.isna() | df_raw.astype(str).apply(lambda s: s.str.strip().isin(MISSING_TOKENS))).mean()
    missing_rate = (missing_rate * 100).round(1)
    missing_top = missing_rate[missing_rate > 0].sort_values(ascending=False).head(10)
    return {
        'n_rows': int(len(df_raw)),
        'n_cols': int(df_raw.shape[1]),
        'matched_cols': sorted(expected & canon_cols),
        'missing_expected': sorted(expected - canon_cols - {'사원번호', '이름'}),
        'extra_cols': sorted(canon_cols - expected - set(LEAKAGE_DROP)),
        'missing_top': missing_top.to_dict(),
    }


@st.cache_data(show_spinner=True)
def load_and_preprocess(uploaded_file):
    """반환: (df, X, y, label_encoders, validation, err)"""
    df_raw, err = load_any(uploaded_file)
    if err:
        return None, None, None, None, None, err

    validation = _build_validation(df_raw)

    df = sanitize_df(df_raw.copy())
    df = normalize_columns(df)

    if '상태' not in df.columns:
        return None, None, None, None, None, "필수 컬럼 '상태'(0=재직,1=퇴직)가 없습니다. ('재직' 컬럼을 주면 자동 변환됩니다)"

    if '퇴직일' in df.columns:
        df['퇴직일'] = pd.to_datetime(df['퇴직일'], errors='coerce')

    label_encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    non_feature_cols = DROP_COLS_BASE + ['상태'] + LEAKAGE_DROP

    # 스키마 외 텍스트 컬럼도 df 자체에 인코딩해 저장
    # (X와 df가 항상 같은 값을 갖도록 하여 df[X.columns] 예측 시 dtype 오류 방지)
    for c in df.columns:
        if c in non_feature_cols or c in label_encoders:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.strip()
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c])
            label_encoders[c] = le

    X = df.drop(columns=[c for c in non_feature_cols if c in df.columns], errors='ignore')
    y = df['상태']

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

    y = y.fillna(0).astype(int)
    return df, X, y, label_encoders, validation, None
