import streamlit as st
import pandas as pd
import numpy as np

# NumPy 2.0 compatibility shim
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score
from scipy.stats import chi2_contingency

# ★ XGBoost
import xgboost as xgb
import shap

# =========================
# 페이지/스타일
# =========================
# =========================
# 페이지/스타일 (Clean & Minimal UI)
# =========================
st.set_page_config(page_title="HR Insight Dashboard", layout="wide")
pio.templates.default = "plotly_white"

# 폰트 및 컬러 팔레트 정의
PRIMARY_COLOR = "#2563EB"
BG_COLOR = "#F8F9FA"       # Standard Light Gray
CARD_BG = "#FFFFFF"        # Solid White
TEXT_COLOR = "#333333"

# 차트 컬러 팔레트 (Corporate Custom: Purple & Cyan)
COLORS = {
    "primary": "#48C0D8",      # Corporate Cyan (Bar Charts)
    "secondary": "#5548C7",    # Corporate Purple (Line/Trend)
    "success": "#48C0D8",      # Cyan
    "danger": "#5548C7",       # Purple (Emphasis/Warning)
    "warning": "#F59E0B",      # Amber
    "info": "#48C0D8",         # Cyan
    "light": "#F8FAFC",        # Light
    "dark": "#334155",         # Dark
    "sequence": ["#48C0D8", "#5548C7", "#7DD3FC", "#A5B4FC", "#C4B5FD"] # Cyan & Purple Mix
}

def set_font(fig):
    layout_updates = {
        'font': dict(family="Pretendard, -apple-system, system-ui, sans-serif", size=14, color=TEXT_COLOR),
        'paper_bgcolor': "rgba(0,0,0,0)",
        'plot_bgcolor': "rgba(0,0,0,0)",
        'margin': dict(t=40, b=20, l=20, r=20)
    }
    
    # title이 있는 경우에만 title 관련 폰트 설정 추가
    if fig.layout.title and fig.layout.title.text:
        layout_updates['title_font_size'] = 18
        layout_updates['title_font_family'] = "Pretendard, sans-serif"
        layout_updates['title_font_color'] = "#111827"
    
    fig.update_layout(**layout_updates)
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#E5E7EB")
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=False)
    return fig

# Clean Minimal CSS Injection
st.markdown(f"""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    /* 전체 배경 및 폰트 */
    .stApp {{
        background-color: {BG_COLOR};
        font-family: 'Pretendard', sans-serif;
        color: {TEXT_COLOR};
    }}

    /* 헤더 숨김 */
    header {{visibility: hidden;}}
    
    /* 메인 컨테이너 */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }}

    /* 카드 스타일 (Clean Flat) */
    div[data-testid="stMetric"], div.stDataFrame, .plotly-graph-div {{
        background: {CARD_BG};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); /* 아주 가벼운 그림자 */
        border: 1px solid #E5E7EB;
    }}
    
    /* 호버 효과 제거 또는 아주 약하게 */
    div[data-testid="stMetric"]:hover, .plotly-graph-div:hover {{
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }}

    /* 메트릭 텍스트 스타일 */
    div[data-testid="stMetricLabel"] {{
        font-size: 0.9rem;
        color: #6B7280;
        font-weight: 500;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
    }}

    /* 데이터프레임 헤더 스타일 */
    div[data-testid="stDataFrame"] table th {{
        background-color: #F3F4F6 !important;
        color: #374151 !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #E5E7EB !important;
        text-align: center !important;
    }}
    div[data-testid="stDataFrame"] table td {{
        text-align: center !important;
        color: #4B5563 !important;
        font-size: 0.95rem;
        border-bottom: 1px solid #F3F4F6 !important;
    }}

    /* 사이드바 스타일 */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #48C0D8 0%, #3BADC7 100%);
        border-right: none;
        padding-top: 0;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 0;
    }}
    /* 사이드바 내부 텍스트 색상 */
    section[data-testid="stSidebar"] * {{
        color: rgba(255, 255, 255, 0.85) !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
    }}
    /* 사이드바 메뉴 버튼 */
    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        text-align: left;
        padding: 14px 18px !important;
        border-radius: 8px;
        border: none;
        font-size: 15px;
        font-weight: 500;
        font-family: 'Pretendard', sans-serif;
        background: transparent !important;
        color: rgba(255, 255, 255, 0.85) !important;
        transition: all 0.15s ease;
        margin-bottom: 2px;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255, 255, 255, 0.15) !important;
        color: #FFFFFF !important;
    }}
    section[data-testid="stSidebar"] .stButton > button:focus {{
        box-shadow: none !important;
    }}
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1) !important;
    }}
    /* 사이드바 파일 업로더 */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px;
        border: 1px dashed rgba(255, 255, 255, 0.3);
    }}
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button {{
        background: rgba(255, 255, 255, 0.9) !important;
        border: none !important;
        color: #2A9BB0 !important;
        font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] button:hover {{
        background: rgba(255, 255, 255, 1) !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] small,
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] span,
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] p,
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] div {{
        color: #6B7280 !important;
    }}
    /* 사이드바 구분선 */
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
    }}

    /* 버튼 스타일 */
    button {{
        border-radius: 6px !important;
        box-shadow: none !important;
    }}

    /* 타이틀 스타일 */
    h1, h2, h3 {{
        font-family: 'Pretendard', sans-serif;
        font-weight: 700;
        color: #111827;
    }}
    
    /* 구분선 스타일 */
    hr {{
        margin: 2rem 0;
        border-color: #E5E7EB;
    }}
    </style>
    """, unsafe_allow_html=True)

def reason_to_tags(reason: str) -> list:
    """예측사유 문자열을 태그 딕셔너리 리스트로 변환"""
    tags = []
    if "서울" in reason:
        tags.append({"label": "서울근무", "color": "red"})
    if "인센티브" in reason and "'Y'" in reason:
        tags.append({"label": "인센티브Y", "color": "amber"})
    if "파트장" in reason and "직책" in reason:
        tags.append({"label": "파트장직책", "color": "blue"})
    if "퇴직률" in reason:
        import re
        matches = re.findall(r"(\S+)\s+'[^']+'\s+퇴직률", reason)
        for m in matches:
            if m not in ['직책']:
                tags.append({"label": m, "color": "blue"})
    if "↓" in reason:
        tags.append({"label": "평균이하", "color": "amber"})
    if "↑" in reason:
        tags.append({"label": "평균이상", "color": "red"})
    if "복합" in reason:
        tags.append({"label": "복합요인", "color": "gray"})
    return tags if tags else [{"label": "기타", "color": "gray"}]


def build_core_talent_html(df) -> str:
    """핵심인재 전체 리스트를 접이식 HTML 테이블로 생성"""
    import streamlit.components.v1 as components

    tag_colors = {
        "red":   ("rgba(162,28,28,.12)", "#991f1f"),
        "amber": ("rgba(146,88,0,.12)",  "#7a4e00"),
        "blue":  ("rgba(20,80,150,.12)", "#0d4a8a"),
        "gray":  ("rgba(100,100,100,.1)","#555555"),
    }

    # 컬럼 구성 확인
    has_incentive = '인센티브' in df.columns
    has_grade = '평가등급' in df.columns
    has_reason = '예측사유' in df.columns

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        risk_str = str(row.get("예측퇴직위험", "0%")).replace("%", "")
        try:
            risk = float(risk_str)
        except Exception:
            risk = 0.0

        if risk >= 7:
            risk_color = "#c0392b"
        elif risk >= 4:
            risk_color = "#b87a00"
        else:
            risk_color = "#1a7a3c"
        risk_text = f"{risk}%"

        grade = str(row.get("평가등급", ""))
        if grade in ["EE", "AA", "SS"]:
            grade_bg, grade_color = "rgba(20,130,60,.13)", "#0e6b30"
        else:
            grade_bg, grade_color = "rgba(180,120,0,.13)", "#7a5200"

        tags_html = ""
        reason_text = ""
        if has_reason:
            reason_text = str(row.get("예측사유", ""))
            tags = reason_to_tags(reason_text)
            for t in tags:
                bg, fc = tag_colors.get(t["color"], tag_colors["gray"])
                tags_html += (
                    f'<span style="display:inline-flex;align-items:center;padding:2px 8px;'
                    f'border-radius:20px;font-size:11px;font-weight:500;margin:1px 2px;'
                    f'background:{bg};color:{fc}">{t["label"]}</span>'
                )

        incentive_td = f'<td style="padding:9px 10px;text-align:center">{row.get("인센티브","")}</td>' if has_incentive else ""
        grade_td = f"""<td style="padding:9px 10px">
            <span style="padding:2px 7px;border-radius:4px;font-size:11px;font-weight:500;
              background:{grade_bg};color:{grade_color}">{grade}</span>
          </td>""" if has_grade else ""

        rows_html += f"""
        <tr class="main-row" onclick="toggle({i})" style="cursor:pointer">
          <td>
            <span id="chev-{i}" style="font-size:9px;color:#aaa;display:inline-block;transition:transform .2s">▶</span>
          </td>
          <td style="color:#888;white-space:nowrap">{row.get("사원번호","")}</td>
          <td><strong style="color:#334155">{row.get("이름","")}</strong></td>
          <td style="white-space:nowrap">{row.get("소속조직","")}</td>
          <td style="white-space:nowrap">{row.get("직책","")}</td>
          {grade_td}
          {incentive_td}
          <td style="font-weight:600;color:{risk_color};white-space:nowrap">{risk_text}</td>
          <td style="text-align:left">{tags_html}</td>
        </tr>
        <tr id="detail-{i}" class="detail-row" style="display:none">
          <td colspan="9" style="padding:0">
            <div style="padding:10px 12px 10px 44px;font-size:0.85rem;color:#64748B;border-bottom:1px solid #E2E8F0;text-align:left">
              <span style="font-size:0.75rem;font-weight:700;color:#94A3B8;letter-spacing:.04em;margin-right:6px">예측사유</span>{reason_text}
            </div>
          </td>
        </tr>
        """

    incentive_th = '<th>인센티브</th>' if has_incentive else ""
    grade_th = '<th>평가등급</th>' if has_grade else ""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{ font-family: 'Pretendard', 'Noto Sans KR', sans-serif; font-size: 0.9rem; color: #475569; }}
      .table-wrap {{
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        background: #FFFFFF;
      }}
      table {{ width: 100%; border-collapse: collapse; }}
      th {{
        background-color: #F1F5F9;
        color: #334155;
        font-weight: 700;
        text-align: center;
        padding: 10px 12px;
        border-bottom: 2px solid #CBD5E1;
        font-size: 0.95rem;
        white-space: nowrap;
      }}
      td {{
        text-align: center;
        padding: 8px 12px;
        border-bottom: 1px solid #E2E8F0;
        color: #475569;
        font-size: 0.9rem;
        vertical-align: middle;
      }}
      /* Zebra Striping */
      .main-row:nth-child(4n+1) td {{ background-color: #F8FAFC; }}
      /* Hover Effect */
      .main-row:hover td {{
        background-color: #E0F2FE !important;
        color: #0284C7;
        transition: background-color 0.2s ease;
      }}
      /* Detail row */
      .detail-row td {{ background-color: #F8FAFC; }}
    </style>
    </head>
    <body>
    <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th style="width:22px"></th>
          <th>사원번호</th><th>이름</th><th>소속조직</th><th>직책</th>
          {grade_th}
          {incentive_th}
          <th>예측퇴직위험</th><th>예측사유</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
    </div>
    <script>
      function toggle(idx) {{
        var detail = document.getElementById('detail-' + idx);
        var chev   = document.getElementById('chev-'   + idx);
        var isOpen = detail.style.display !== 'none';
        detail.style.display = isOpen ? 'none' : 'table-row';
        chev.style.transform  = isOpen ? 'rotate(0deg)' : 'rotate(90deg)';
      }}
    </script>
    </body>
    </html>
    """
    return html


def show_table_centered(df):
    """
    Streamlit dataframe의 정렬 이슈를 해결하기 위해 HTML로 직접 렌더링합니다.
    """
    try:
        df_disp = df.fillna('-')
        
        # HTML로 변환
        html_table = df_disp.to_html(index=False, escape=False)
        
        # 커스텀 CSS 적용 (Option 2: Striped Style - Compact)
        # Markdown에서 들여쓰기가 있으면 코드 블록으로 인식될 수 있으므로 들여쓰기를 제거합니다.
        st.markdown(f"""
<style>
.custom-table-container {{
    font-family: 'Pretendard', sans-serif;
    margin-bottom: 1.5rem;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #E2E8F0; /* Light Border */
}}
.custom-table-container table {{
    width: 100%;
    border-collapse: collapse;
    background-color: #FFFFFF;
}}
.custom-table-container th {{
    background-color: #F1F5F9; /* Light Slate Header */
    color: #334155;
    font-weight: 700;
    text-align: center !important;
    padding: 10px 12px; /* Reduced padding */
    border-bottom: 2px solid #CBD5E1;
    font-size: 0.95rem;
}}
.custom-table-container td {{
    text-align: center !important;
    padding: 8px 12px; /* Reduced padding */
    border-bottom: 1px solid #E2E8F0;
    color: #475569;
    font-size: 0.9rem;
    vertical-align: middle;
}}
/* Zebra Striping */
.custom-table-container tr:nth-child(even) {{
    background-color: #F8FAFC; /* Very Light Slate */
}}
.custom-table-container tr:last-child td {{
    border-bottom: none;
}}
/* Hover Effect */
.custom-table-container tr:hover td {{
    background-color: #E0F2FE; /* Light Sky Blue */
    color: #0284C7;
    transition: background-color 0.2s ease;
}}
</style>
<div class="custom-table-container">
    {html_table}
</div>
""", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"테이블 렌더링 오류: {e}")
        st.dataframe(df)

# =========================
# 업로드
# =========================
st.sidebar.markdown("""
<div style="font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.7); margin-bottom: 8px; letter-spacing: 1px; text-transform: uppercase; text-align: center;">
    데이터 업로드
</div>
""", unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("회사 데이터 업로드 (csv/xlsx)", type=["csv", "xlsx"], label_visibility="collapsed")

# 임계값(정책상 고정)
th = 0.50

# =========================
# 유틸: 파일 로드/정리
# =========================
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

# 업로드 컬럼 표준화(과거 명칭 → 현재 표준명)
# - 타깃: '재직' → '상태'(재직=0, 퇴직=1)
# - 과거 이직 지표: '이직' → '경력입사여부', '이직횟수' → '입사전이직횟수'
CANON_MAP = {
    "재직": "상태",
    "이직": "경력입사여부",
    "이직횟수": "입사전이직횟수",
    # 최근 스키마 변경
    "직급": "직책"
}

# 상태(타깃) 변환 맵
TARGET_MAP = {
    'Y':1, 'YES':1, 'Yes':1, 'yes':1, '퇴직':1, 1:1, '1':1, True:1,
    'N':0, 'NO':0,  'No':0,  'no':0,  '재직':0, 0:0, '0':0, False:0
}

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
            import streamlit as _st
            _st.warning(f"'상태' 컬럼에서 인식되지 않은 값 {unrecognized}개가 있습니다. 해당 행은 재직(0)으로 처리됩니다. 데이터를 확인하세요.")
        df['상태'] = df['상태'].fillna(0).astype(int)
    return df

# 식별자/보고용 드롭
DROP_COLS_BASE = ['사원번호','이름']

# 예측에 쓰면 안 되는(누수/사후 정보/날짜/자유텍스트) 컬럼
LEAKAGE_DROP = ['퇴직일', '퇴직사유', '퇴직후이직처']

# ---------- 스키마(최신) ----------
NUM_COLS_HINT = [
    '나이','승진후경과연수','근무연수','기본급','입사전이직횟수','보유역량'
]
CAT_COLS = [
    '성별','직위','직무','직책','소속조직','팀','채용유형','근무지역','국가핵심기술관리',
    '최종교육수준','전공','직무역할','결혼',
    '경력입사여부','연장근무','재택근무','평가등급','핵심인재','인센티브'
]

def get_label(val, col, encoders):
    if col in encoders:
        try:
            return encoders[col].inverse_transform([int(val)])[0]
        except Exception:
            return val
    return val

def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    if ct.empty or ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(ct)[0]
    n = ct.values.sum()
    r, k = ct.shape
    denom = min(k - 1, r - 1)
    if denom <= 0 or n == 0:
        return 0.0
    return np.sqrt((chi2 / n) / denom)

def bucketize_numeric(series: pd.Series, bins="quartile"):
    s = pd.to_numeric(series, errors='coerce')
    if s.nunique() < 4:
        return pd.cut(s, bins=3, include_lowest=True)
    if bins == "decile":
        try:
            return pd.qcut(s, 10, duplicates='drop')
        except Exception:
            return pd.qcut(s, 4, duplicates='drop')
    else:
        return pd.qcut(s, 4, duplicates='drop')

def _fmt_range(v1, v2, unit=""):
    v1 = max(0, float(v1)); v2 = max(0, float(v2))
    v1 = round(v1); v2 = round(v2)
    return f"{v1:,.0f}~{v2:,.0f}{unit}"

def humanize_interval_label(var: str, interval) -> str:
    left = interval.left if hasattr(interval, 'left') else None
    right = interval.right if hasattr(interval, 'right') else None
    if left is None or right is None:
        return str(interval)
    salary_like = ['기본급','연봉','급여','월급']
    years_like  = ['근무연수','승진후경과연수']
    age_like    = ['나이','연령']
    if any(k in var for k in salary_like):
        return _fmt_range(left, right, unit="만원")
    elif var in years_like:
        return _fmt_range(left, right, unit="년")
    elif var in age_like:
        return _fmt_range(left, right, unit="세")
    else:
        return _fmt_range(left, right, unit="")

def format_explain_headline(var_name, bucket_label, rate, overall, action):
    judge = "대비 높음" if rate >= overall else "대비 낮음"
    return f"{var_name} {bucket_label} 퇴직률 {rate:.1f}% — 평균 {overall:.1f}% {judge}. 필요: {action}"

def format_explain_compact(var_name, bucket_label, rate, overall, action):
    line1 = f"현상: {var_name} {bucket_label} 퇴직률 {rate:.1f}% (평균 {overall:.1f}%)"
    line2 = f"판단: {'높음' if rate >= overall else '낮음'}  ·  필요: {action}"
    return line1 + "\n" + line2

def render_explanation(var_name, bucket_label, rate, overall, n=None, share=None, delta=None, action="리텐션 정책 점검", explain_mode="헤드라인"):
    if explain_mode == "헤드라인":
        st.markdown(format_explain_headline(var_name, bucket_label, rate, overall, action))
    elif explain_mode == "콤팩트":
        st.markdown(format_explain_compact(var_name, bucket_label, rate, overall, action))
    else:
        st.markdown(format_explain_headline(var_name, bucket_label, rate, overall, action))
        with st.expander("근거 보기"):
            if n is not None and share is not None:
                st.write(f"- 그룹 인원수: {n}명 ({share:.1f}%)")
            if delta is not None:
                st.write(f"- 전기 대비 퇴직률 변화: {delta:+.1f}%p")

def add_pdf_button():
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        var parentDoc = window.parent.document;

        // 이미 버튼이 있으면 중복 생성 방지
        if (parentDoc.getElementById('pdf-download-btn')) return;

        // Font Awesome 로드
        if (!parentDoc.querySelector('link[href*="font-awesome"]')) {
            var fa = parentDoc.createElement('link');
            fa.rel = 'stylesheet';
            fa.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css';
            parentDoc.head.appendChild(fa);
        }

        // 프린트용 CSS + 버튼 스타일 주입
        var style = parentDoc.createElement('style');
        style.textContent = `
            #pdf-download-btn {
                position: fixed;
                top: 60px;
                right: 20px;
                z-index: 999999;
                background-color: #48C0D8;
                color: white;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-family: 'Pretendard', sans-serif;
            }
            #pdf-download-btn:hover {
                background-color: #3BADC7;
            }

            @media print {
                /* 사이드바 숨김 */
                section[data-testid="stSidebar"],
                [data-testid="stSidebarNav"],
                [data-testid="collapsedControl"] {
                    display: none !important;
                }
                /* Streamlit 헤더/푸터 숨김 */
                header, footer,
                .stDeployButton,
                [data-testid="stToolbar"],
                [data-testid="stDecoration"],
                [data-testid="stStatusWidget"] {
                    display: none !important;
                }
                /* PDF 버튼 자체 숨김 */
                #pdf-download-btn {
                    display: none !important;
                }
                /* iframe(components) 영역 숨김 - 빈 공간 제거 */
                iframe[title="streamlit_components.v1.components.html"] {
                    display: none !important;
                }
                /* 본문 영역을 전체 너비로 */
                section[data-testid="stMain"],
                .main,
                [data-testid="stAppViewContainer"] {
                    margin-left: 0 !important;
                    padding-left: 0 !important;
                    width: 100% !important;
                    max-width: 100% !important;
                }
                .block-container {
                    max-width: 100% !important;
                    padding: 1rem !important;
                }
                /* 배경색 유지 */
                .stApp {
                    background-color: white !important;
                }
                /* 페이지 설정 */
                @page {
                    size: A4 landscape;
                    margin: 10mm;
                }
            }
        `;
        parentDoc.head.appendChild(style);

        // 버튼 생성 및 부모 문서에 삽입
        var btn = parentDoc.createElement('button');
        btn.id = 'pdf-download-btn';
        btn.innerHTML = '<i class="fa fa-print"></i> PDF 저장';
        btn.onclick = function() {
            window.parent.print();
        };
        parentDoc.body.appendChild(btn);
    })();
    </script>
    """, height=0)

# =========================
# 데이터 로딩/전처리/인코딩
# =========================
@st.cache_data(show_spinner=True)
def load_and_preprocess(uploaded_file):
    df_raw, err = load_any(uploaded_file)
    if err:
        return None, None, None, None, err

    df = sanitize_df(df_raw.copy())
    df = normalize_columns(df)

    if '상태' not in df.columns:
        return None, None, None, None, "필수 컬럼 '상태'(0=재직,1=퇴직)가 없습니다. ('재직' 컬럼을 주면 자동 변환됩니다)"

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
    X = df.drop(columns=[c for c in non_feature_cols if c in df.columns], errors='ignore')
    y = df['상태']

    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
        else:
            X[c] = X[c].astype(str).str.strip()
            tmp_le = LabelEncoder()
            X[c] = tmp_le.fit_transform(X[c])

    y = y.fillna(0).astype(int)
    return df, X, y, label_encoders, None

# =========================
# 모델 학습(XGBoost) + 중요도
# =========================
@st.cache_data(show_spinner=True)
def train_model_with_calibration(X, y):
    if y.nunique() < 2:
        return None, {"error": "타깃 클래스가 한 종류만 있습니다. 재직(0)/퇴직(1)이 모두 포함되도록 데이터를 확인하세요."}, None

    # ⚖️ 클래스 불균형 보정용 가중치 계산
    pos_count = int(y.sum())                 # 퇴직(1)
    neg_count = int(len(y) - pos_count)      # 재직(0)

    if pos_count == 0:
        scale_pos_weight = 1.0
    else:
        scale_pos_weight = neg_count / pos_count

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost 모델 정의
    model = xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=False,

        # 🔥 핵심: 퇴직(1) 클래스에 가중치 적용
        scale_pos_weight=scale_pos_weight
    )

    # 학습
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 성능 지표 계산
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "cv_mean": {"accuracy": np.nan, "f1": np.nan, "roc_auc": np.nan},
        "y_test": y_test,
        "y_proba_test": y_proba,
    }

    # 피쳐 중요도 계산
    try:
        booster = model.get_booster()

        def _score_to_series(importance_type):
            score = booster.get_score(importance_type=importance_type)
            cols = list(X.columns)

            if set(score.keys()) & set(cols):
                s = pd.Series(score)
                return s.reindex(cols).fillna(0.0)

            mapping = {f"f{i}": col for i, col in enumerate(cols)}
            s = pd.Series({mapping[k]: v for k, v in score.items() if k in mapping})
            return s.reindex(cols).fillna(0.0)

        imp = _score_to_series("gain")
        if float(imp.sum()) == 0.0:
            imp = _score_to_series("total_gain")
        if float(imp.sum()) == 0.0:
            imp = _score_to_series("weight")
        if float(imp.sum()) == 0.0:
            imp = pd.Series(model.feature_importances_, index=X.columns).fillna(0.0)

        feature_importance = imp.sort_values(ascending=False)

    except Exception:
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).fillna(0.0).sort_values(ascending=False)

    return model, metrics, feature_importance

# =========================
# 앱 시작
# =========================
if uploaded is None:
    st.info("사이드바에서 CSV 또는 Excel 파일을 업로드해주세요.")
    st.stop()

df, X, y, label_encoders, load_err = load_and_preprocess(uploaded)
if load_err:
    st.error(load_err); st.stop()
if df is None:
    st.warning("데이터를 불러올 수 없습니다. 파일 형식을 확인해주세요."); st.stop()

model, metrics, feature_importance = train_model_with_calibration(X, y)
if isinstance(metrics, dict) and "error" in metrics:
    st.error(metrics["error"]); st.stop()

# 상위 중요 변수 6개 사용
top_features = feature_importance.head(6).index.tolist() if feature_importance is not None and len(feature_importance) > 0 else []

# 사이드바 메뉴
st.sidebar.markdown("""
<div style="padding: 30px 16px 20px 16px; text-align: center;">
    <div style="
        display: inline-block;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 14px;
        padding: 22px 28px;
        width: 100%;
        box-sizing: border-box;
    ">
        <div style="font-size: 26px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px; line-height: 1.3;">
            핵심인재
        </div>
        <div style="font-size: 26px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px; line-height: 1.3;">
            퇴직예측모델
        </div>
        <div style="margin-top: 12px; height: 3px; width: 40px; background: rgba(255,255,255,0.4); border-radius: 2px; margin-left: auto; margin-right: auto;"></div>
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
menu_items = ["전체 현황", "핵심인재 현황", "개인별 현황"]
if "menu" not in st.session_state:
    st.session_state["menu"] = "전체 현황"
for item in menu_items:
    is_active = st.session_state["menu"] == item
    if st.sidebar.button(item, key=f"menu_{item}", type="primary" if is_active else "secondary", use_container_width=True):
        st.session_state["menu"] = item
        st.rerun()
menu = st.session_state["menu"]
st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="margin-top: 200px; text-align: center;">
    <div style="font-size: 16px; font-weight: 600; color: rgba(255, 255, 255, 0.6); letter-spacing: 1px;">
        CTO인사지원팀
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# 1) 전체 현황
# =========================
if menu == "전체 현황":
    add_pdf_button()
    st.title("전체 현황 대시보드")
    # 전체 예측 확률 — 한 번만 계산 후 재사용
    _all_proba = model.predict_proba(df[X.columns])[:, 1]
    st.markdown("""
    <div style="background-color: #F8FAFC; padding: 15px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 20px;">
        <p style="margin: 0; font-size: 14px; color: #64748B; line-height: 1.6;">
        본 대시보드는 XGBoost 기반 분류모델을 활용해 재직자의 향후 퇴직 위험을 예측합니다.<br>
        예측 결과는 통계적 패턴 기반으로 산출되며, 외부 환경이나 개인적 사유 등 모델이 반영할 수 없는 요인은 포함되지 않습니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    total_rate = float(df['상태'].mean() * 100)

    core_rate = np.nan
    if '핵심인재' in df.columns:
        try:
            if '핵심인재' in label_encoders:
                classes = list(label_encoders['핵심인재'].classes_)
                # 부정/빈 값 키워드 (이 외에는 모두 핵심인재로 간주)
                neg_set = {'미입력', 'NAN', 'NONE', '', 'N', 'NO', 'FALSE', '0', 'nan'}
                pos_idx = [i for i, v in enumerate(classes) if str(v).strip().upper() not in neg_set]
                core_mask = df['핵심인재'].isin(pos_idx)
            else:
                neg_set = {'미입력', 'NAN', 'NONE', '', 'N', 'NO', 'FALSE', '0', 'nan'}
                core_mask = ~df['핵심인재'].astype(str).str.strip().str.upper().isin(neg_set)
            core_subset = df.loc[core_mask]
            core_rate = float(core_subset['상태'].mean() * 100) if len(core_subset) > 0 else np.nan
        except Exception:
            core_rate = np.nan

    if '퇴직일' in df.columns:
        try:
            if not np.issubdtype(df['퇴직일'].dtype, np.datetime64):
                df['퇴직일'] = pd.to_datetime(df['퇴직일'], errors='coerce')
            month_counts = df.loc[df['상태'] == 1, '퇴직일'].dropna().dt.to_period('M').value_counts().sort_index()
            if len(month_counts) >= 2:
                trend = "증가" if month_counts.iloc[-1] > month_counts.iloc[-2] else "감소"
            elif len(month_counts) == 1:
                trend = "변화 판단 불가"
            else:
                trend = "데이터 부족"
        except Exception:
            trend = "데이터 부족"
    else:
        trend = "퇴직일 데이터 없음"

    org_text = "데이터 없음"
    if '소속조직' in df.columns:
        try:
            org_series = df.loc[df['상태'] == 1, '소속조직']
            if '소속조직' in label_encoders:
                try:
                    org_series = pd.Series(label_encoders['소속조직'].inverse_transform(org_series.astype(int)))
                except Exception:
                    pass
            org_summary = org_series.value_counts().head(3)
            if len(org_summary) > 0:
                org_text = ", ".join([f"{idx} 조직 {val}명" for idx, val in org_summary.items()])
        except Exception:
            pass

    reason_text = "데이터 없음"
    if '퇴직사유' in df.columns:
        try:
            vc = df.loc[df['상태'] == 1, '퇴직사유'].astype(str).str.strip().replace(['', 'nan', 'NaN', 'None'], '미기재')
            reason_summary = (vc.value_counts(normalize=True).head(3) * 100).round(1)
            if len(reason_summary) > 0:
                reason_text = ", ".join([f"{idx}({val}%)" for idx, val in reason_summary.items()])
        except Exception:
            pass

    dest_text = "데이터 없음"; num_move = 0
    try:
        moved_df = df[df['상태'] == 1]
        if '퇴직후이직처' in moved_df.columns:
            dest_series = moved_df['퇴직후이직처'].astype(str).str.strip().replace(['', 'nan', 'NaN', 'None'], '미기재')
            num_move = int((dest_series != '미기재').sum())
            top_dest = dest_series[dest_series != '미기재'].value_counts().head(3)
            if len(top_dest) > 0:
                dest_text = ", ".join([f"{idx} {val}명" for idx, val in top_dest.items()])
    except Exception:
        pass

    total_employees = len(df)
    left_count = int(df['상태'].sum())
    active_count = total_employees - left_count
    overall_rate = (left_count / total_employees) * 100 if total_employees else 0
    retention_rate = 100 - overall_rate

    # 전체 예측 퇴직위험 평균(모든 임직원의 예측 확률 평균)
    overall_pred_mean = float(_all_proba.mean() * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("재직자/퇴직자", f"{active_count:,}명/{left_count:,}명")
    c2.metric("퇴직률", f"{overall_rate:.1f}%")
    c3.metric("재직률", f"{retention_rate:.1f}%")
    c4.metric("전체 퇴직위험 평균", f"{overall_pred_mean:.1f}%")

    core_rate_disp = f"{core_rate:.1f}%" if not np.isnan(core_rate) else "-"
    st.markdown(f"""
> 🔹 전체 퇴직률 **{total_rate:.1f}%**, 핵심인재 퇴직률 **{core_rate_disp}**이며 월별 퇴직 추이는 **{trend}** 하고 있습니다.  
> 🔹 조직별 퇴직자는 {org_text} 입니다.  
> 🔹 퇴직 사유로는 **{reason_text}** 이며, 이직 인원 **{num_move}명**의 주요 이직처는 **{dest_text}** 입니다.  
> 🔹 해당 데이터는 사무기술직이 대상이며, 임원 및 계약직은 제외하였습니다.
""")

    a, b = st.columns([1,2])
    with a:
        st.subheader("인원 현황 비율")
        cnts = df['상태'].value_counts()
        labels = ['재직(0)', '퇴직(1)']
        values = [cnts.get(0,0), cnts.get(1,0)]
        # Pie Chart: Use Secondary (Gray) for Stay, Primary (Blue) for Leave - Clean & Minimal
        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.6,
                                         marker_colors=[COLORS['secondary'], COLORS['primary']], textinfo='label+percent')])
        fig_pie.update_layout(height=350, showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        fig_pie.add_annotation(text=f"퇴직률<br>{overall_rate:.1f}%", showarrow=False, font=dict(size=15, color=TEXT_COLOR))
        st.plotly_chart(set_font(fig_pie), use_container_width=True)
    with b:
        st.subheader("퇴직 영향력 상위 변수 (모델 기반)")
        if feature_importance is not None and len(feature_importance) > 0:
            top_imp = feature_importance[feature_importance > 0].head(10)
            if len(top_imp) == 0:
                st.info("영향도가 0으로 계산되었습니다.")
            else:
                top_imp_pct = (top_imp / feature_importance.sum() * 100)
                fig_imp = px.bar(
                    top_imp_pct,
                    orientation="h",
                    color=top_imp_pct.values,
                    color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                    labels={"value": "영향도(%)", "index": "변수"}
                )
                fig_imp.update_layout(height=350, showlegend=False)
                st.plotly_chart(set_font(fig_imp), use_container_width=True)
        else:
            st.info("피처 중요도를 계산할 수 없습니다. 데이터 규모를 확인하세요.")

    # 변수별 퇴직 영향 해석 (XGBoost 영향도 + SHAP 방향)
    if feature_importance is not None and len(feature_importance) > 0:
        top_imp = feature_importance[feature_importance > 0].head(10)
        if len(top_imp) > 0:
            st.markdown("---")
            st.subheader("변수별 퇴직 영향 해석")

            # SHAP은 방향 판단용으로만 사용
            _shap_direction = {}
            try:
                _shap_explainer = shap.TreeExplainer(model.named_steps['model'] if hasattr(model, 'named_steps') else model)
                _shap_values = _shap_explainer.shap_values(df[X.columns])
                if isinstance(_shap_values, list):
                    _shap_values = _shap_values[1]
                _shap_df = pd.DataFrame(_shap_values, columns=X.columns)
                _shap_direction = _shap_df.mean().to_dict()
            except Exception:
                pass

            _total_imp = feature_importance.sum()
            interpretation_rows = []
            for feat in top_imp.index:
                importance_pct = (top_imp[feat] / _total_imp * 100) if _total_imp > 0 else 0
                direction = _shap_direction.get(feat, 0)

                left_group = df[df['상태'] == 1][feat].mean()
                stay_group = df[df['상태'] == 0][feat].mean()

                feat_display = feat
                detail = ""
                if feat in label_encoders:
                    le = label_encoders[feat]
                    classes = list(le.classes_)
                    if left_group > stay_group:
                        high_idx = min(int(round(left_group)), len(classes)-1)
                        detail = f"'{classes[high_idx]}' 값일 때 퇴직 확률 상승"
                    else:
                        low_idx = min(int(round(stay_group)), len(classes)-1)
                        detail = f"'{classes[low_idx]}' 값일 때 퇴직 확률 하락"
                else:
                    if left_group > stay_group:
                        detail = f"값이 높을수록 퇴직 확률 상승 (퇴직자 평균: {left_group:.1f}, 재직자 평균: {stay_group:.1f})"
                    else:
                        detail = f"값이 낮을수록 퇴직 확률 상승 (퇴직자 평균: {left_group:.1f}, 재직자 평균: {stay_group:.1f})"

                arrow = "↑ 퇴직 위험 증가" if direction > 0 else "↓ 퇴직 위험 감소"
                interpretation_rows.append({
                    "변수": feat_display,
                    "영향도": f"{importance_pct:.1f}%",
                    "방향": arrow,
                    "해석": detail
                })

            # 기타 행 추가
            top_sum = sum(top_imp[feat] for feat in top_imp.index)
            etc_pct = ((_total_imp - top_sum) / _total_imp * 100) if _total_imp > 0 else 0
            if etc_pct > 0:
                interpretation_rows.append({
                    "변수": "기타",
                    "영향도": f"{etc_pct:.1f}%",
                    "방향": "-",
                    "해석": f"나머지 {len(feature_importance) - len(top_imp)}개 변수의 합산"
                })

            interp_df = pd.DataFrame(interpretation_rows)
            st.markdown("""
            <div style="background-color: #F8FAFC; padding: 14px 18px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 12px;">
                <p style="margin: 0; font-size: 13px; color: #64748B; line-height: 1.7;">
                <b>영향도</b>: 퇴직 예측에서 해당 변수가 차지하는 비중 (전체 합계 = 100%)<br>
                <b>방향</b>: 해당 변수가 전반적으로 퇴직 확률을 높이는지(↑) 낮추는지(↓)<br>
                <b>해석</b>: 퇴직자와 재직자 그룹 간 실제 평균 비교 기반 설명
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(interp_df, use_container_width=True, hide_index=True)

    # 퇴직 추이 및 사유 현황
    st.markdown("---")
    tcol1, tcol2 = st.columns([1.3, 1])
    with tcol1:
        st.subheader("월별 퇴직 추이")
        if '퇴직일' in df.columns:
            df_left = df[(df['상태'] == 1) & (df['퇴직일'].notna())].copy()
            if len(df_left) > 0:
                df_left['퇴직년월'] = df_left['퇴직일'].dt.to_period('M')
                monthly = df_left.groupby('퇴직년월').size().reset_index()
                monthly.columns = ['년월', '퇴직자 수']
                monthly['월라벨'] = monthly['년월'].astype(str)

                max_cnt = int(monthly['퇴직자 수'].max()) if len(monthly)>0 else 0
                y_max = max_cnt * 1.25 + 0.5 if max_cnt > 0 else 1
                fig_month = go.Figure()
                # Bar: Primary Blue, Line: Darker Blue or Slate
                fig_month.add_bar(
                    x=monthly['월라벨'], y=monthly['퇴직자 수'], name='퇴직자 수', marker_color=COLORS['primary'],
                    text=monthly['퇴직자 수'], textposition='outside', cliponaxis=False
                )
                fig_month.add_trace(go.Scatter(x=monthly['월라벨'], y=monthly['퇴직자 수'],
                                               mode='lines+markers', name='추세선', line=dict(color=COLORS['secondary'], width=3)))
                fig_month.update_layout(xaxis_title="월", yaxis_title="명", height=320)
                fig_month.update_yaxes(range=[0, y_max])
                st.plotly_chart(set_font(fig_month), use_container_width=True)
            else:
                st.info("퇴직일 데이터가 비어 있어 추이를 표시할 수 없습니다.")
        else:
            st.info("'퇴직일' 컬럼이 없어 월별 퇴직 추이를 표시할 수 없습니다.")
    with tcol2:
        st.subheader("퇴직 사유")
        if '퇴직사유' in df.columns:
            reason_df = df[df['상태'] == 1]
            vc = reason_df['퇴직사유'].astype(str).str.strip()
            # 빈값과 미기재 항목 제외
            vc = vc[~vc.isin(['', 'nan', 'NaN', 'None', '미기재'])]
            counts_series = vc.value_counts(dropna=True)
            if counts_series.sum() > 0:
                top_n = 9
                top_counts = counts_series.head(top_n)
                others = counts_series.iloc[top_n:].sum()
                labels = list(top_counts.index)
                values = list(top_counts.values)
                if others > 0:
                    labels.append('기타')
                    values.append(others)

                pie_df = pd.DataFrame({'퇴직사유': labels, '건수': values})
                # Pie: Use the defined sequence
                fig_reason = px.pie(pie_df, names='퇴직사유', values='건수', hole=0.45,
                                    color_discrete_sequence=COLORS['sequence'])
                fig_reason.update_traces(textposition='inside', textinfo='percent+label')
                fig_reason.update_layout(height=320, showlegend=True)
                st.plotly_chart(set_font(fig_reason), use_container_width=True)
            else:
                st.info("퇴직 사유 데이터가 없어 현황을 표시할 수 없습니다.")
        else:
            st.info("'퇴직사유' 컬럼이 없어 현황을 표시할 수 없습니다.")

    # 퇴직 후 이직처 현황
    st.subheader("주요 이직처")
    if '퇴직후이직처' in df.columns:
        moved_df = df[df['상태'] == 1]
        dest = moved_df['퇴직후이직처'].astype(str).str.strip()
        # 빈값과 미기재 항목 제외
        dest = dest[~dest.isin(['', 'nan', 'NaN', 'None', '미기재'])]
        dest_counts = dest.value_counts(dropna=True)
        if dest_counts.sum() > 0:
            dest_df = dest_counts.head(15).reset_index()
            dest_df.columns = ['이직처', '건수']
            # Bar: Unified Blue Gradient
            fig_dest = px.bar(dest_df, x='건수', y='이직처', orientation='h',
                              labels={'건수':'건수','이직처':'이직처'},
                              color='건수', color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                              text='건수')
            x_max = float(dest_df['건수'].max()) if len(dest_df)>0 else 0
            fig_dest.update_traces(textposition='outside', cliponaxis=False)
            fig_dest.update_layout(height=320, showlegend=False,
                                   xaxis=dict(range=[0, x_max*1.15 if x_max>0 else 1]))
            st.plotly_chart(set_font(fig_dest), use_container_width=True)
        else:
            st.info("이직처 데이터가 없어 현황을 표시할 수 없습니다.")
    else:
        st.info("'퇴직후이직처' 컬럼이 없어 현황을 표시할 수 없습니다.")

    st.markdown("---")
    st.subheader("주요 변수별 퇴직률 분포 현황")



    cols = st.columns(2)
    for i, var in enumerate(top_features):
        with cols[i % 2]:
            if var in label_encoders:
                df_plot = df.copy()
                df_plot[var+"_name"] = label_encoders[var].inverse_transform(df[var])
                group_rates = df_plot.groupby(var+"_name")['상태'].mean()*100
                x_vals = group_rates.index
                # Bar: Unified Blue Gradient
                fig_bar = px.bar(x=x_vals, y=group_rates.values, color=group_rates.values,
                                 color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                                 title=f"{var} 별 퇴직률(%)", labels={'x':var,'y':'퇴직률(%)'})
                fig_bar.update_layout(height=300, showlegend=False)
                st.plotly_chart(set_font(fig_bar), use_container_width=True)

                max_grp = group_rates.idxmax()
                rate = float(group_rates.max())
                n = int((df_plot[var+"_name"] == max_grp).sum())
                share = n / len(df_plot) * 100
                diff = rate - overall_rate  # 전체 평균 퇴직률 대비 차이 (%p)

                # 위험 레벨 라벨
                if diff >= 3:
                    risk_label = "고위험"
                elif diff >= 0:
                    risk_label = "주의"
                else:
                    risk_label = "낮은 위험"

                accent_color = "#06B6D4"  # 기존 카드 색 유지

                st.markdown(f"""
                <div style="
                    background-color: white;
                    border: 1px solid #E5E7EB;
                    border-left: 4px solid {accent_color};
                    border-radius: 8px;
                    padding: 20px;
                    margin: 16px 0;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <span style="
                            font-weight: 700; 
                            color: {accent_color}; 
                            font-size: 16px;
                            letter-spacing: -0.02em;
                        ">주의 그룹: {var} = '{max_grp}'</span>
                    </div>
                    <div style="color: #374151; font-size: 14px; line-height: 1.6;">
                        <div style="margin-bottom: 8px; display: flex; justify-content: space-between;">
                            <span style="color: #6B7280;">퇴직률</span>
                            <span>
                                <strong style="color: #111827;">{rate:.1f}%</strong>
                                <span style="color: {accent_color}; font-size: 0.9em; margin-left: 4px;">
                                    ({diff:+.1f}%p {('높음' if diff > 0 else '낮음' if diff < 0 else '동일')}, {risk_label})
                                </span>
                            </span>
                        </div>
                        <div style="margin-bottom: 0px; display: flex; justify-content: space-between;">
                            <span style="color: #6B7280;">대상 인원</span>
                            <span>
                                <strong style="color: #111827;">{n:,}명 / 전체 {total_employees:,}명</strong>
                                <span style="color: #9CA3AF; font-size: 0.9em; margin-left: 4px;">
                                    (해당 그룹 비중 {share:.1f}%)
                                </span>
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                try:
                    bins = bucketize_numeric(df[var], bins="quartile")
                    df_tmp = pd.DataFrame({var: df[var], 'bin': bins, '상태': df['상태']})
                    group_rates = df_tmp.groupby('bin')['상태'].mean()*100
                    nice_labels = [humanize_interval_label(var, b) for b in group_rates.index]

                    # Bar: Unified Blue Gradient
                    fig_bar = px.bar(x=nice_labels, y=group_rates.values,
                                     color=group_rates.values, color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                                     title=f"{var} 별 퇴직률(%)", labels={'x':'구간','y':'퇴직률(%)'})
                    fig_bar.update_layout(height=300, showlegend=False)
                    st.plotly_chart(set_font(fig_bar), use_container_width=True)

                    idxmax = group_rates.idxmax()
                    rate = float(group_rates.max())
                    bucket_label = humanize_interval_label(var, idxmax)
                    n = int((df_tmp['bin'] == idxmax).sum())
                    share = n / len(df_tmp) * 100

                    if var in ['기본급','연봉','급여','월급']:
                        action = "급여 밴드 재설계 및 핵심 보상 조정"
                    elif var in ['근무연수','승진후경과연수']:
                        action = "온보딩·멘토링 및 승진 로드맵 강화"
                    elif var in ['나이','연령','보유역량']:
                        action = "경력개발·역할 확장 및 멘토링"
                    else:
                        action = "리텐션 정책 점검"

                    diff = rate - overall_rate  # 전체 평균 대비 차이 (%p)

                    # 위험 레벨
                    if diff >= 3:
                        risk_label = "고위험"
                    elif diff >= 0:
                        risk_label = "주의"
                    else:
                        risk_label = "낮은 위험"

                    accent_color = "#06B6D4"

                    st.markdown(f"""
                    <div style="
                        background-color: white;
                        border: 1px solid #E5E7EB;
                        border-left: 4px solid {accent_color};
                        border-radius: 8px;
                        padding: 20px;
                        margin: 16px 0;
                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
                    ">
                        <div style="display: flex; align-items: center; margin-bottom: 12px;">
                            <span style="
                                font-weight: 700; 
                                color: {accent_color}; 
                                font-size: 16px;
                                letter-spacing: -0.02em;
                            ">주의 그룹: {var} = {bucket_label}</span>
                        </div>
                        <div style="color: #374151; font-size: 14px; line-height: 1.6;">
                            <div style="margin-bottom: 8px; display: flex; justify-content: space-between;">
                                <span style="color: #6B7280;">퇴직률</span>
                                <span>
                                    <strong style="color: #111827;">{rate:.1f}%</strong>
                                    <span style="color: {accent_color}; font-size: 0.9em; margin-left: 4px;">
                                        ({diff:+.1f}%p {('높음' if diff > 0 else '낮음' if diff < 0 else '동일')}, {risk_label})
                                    </span>
                                </span>
                            </div>
                            <div style="margin-bottom: 0px; display: flex; justify-content: space-between;">
                                <span style="color: #6B7280;">대상 인원</span>
                                <span>
                                    <strong style="color: #111827;">{n:,}명 / 전체 {total_employees:,}명</strong>
                                    <span style="color: #9CA3AF; font-size: 0.9em; margin-left: 4px;">
                                        (해당 그룹 비중 {share:.1f}%)
                                    </span>
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception:
                    st.info(f"{var}: 버킷팅/집계가 어려워 해설을 생략합니다.")

    # 상관분석
    st.markdown("---")
    st.subheader("숫자형 변수별 퇴직 영향")
    num_cols = [c for c in X.columns if (c not in label_encoders) and pd.api.types.is_numeric_dtype(df[c])]
   
    if len(num_cols) > 0:
        pearson = df[num_cols + ['상태']].corr(numeric_only=True)['상태'].drop('상태').sort_values(key=np.abs, ascending=False)
        pearson_df = pearson.reset_index().rename(columns={'index': '변수', '상태': '상관계수'})
        pearson_df['상관계수'] = pearson_df['상관계수'].round(2)


        # 해석 컬럼 추가
        def _num_interpret(row):
            v = row['상관계수']
            name = row['변수']
            strength = "강한" if abs(v) >= 0.5 else ("보통" if abs(v) >= 0.3 else "약한")
            if v > 0:
                return f"{strength} 관련 | {name} 값이 높을수록 퇴직 가능성 증가"
            elif v < 0:
                return f"{strength} 관련 | {name} 값이 높을수록 퇴직 가능성 감소"
            else:
                return "관련 없음"

        pearson_df['해석'] = pearson_df.apply(_num_interpret, axis=1)
        fig_corr = px.bar(x=pearson_df['변수'], y=pearson_df['상관계수'],
                          color=pearson_df['상관계수'], color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])])
        fig_corr.update_layout(yaxis_title="상관계수", height=300)
        st.plotly_chart(set_font(fig_corr), use_container_width=True)
        show_table_centered(pearson_df)

    st.subheader("범주형 변수별 퇴직 영향")
    cat_vs = []
    for c in [col for col in CAT_COLS if col in df.columns]:
        try:
            cat_vs.append((c, cramers_v(df[c], df['상태'])))
        except Exception:
            pass
    if len(cat_vs) > 0:
        cv_df = pd.DataFrame(cat_vs, columns=['변수','관련도']).sort_values('관련도', ascending=False)
        cv_df['관련도'] = cv_df['관련도'].round(2)

        def _cat_interpret(row):
            v = row['관련도']
            strength = "강한" if v >= 0.5 else ("보통" if v >= 0.3 else "약한")
            return f"{strength} 관련 | {row['변수']}에 따라 퇴직 비율 차이가 {'크게' if v >= 0.3 else '다소'} 존재"

        cv_df['해석'] = cv_df.apply(_cat_interpret, axis=1)
        show_table_centered(cv_df)

        st.markdown("""
        <div style="background-color: #F8FAFC; padding: 14px 18px; border-radius: 8px; border: 1px solid #E2E8F0; margin-top: 12px;">
            <p style="margin: 0; font-size: 13px; color: #64748B; line-height: 1.7;">
            <b>상관계수</b> (숫자형): -1 ~ +1 범위. 양수면 값이 높을수록 퇴직 증가, 음수면 감소<br>
            <b>관련도</b> (범주형): 0 ~ 1 범위. 1에 가까울수록 해당 범주에 따라 퇴직 비율 차이가 큼
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("조직별 예측 퇴직 위험 순위")
    dept_col = None
    for col in ['소속조직','팀','직무','직위','직책']:
        if col in df.columns:
            dept_col = col; break
    if dept_col:
        df_pred = df.copy()
        df_pred['퇴직예측확률'] = _all_proba
        dept_risk = df_pred.groupby(dept_col)['퇴직예측확률'].mean().sort_values(ascending=False)
        top5 = dept_risk.head(5).reset_index().rename(columns={'퇴직예측확률':'평균 퇴직위험(%)'})
        top5['평균 퇴직위험(%)'] = top5['평균 퇴직위험(%)'].apply(lambda x: f"{x*100:.1f}%")
        if dept_col in label_encoders:
            top5[dept_col] = label_encoders[dept_col].inverse_transform(top5[dept_col])
        show_table_centered(top5)
        if len(dept_risk) > 0:
            top_group_val = dept_risk.index[0]
            top_group_prob = float(dept_risk.iloc[0])
            if dept_col in label_encoders:
                try:
                    top_group_label = label_encoders[dept_col].inverse_transform([int(top_group_val)])[0]
                except Exception:
                    top_group_label = str(top_group_val)
            else:
                top_group_label = str(top_group_val)

            dept_df = df_pred[df_pred[dept_col] == top_group_val]
            global_rate = df['상태'].mean()
            reason_phrases = []

            for var in top_features[:5]:
                if var not in df.columns:
                    continue
                if pd.api.types.is_numeric_dtype(df[var]):
                    try:
                        dept_mean = float(dept_df[var].mean())
                        overall_mean = float(df[var].mean())
                        if np.isnan(overall_mean) or overall_mean == 0:
                            continue
                        ratio = dept_mean / overall_mean
                        lower_bad = ['기본급', '연봉', '급여', '월급', '만족도', '근무연수']
                        higher_bad = ['연장근무', '야근', '입사전이직횟수', '이직횟수']
                        if any(k in var for k in lower_bad) and ratio <= 0.9:
                            reason_phrases.append(f"{var} 수준이 전체 대비 낮은 편")
                        elif any(k in var for k in higher_bad) and ratio >= 1.1:
                            if '입사전이직횟수' in var or '이직횟수' in var:
                                reason_phrases.append("입사 전 이직횟수가 높은 인력이 다수 포함된")
                            else:
                                reason_phrases.append(f"{var} 수준이 전체 대비 높은 편")
                    except Exception:
                        pass
                if var in label_encoders:
                    try:
                        grp_rate_all = df.groupby(var)['상태'].mean()
                        grp_rate_dept = dept_df.groupby(var)['상태'].mean()
                        if len(grp_rate_dept) == 0:
                            continue
                        cat_val = grp_rate_dept.idxmax()
                        if cat_val in grp_rate_all.index:
                            dept_rate = grp_rate_dept[cat_val]
                            overall_cat_rate = grp_rate_all[cat_val]
                            if dept_rate >= overall_cat_rate * 1.2 and dept_rate > global_rate:
                                try:
                                    cat_label = label_encoders[var].inverse_transform([int(cat_val)])[0]
                                except Exception:
                                    cat_label = str(cat_val)
                                reason_phrases.append(f"{var} 중 '{cat_label}' 그룹의 퇴직률이 전체 대비 높은 편")
                    except Exception:
                        pass
                if len(reason_phrases) >= 2:
                    break

            head_sentence = f"{top_group_label} 조직의 평균 예측 퇴직위험은 **{top_group_prob*100:.1f}%**입니다."
            if reason_phrases:
                if len(reason_phrases) == 1:
                    tail_sentence = f"조직 내 {reason_phrases[0]} 것이 퇴직 위험 상승에 영향을 준 것으로 보입니다."
                else:
                    tail_sentence = (f"조직 내 {reason_phrases[0]}이며 {reason_phrases[1]} 것이 "
                                     f"퇴직 위험 상승에 영향을 준 것으로 보입니다.")
            else:
                tail_sentence = "단일 요인보다는 여러 변수의 복합적인 패턴이 반영된 결과로 해석됩니다."
            st.markdown(head_sentence + "  \n" + tail_sentence)

    st.divider()
    st.subheader("개인별 예측 퇴직 위험 순위")
    df_pred2 = df.copy()
    df_pred2['퇴직예측확률'] = _all_proba
    if '상태' in df_pred2.columns:
        active_pred = df_pred2[df_pred2['상태'] == 0].copy()
    else:
        active_pred = df_pred2.copy()

    if len(active_pred) == 0:
        st.info("재직 중인 직원이 없어 개인별 예측 순위를 표시할 수 없습니다.")
    else:
        top10 = active_pred.sort_values('퇴직예측확률', ascending=False).head(10)
        disp_cols = [c for c in ['사원번호','이름','직무','소속조직','팀','직책'] if c in df.columns] + ['퇴직예측확률']
        top10_disp = top10[disp_cols].rename(columns={'퇴직예측확률':'퇴직위험확률'})
        top10_disp['퇴직위험확률'] = top10_disp['퇴직위험확률'].apply(lambda x: f"{x*100:.1f}%")
        for c in ['직무','소속조직','팀','직책']:
            if c in label_encoders and c in top10_disp.columns:
                top10_disp[c] = label_encoders[c].inverse_transform(top10_disp[c])
        show_table_centered(top10_disp)

    st.markdown("---")
    with st.expander("모델 설명 및 신뢰도", expanded=False):

        # 쉬운 설명
        _acc = metrics['accuracy'] * 100
        _f1 = metrics['f1']
        _roc = metrics['roc_auc']

        # 신뢰도 등급
        if _roc >= 0.9:
            _grade = "매우 높음"
        elif _roc >= 0.8:
            _grade = "높음"
        elif _roc >= 0.7:
            _grade = "보통"
        else:
            _grade = "낮음"

        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        _total_test = tn + fp + fn + tp
        _caught = tp
        _missed = fn
        _false_alarm = fp

        st.markdown(f"""
---
### 이 예측 모델, 쉽게 이해하기

**1. 이 모델은 무엇을 하나요?**

직원들의 근속연수, 직무, 조직, 급여 등 **여러 정보를 종합**해서,
"이 직원이 앞으로 퇴직할 가능성이 높은지 낮은지"를 **확률(%)**로 알려줍니다.
예를 들어 퇴직 위험 72%라면, 과거에 비슷한 조건의 직원 100명 중 약 72명이 퇴직했다는 의미입니다.

**2. 얼마나 정확한가요?**

모델을 학습에 사용하지 않은 별도 데이터(**{_total_test}명**)로 검증한 결과입니다.
- 전체 정확도: **{_acc:.0f}%** (100명 중 약 {_acc:.0f}명을 맞춤)
- 실제 퇴직자 {tp + fn}명 중 **{_caught}명을 사전에 포착**했고, {_missed}명은 놓쳤습니다.
- 재직자 중 **{_false_alarm}명은 퇴직으로 잘못 예측**했습니다 (과잉 경보).
- 모델 신뢰도 등급: **{_grade}**

**3. 영향도(%)는 무엇인가요?**

모델이 퇴직 여부를 판단할 때 **어떤 정보를 얼마나 중요하게 봤는지**를 비율로 나타낸 것입니다.
예: 핵심인재 여부 38%, 근속연수 15% → 모델이 퇴직을 예측할 때 핵심인재 여부를 가장 많이 참고했다는 뜻입니다.

**4. 상관계수 / 관련도는 무엇인가요?**

- **상관계수**(-1 ~ +1): 숫자형 변수와 퇴직의 관계입니다.
  - +0.5 이상이면 "그 값이 높을수록 퇴직이 많다"
  - -0.5 이하면 "그 값이 높을수록 퇴직이 적다"
  - 0에 가까우면 퇴직과 별 관련이 없습니다.
- **관련도**(0 ~ 1): 범주형 변수(직무, 조직 등)와 퇴직의 관계입니다.
  - 1에 가까울수록 "어떤 그룹이냐에 따라 퇴직 비율 차이가 크다"는 뜻입니다.

**5. 주의할 점**

- 이 모델은 **과거 데이터의 패턴**을 학습한 것이므로, 미래를 100% 맞추지는 못합니다.
- 경기 변동, 조직개편, 개인 사정 등 **데이터에 없는 요인은 반영되지 않습니다.**
- "퇴직 위험이 높다" = "반드시 퇴직한다"가 아니라, **"관심을 가지고 살펴볼 필요가 있다"**는 신호입니다.
- HR담당자와 리더의 **정성적 판단과 함께 보조 도구로 활용**하는 것을 권장합니다.

---
        """)

        st.divider()

        # 혼동행렬 (1개로 통합)
        st.subheader("예측 정확도 상세")
        colA, colB = st.columns([1, 1])

        with colA:
            cm = metrics['confusion_matrix']
            fig_cm = px.imshow(
                cm,
                labels=dict(x="예측", y="실제", color="건수"),
                x=['재직(0)','퇴직(1)'],
                y=['재직(0)','퇴직(1)'],
                text_auto=True,
                color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                title="혼동행렬 (테스트셋)"
            )
            st.plotly_chart(set_font(fig_cm), use_container_width=True)

        with colB:
            # 혼동행렬 쉬운 해석
            cm = metrics['confusion_matrix']
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            st.markdown(f"""
            <div style="background-color: #F8FAFC; padding: 16px; border-radius: 8px; border: 1px solid #E2E8F0;">
                <p style="font-size: 14px; color: #334155; line-height: 2.0; margin: 0;">
                    <b>재직자를 재직으로 맞춘 경우:</b> {tn}명<br>
                    <b>퇴직자를 퇴직으로 맞춘 경우:</b> {tp}명<br>
                    <b>재직자인데 퇴직으로 잘못 예측 (과잉 경보):</b> {fp}명<br>
                    <b>퇴직자인데 재직으로 잘못 예측 (놓친 퇴직자):</b> {fn}명
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # 성능 지표 상세
        st.subheader("성능 지표 상세")
        st.markdown("""
**모델 성능 요약 (테스트 데이터 기준)**

- 정확도(Accuracy): 전체 예측 중 맞춘 비율 → **{:.1f}%**
- F1 Score: 퇴직자를 놓치지 않으면서도, 과잉 경보를 얼마나 줄였는지 보는 균형 지표 → **{:.2f}**
- ROC AUC: 재직자와 퇴직자를 얼마나 잘 구분하는지(1에 가까울수록 좋음) → **{:.2f}**
- PR-AUC: 실제 퇴직자가 적은 상황에서,
  '위험이라고 찍은 사람들 중 진짜 퇴직자 비율'을 얼마나 잘 유지하는지 보는 지표 → **{:.2f}**
        """.format(
            metrics['accuracy']*100,
            metrics['f1'],
            metrics['roc_auc'],
            metrics['pr_auc']
        ))

        st.divider()

        # 모델 설명
        st.subheader("모델 작동 원리")
        st.markdown("""
- 이 대시보드는 **XGBoost 기반 이진 분류 모델**로,
  각 직원의 특성을 입력받아 **퇴직(1) / 재직(0) 확률**을 예측합니다.
- 직무, 조직, 연장근무, 보상수준 등 여러 변수가 서로 섞여 작용하는 패턴을 함께 학습합니다.
- 경기, 조직개편, 경영전략 변화처럼 데이터에 없는 외부 요인은 반영하지 못하므로,
  **HR/리더의 정성적 판단과 함께 쓰는 보조 도구**로 보는 것이 적절합니다.
        """)

# =========================
# 2) 핵심인재 현황
# =========================
if menu == "핵심인재 현황":
    add_pdf_button()
    st.title("핵심인재 퇴직예측")

    core_col = '핵심인재'
    if core_col not in df.columns:
        st.error("'핵심인재' 컬럼이 없습니다.")
    else:
        # 🔹 1단계: 핵심인재 전체(재직+퇴직) 추출
        if core_col in label_encoders:
            classes = list(label_encoders[core_col].classes_)
            neg_set = {'미입력', 'NAN', 'NONE', '', 'N', 'NO', 'FALSE', '0', 'nan'}
            pos_idx = [i for i, v in enumerate(classes) if str(v).strip().upper() not in neg_set]
            core_mask = df[core_col].isin(pos_idx)
        else:
            neg_set = {'미입력', 'NAN', 'NONE', '', 'N', 'NO', 'FALSE', '0', 'nan'}
            core_mask = ~df[core_col].astype(str).str.strip().str.upper().isin(neg_set)

        core_all = df.loc[core_mask]          # 🔹 핵심인재 전체 (재직+퇴직)
        core_active = core_all[core_all['상태'] == 0].copy()  # 🔥 예측 대상: 재직 핵심인재만

        total_core = len(core_all)
        core_left = int(core_all['상태'].sum())
        core_rate = (core_left / total_core * 100) if total_core > 0 else 0
        all_rate = df['상태'].mean() * 100

        core_pred_mean = (
            model.predict_proba(core_active[X.columns])[:, 1].mean() * 100
            if len(core_active) > 0 else 0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("핵심인재 수", f"{total_core:,}명")
        c2.metric("핵심인재 퇴직자", f"{core_left}명")
        c3.metric("핵심인재 퇴직률", f"{core_rate:.1f}%")
        c4.metric("핵심인재 퇴직위험 평균", f"{core_pred_mean:.1f}%")

        try:
            # -----------------------------
            # 🔥 2) 예측 퇴직위험(고위험군) — 재직 핵심인재만
            # -----------------------------
            if len(core_active) > 0:
                core_pred = core_active.copy()
                core_pred['퇴직예측확률'] = model.predict_proba(core_pred[X.columns])[:, 1]

                threshold_90 = core_pred['퇴직예측확률'].quantile(0.90)
                high_risk_core = core_pred[core_pred['퇴직예측확률'] >= threshold_90]
                high_risk_count = len(high_risk_core)
                high_risk_rate = (high_risk_count / len(core_pred)) * 100
            else:
                high_risk_count = 0
                high_risk_rate = 0

            # -----------------------------
            # 🔹 3) 핵심인재 퇴직자 중 주요 퇴직사유 — 퇴직한 핵심인재 기준
            # -----------------------------
            if '퇴직사유' in core_all.columns:
                reason_series = (
                    core_all[core_all['상태'] == 1]['퇴직사유']
                    .astype(str).str.strip()
                    .replace(['', 'nan', 'None'], '미기재')
                )
                reason_top = reason_series.value_counts(normalize=True).head(3) * 100
                reason_text = ", ".join([f"{idx} {val:.1f}%" for idx, val in reason_top.items()]) \
                              if len(reason_top) > 0 else "데이터 없음"
            else:
                reason_text = "퇴직사유 데이터 없음"

            # -----------------------------
            # 🔹 4) 요약 문구
            # -----------------------------
            st.markdown(f"""
> 🔹 핵심인재 총 **{total_core}명** 중 실제 퇴직자는 **{core_left}명({core_rate:.1f}%)**입니다.  
> 🔹 현재 재직 중인 핵심인재 중 **{high_risk_count}명({high_risk_rate:.1f}%)**이 AI 기준 상위 10% 고위험군입니다.  
> 🔹 핵심인재 퇴직의 주요 사유는 **{reason_text}** 입니다.  
            """)

        except Exception:
            st.info("핵심인재 요약 인사이트를 계산할 수 없습니다. 데이터 구조를 확인해주세요.")

        # -----------------------------
        # 🆕 NEW: 퇴직위험 등급 분포 + 조직별 히트맵
        # -----------------------------
        st.markdown("---")
        if len(core_active) > 0:
            _core_proba = model.predict_proba(core_active[X.columns])[:, 1]

            # 등급 분류
            high_cnt = int((_core_proba >= 0.70).sum())
            mid_cnt = int(((_core_proba >= 0.30) & (_core_proba < 0.70)).sum())
            low_cnt = int((_core_proba < 0.30).sum())

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.subheader("퇴직위험 등급 분포")
                risk_labels = ['고위험 (≥70%)', '중위험 (30~70%)', '저위험 (<30%)']
                risk_values = [high_cnt, mid_cnt, low_cnt]
                risk_colors = ['#F4A7A7', '#FDE68A', '#48C0D8']
                fig_risk = go.Figure(data=[go.Pie(
                    labels=risk_labels, values=risk_values, hole=0.55,
                    marker_colors=risk_colors,
                    textinfo='percent',
                    textposition='inside',
                    insidetextfont=dict(size=13),
                    hovertemplate='%{label}<br>%{value}명 (%{percent})<extra></extra>'
                )])
                fig_risk.update_layout(
                    height=350, showlegend=True,
                    legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.02),
                    margin=dict(l=20, r=130, t=20, b=20)
                )
                fig_risk.add_annotation(text=f"재직<br>핵심인재<br>{len(core_active)}명",
                    showarrow=False, font=dict(size=13, color=TEXT_COLOR))
                st.plotly_chart(set_font(fig_risk), use_container_width=True)

            with chart_col2:
                st.subheader("조직별 핵심인재 퇴직위험")
                if '소속조직' in core_active.columns:
                    _ca_tmp = core_active.copy()
                    _ca_tmp['_pred'] = _core_proba
                    org_risk = _ca_tmp.groupby('소속조직')['_pred'].mean().sort_values(ascending=True)
                    org_labels = []
                    for idx in org_risk.index:
                        if '소속조직' in label_encoders:
                            try:
                                org_labels.append(label_encoders['소속조직'].inverse_transform([int(idx)])[0])
                            except Exception:
                                org_labels.append(str(idx))
                        else:
                            org_labels.append(str(idx))
                    def _risk_color(v):
                        if v >= 0.70: return '#F4A7A7'
                        elif v >= 0.30: return '#FDE68A'
                        else: return '#48C0D8'
                    fig_org = go.Figure(go.Bar(
                        x=org_risk.values * 100, y=org_labels,
                        orientation='h',
                        marker_color=[_risk_color(v) for v in org_risk.values],
                        text=[f'{v*100:.1f}%' for v in org_risk.values],
                        textposition='outside'
                    ))
                    fig_org.update_layout(height=350, xaxis_title='평균 퇴직위험(%)', yaxis_title='',
                        margin=dict(l=100))
                    st.plotly_chart(set_font(fig_org), use_container_width=True)
                else:
                    st.info("'소속조직' 컬럼이 없어 조직별 위험도를 표시할 수 없습니다.")

        # -----------------------------
        # 🆕 NEW: 핵심인재 고위험 Top 10
        # -----------------------------
        st.markdown("---")
        st.subheader("핵심인재 고위험군")
        if len(core_active) > 0:
            _top10_df = core_active.copy()
            _top10_df['퇴직예측확률'] = model.predict_proba(_top10_df[X.columns])[:, 1]
            _top10 = _top10_df.sort_values('퇴직예측확률', ascending=False).head(10)

            top10_disp = _top10.copy()
            top10_base = ['사원번호','이름','소속조직','팀','직책','직무','평가등급']
            top10_show = [c for c in top10_base if c in top10_disp.columns] + ['예측퇴직위험']
            top10_disp['예측퇴직위험'] = top10_disp['퇴직예측확률'].apply(lambda x: f"{x*100:.1f}%")
            for c in ['소속조직','팀','직책','직무','평가등급']:
                if c in label_encoders and c in top10_disp.columns:
                    top10_disp[c] = label_encoders[c].inverse_transform(top10_disp[c])

            # 위험 등급 컬럼 추가
            def _risk_badge(prob):
                if prob >= 0.70:
                    return '<span style="color:#F4A7A7;">●</span> 고위험'
                elif prob >= 0.30:
                    return '<span style="color:#FDE68A;">●</span> 중위험'
                else:
                    return '<span style="color:#48C0D8;">●</span> 저위험'
            top10_disp['위험등급'] = _top10['퇴직예측확률'].apply(_risk_badge)
            top10_show.insert(-1, '위험등급')

            show_table_centered(top10_disp[top10_show])
            st.caption("상위 10명은 AI 예측 기반 퇴직 확률이 가장 높은 핵심인재이며, 선제적 리텐션 조치가 필요합니다.")
        else:
            st.info("재직 중인 핵심인재가 없어 Top 10을 표시할 수 없습니다.")

        st.markdown("---")
        st.subheader("핵심인재 전체 리스트")

        # -----------------------------
        # 🔥 5) 핵심인재 전체 리스트 — 재직 핵심인재만 예측
        # -----------------------------
        if len(core_active) > 0:
            X_core = core_active[X.columns]
            core_df_pred = core_active.copy()
            core_df_pred['퇴직예측확률'] = model.predict_proba(X_core)[:,1]

            # ① 통계 캐시 생성 (전체 df 기준) — 확장 버전
            stats_cache = {}
            for f in top_features:
                if f in X.columns:
                    if f in label_encoders:
                        grp = df.groupby(f)['상태'].agg(['mean','count'])
                        stats_cache[f] = {
                            'type': 'cat',
                            'rates': grp['mean'].to_dict(),
                            'counts': grp['count'].to_dict()
                        }
                    elif pd.api.types.is_numeric_dtype(X[f]):
                        stats_cache[f] = {
                            'type': 'num',
                            'mean': float(X[f].mean()),
                            'std': float(X[f].std()) if X[f].std() > 0 else 1.0
                        }
                    else:
                        grp = df.groupby(f)['상태'].agg(['mean','count'])
                        stats_cache[f] = {
                            'type': 'cat',
                            'rates': grp['mean'].to_dict(),
                            'counts': grp['count'].to_dict()
                        }
            
            global_rate = df['상태'].mean()

            # ② 예측사유 생성 함수 — 구체적 수치 포함
            def get_reason(row):
                reasons = []
                for f in top_features[:5]:
                    if f not in stats_cache:
                        continue
                    info = stats_cache[f]
                    val = row[f]

                    # 숫자형 변수
                    if info['type'] == 'num':
                        avg = info['mean']
                        if avg == 0:
                            continue
                        ratio = val / avg
                        pct = ratio * 100

                        # 단위 결정
                        salary_like = ['기본급','연봉','월급','급여']
                        years_like = ['근무연수','승진후경과연수']
                        age_like = ['나이','연령']
                        count_like = ['이직횟수','입사전이직횟수','보유역량']

                        if any(k in f for k in salary_like):
                            unit = "만원"
                        elif any(k in f for k in years_like):
                            unit = "년"
                        elif any(k in f for k in age_like):
                            unit = "세"
                        else:
                            unit = ""

                        val_str = f"{val:,.0f}{unit}" if unit else f"{val:,.1f}"
                        avg_str = f"{avg:,.0f}{unit}" if unit else f"{avg:,.1f}"

                        lower_bad = ['기본급','연봉','월급','급여','만족도','워라밸','환경만족','관계만족','근무연수','보유역량']
                        higher_bad = ['야근','연장근무','초과근무','이직횟수','입사전이직횟수','통근거리','거리','승진후경과연수']

                        if any(lb in f for lb in lower_bad) and ratio < 0.85:
                            reasons.append(f"{f} {val_str} (평균 {avg_str}의 {pct:.0f}%↓)")
                        elif any(hb in f for hb in higher_bad) and ratio > 1.15:
                            reasons.append(f"{f} {val_str} (평균 {avg_str}의 {pct:.0f}%↑)")
                        else:
                            if ratio < 0.7:
                                reasons.append(f"{f} {val_str} (평균 {avg_str}의 {pct:.0f}%↓)")
                            elif ratio > 1.3:
                                reasons.append(f"{f} {val_str} (평균 {avg_str}의 {pct:.0f}%↑)")

                    # 범주형 변수
                    else:
                        rates = info['rates']
                        if val in rates:
                            grp_rate = rates[val]
                            if grp_rate > global_rate * 1.2:
                                if f in label_encoders:
                                    try:
                                        val_label = label_encoders[f].inverse_transform([int(val)])[0]
                                    except Exception:
                                        val_label = str(val)
                                else:
                                    val_label = str(val)
                                multiplier = grp_rate / global_rate if global_rate > 0 else 0
                                reasons.append(
                                    f"{f} '{val_label}' 퇴직률 {grp_rate*100:.1f}% (평균 {global_rate*100:.1f}%의 {multiplier:.1f}배)"
                                )

                if reasons:
                    return " / ".join(reasons)
                else:
                    # 복합 요인일 때 가장 영향력 높은 변수 언급
                    top_f = top_features[0] if len(top_features) > 0 else ""
                    return f"복합 요인 (주요 영향: {top_f})" if top_f else "복합 요인"

            core_df_pred['예측사유'] = core_df_pred.apply(get_reason, axis=1)

            all_core = core_df_pred.sort_values('퇴직예측확률', ascending=False)

            base_cols = ['사원번호','이름','소속조직','팀','직책','직무','평가등급','인센티브']
            final_cols = [c for c in base_cols if c in all_core.columns] + ['예측퇴직위험', '예측사유']

            disp = all_core.copy()
            disp = disp.rename(columns={'퇴직예측확률':'예측퇴직위험'})
            disp['예측퇴직위험'] = disp['예측퇴직위험'].apply(lambda x: f"{x*100:.1f}%")

            for c in ['소속조직','팀','직책','직무','평가등급','인센티브']:
                if c in label_encoders and c in disp.columns:
                    disp[c] = label_encoders[c].inverse_transform(disp[c])

            import streamlit.components.v1 as components
            _row_count = len(disp[final_cols])
            _table_height = min(max(400, _row_count * 45 + 60), 800)
            _html = build_core_talent_html(disp[final_cols])
            components.html(_html, height=_table_height, scrolling=True)
            st.caption("행을 클릭하면 예측사유 상세를 확인할 수 있습니다.")
        else:
            st.info("재직 중인 핵심인재가 없어 예측 리스트를 표시할 수 없습니다.")

        # -----------------------------
        # 🆕 NEW: 핵심인재 퇴직 추이 + 퇴직사유 차트
        # -----------------------------
        st.markdown("---")
        trend_col1, trend_col2 = st.columns([1.3, 1])

        with trend_col1:
            st.subheader("핵심인재 월별 퇴직 추이")
            if '퇴직일' in core_all.columns:
                core_left_df = core_all[(core_all['상태'] == 1) & (core_all['퇴직일'].notna())].copy()
                if len(core_left_df) > 0:
                    core_left_df['퇴직년월'] = core_left_df['퇴직일'].dt.to_period('M')
                    c_monthly = core_left_df.groupby('퇴직년월').size().reset_index()
                    c_monthly.columns = ['년월', '퇴직자 수']
                    c_monthly['월라벨'] = c_monthly['년월'].astype(str)

                    c_max = int(c_monthly['퇴직자 수'].max()) if len(c_monthly) > 0 else 0
                    c_ymax = c_max * 1.25 + 0.5 if c_max > 0 else 1
                    fig_ctrend = go.Figure()
                    fig_ctrend.add_bar(
                        x=c_monthly['월라벨'], y=c_monthly['퇴직자 수'],
                        name='퇴직자 수', marker_color=COLORS['primary'],
                        text=c_monthly['퇴직자 수'], textposition='outside', cliponaxis=False
                    )
                    fig_ctrend.add_trace(go.Scatter(
                        x=c_monthly['월라벨'], y=c_monthly['퇴직자 수'],
                        mode='lines+markers', name='추세선',
                        line=dict(color=COLORS['secondary'], width=3)
                    ))
                    fig_ctrend.update_layout(xaxis_title='월', yaxis_title='명', height=320)
                    fig_ctrend.update_yaxes(range=[0, c_ymax])
                    st.plotly_chart(set_font(fig_ctrend), use_container_width=True)
                else:
                    st.info("핵심인재 퇴직일 데이터가 비어 있어 추이를 표시할 수 없습니다.")
            else:
                st.info("'퇴직일' 컬럼이 없어 핵심인재 월별 퇴직 추이를 표시할 수 없습니다.")

        with trend_col2:
            st.subheader("핵심인재 퇴직사유")
            if '퇴직사유' in core_all.columns:
                c_reason_df = core_all[core_all['상태'] == 1]
                c_vc = c_reason_df['퇴직사유'].astype(str).str.strip()
                c_vc = c_vc[~c_vc.isin(['', 'nan', 'NaN', 'None', '미기재'])]
                c_counts = c_vc.value_counts(dropna=True)
                if c_counts.sum() > 0:
                    c_top_n = 8
                    c_top = c_counts.head(c_top_n)
                    c_others = c_counts.iloc[c_top_n:].sum()
                    c_labels = list(c_top.index)
                    c_values = list(c_top.values)
                    if c_others > 0:
                        c_labels.append('기타')
                        c_values.append(c_others)
                    c_pie_df = pd.DataFrame({'퇴직사유': c_labels, '건수': c_values})
                    fig_creason = px.pie(c_pie_df, names='퇴직사유', values='건수', hole=0.45,
                                        color_discrete_sequence=COLORS['sequence'])
                    fig_creason.update_traces(textposition='inside', textinfo='percent+label')
                    fig_creason.update_layout(height=320, showlegend=True)
                    st.plotly_chart(set_font(fig_creason), use_container_width=True)
                else:
                    st.info("핵심인재 퇴직 사유 데이터가 없습니다.")
            else:
                st.info("'퇴직사유' 컬럼이 없어 퇴직사유 차트를 표시할 수 없습니다.")

        # -----------------------------
        # 🔹 6) 핵심인재 퇴직률 분포 — 핵심인재 전체 기준
        # -----------------------------
        st.divider()
        st.subheader("핵심인재 퇴직률 분포")

        core_vars = [
            c for c in core_all.columns
            if c not in ['사원번호','이름','상태','퇴직일','퇴직사유','퇴직후이직처']
            and core_all[c].nunique() > 1
        ]
        
        for var in core_vars:
            # 범주형 (라벨 인코딩 포함)
            if var in label_encoders:
                df_plot = core_all.copy()
                df_plot[var+"_name"] = label_encoders[var].inverse_transform(core_all[var])
                by_var = df_plot.groupby(var+"_name")['상태'].mean()*100
                x_axis = by_var.index

                fig = px.bar(
                    x=x_axis, y=by_var.values, 
                    color=by_var.values,
                    color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                    title=f"핵심인재 {var}별 퇴직률(%)",
                    labels={'x':var,'y':'퇴직률(%)'}
                )
                fig.update_layout(height=260, showlegend=False)
                st.plotly_chart(set_font(fig), use_container_width=True)

                max_grp = by_var.idxmax()
                rate = float(by_var.max())
                n = int((df_plot[var+"_name"] == max_grp).sum())
                share = n / len(core_all) * 100 if len(core_all) > 0 else 0
                render_explanation(
                    var,
                    f"'{max_grp}'",
                    rate,
                    all_rate,
                    n=n,
                    share=share,
                    action="핵심 리텐션 집중관리",
                    explain_mode="헤드라인"
                )

            # 숫자형 → 구간 버킷 나눠서 퇴직률
            else:
                try:
                    bins = bucketize_numeric(core_all[var], bins="quartile")
                    df_tmp = pd.DataFrame({var: core_all[var], 'bin': bins, '상태': core_all['상태']})
                    by_var = df_tmp.groupby('bin')['상태'].mean()*100
                    nice_labels = [humanize_interval_label(var, b) for b in by_var.index]

                    fig = px.bar(
                        x=nice_labels, y=by_var.values, 
                        color=by_var.values,
                        color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                        title=f"핵심인재 {var}별 퇴직률(%)",
                        labels={'x':'구간','y':'퇴직률(%)'}
                    )
                    fig.update_layout(height=260, showlegend=False)
                    st.plotly_chart(set_font(fig), use_container_width=True)

                    idxmax = by_var.idxmax()
                    rate = float(by_var.max())
                    bucket_label = humanize_interval_label(var, idxmax)
                    n = int((df_tmp['bin'] == idxmax).sum())
                    share = n / len(df_tmp) * 100 if len(df_tmp) > 0 else 0
                    render_explanation(
                        var,
                        f"{bucket_label}",
                        rate,
                        all_rate,
                        n=n,
                        share=share,
                        action="핵심 리텐션 집중관리",
                        explain_mode="헤드라인"
                    )
                except Exception:
                    st.info(f"{var}: 버킷팅/집계가 어려워 분포를 생략합니다.")
# =========================
# 3) 개인별 현황
# =========================
if menu == "개인별 현황":
    add_pdf_button()
    st.title("직원 개별 퇴직 예측")
    # 전체 예측 확률 — 한 번만 계산 후 재사용
    _all_proba_ind = model.predict_proba(df[X.columns])[:, 1]

    # -------------------------
    # 직원 검색 (성명 / 사원번호)
    # -------------------------
    search_mode = st.radio("검색 방식 선택", ["성명", "사원번호"], horizontal=True)
    emp_row = None

    if search_mode == "사원번호":
        if '사원번호' not in df.columns:
            st.error("'사원번호' 컬럼이 없습니다.")
        else:
            id_min = int(pd.to_numeric(df['사원번호'], errors='coerce').min())
            id_max = int(pd.to_numeric(df['사원번호'], errors='coerce').max())
            emp_id = st.number_input("사원번호 입력", min_value=id_min, max_value=id_max, step=1)
            if emp_id in pd.to_numeric(df['사원번호'], errors='coerce').values:
                emp_row = df[pd.to_numeric(df['사원번호'], errors='coerce') == emp_id]
            else:
                st.info("사원번호를 입력하면 예측 결과가 나타납니다.")
    else:
        if '이름' not in df.columns:
            st.error("'이름' 컬럼이 없습니다.")
        else:
            name_input = st.text_input("성명을 입력하세요")
            matched = df[df['이름'] == name_input] if name_input else pd.DataFrame()
            if len(matched) == 1:
                emp_row = matched
            elif len(matched) > 1:
                st.warning(f"동명이인 {len(matched)}명 존재: 사원번호를 꼭 확인하세요!")
                cols_to_show = [c for c in ['사원번호','이름','직무','소속조직','팀','직책'] if c in matched.columns]
                matched_disp = matched.copy()
                for c in ['직무','소속조직','팀','직책']:
                    if c in label_encoders and c in matched_disp.columns:
                        matched_disp[c] = label_encoders[c].inverse_transform(matched_disp[c])
                show_table_centered(matched_disp[cols_to_show])
                emp_id2 = st.number_input(
                    "해당 사원번호를 입력하세요",
                    min_value=int(pd.to_numeric(matched['사원번호'], errors='coerce').min()),
                    max_value=int(pd.to_numeric(matched['사원번호'], errors='coerce').max()),
                    step=1
                )
                if emp_id2 in pd.to_numeric(matched['사원번호'], errors='coerce').values:
                    emp_row = matched[pd.to_numeric(matched['사원번호'], errors='coerce') == emp_id2]
            elif name_input:
                st.info("일치하는 이름이 없습니다. (정확하게 입력하세요)")

    # -------------------------------------------------
    # 예측 및 개별 리포트
    # -------------------------------------------------
    if emp_row is not None and not emp_row.empty:
        emp_X = emp_row[X.columns]
        pred_prob = float(model.predict_proba(emp_X)[0][1])
        # =========================
        # 예측 퇴직 확률 시각화 (게이지 + 분포)
        # =========================
        gauge_col, hist_col = st.columns(2)

        with gauge_col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_prob * 100,
                title={'text': f"{emp_row.get('이름', pd.Series(['-'])).iloc[0]} 퇴직 예측 확률"},
                number={'suffix': '%', 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': '#F4A7A7' if pred_prob >= 0.7 else '#FDE68A' if pred_prob >= 0.3 else '#48C0D8'},
                    'steps': [
                        {'range': [0, 30], 'color': '#E8F8FB'},
                        {'range': [30, 70], 'color': '#FFFDF0'},
                        {'range': [70, 100], 'color': '#FDF3F3'}
                    ],
                    'threshold': {
                        'line': {'color': '#1F2937', 'width': 3},
                        'thickness': 0.75,
                        'value': pred_prob * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=60, b=0, l=30, r=30))
            st.plotly_chart(set_font(fig_gauge), use_container_width=True)

        with hist_col:
            st.markdown("<br>", unsafe_allow_html=True)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=_all_proba_ind * 100, nbinsx=30,
                marker_color=COLORS['primary'], opacity=0.7,
                name='전체 직원 분포'
            ))
            _vline_color = '#F4A7A7' if pred_prob >= 0.7 else '#FDE68A' if pred_prob >= 0.3 else '#48C0D8'
            fig_hist.add_vline(
                x=pred_prob * 100,
                line_dash="dash", line_color=_vline_color, line_width=3,
                annotation_text=f"이 직원: {pred_prob*100:.1f}%",
                annotation_position="top",
                annotation_font_color=_vline_color
            )
            fig_hist.update_layout(
                xaxis_title='퇴직 예측 확률(%)', yaxis_title='직원 수',
                height=260, showlegend=False,
                title='전체 직원 대비 위치'
            )
            st.plotly_chart(set_font(fig_hist), use_container_width=True)

        # 위험등급 배지
        if pred_prob >= 0.70:
            badge_color, badge_text = '#F4A7A7', '🟠 고위험'
        elif pred_prob >= 0.30:
            badge_color, badge_text = '#FDE68A', '🟡 중위험'
        else:
            badge_color, badge_text = '#48C0D8', '🔵 저위험'

        st.markdown(f"""
        <div style="background-color: {badge_color}; padding: 12px 15px; border-radius: 8px; margin-bottom: 20px; text-align: center;">
            <span style="color: #334155; font-size: 16px; font-weight: 600;">
                사원번호 {emp_row.get('사원번호', pd.Series(['-'])).iloc[0]} /
                {emp_row.get('이름', pd.Series(['-'])).iloc[0]} — {badge_text} (퇴직 확률: {pred_prob*100:.1f}%)
            </span>
        </div>
        """, unsafe_allow_html=True)

        # =========================
        # 퇴직 예측 주요 요인 분석
        # =========================
        st.subheader("퇴직 예측 주요 요인")
        _reason_rows = []
        for _rf in top_features[:6]:
            if _rf not in X.columns:
                continue
            _emp_v = emp_row[_rf].iloc[0]
            if _rf in label_encoders:
                try:
                    _vl = label_encoders[_rf].inverse_transform([int(_emp_v)])[0]
                except Exception:
                    _vl = str(_emp_v)
                _grp = df.groupby(_rf)['상태'].mean()
                if _emp_v in _grp.index:
                    _gr = _grp[_emp_v] * 100
                    _ov = df['상태'].mean() * 100
                    _imp = '<span style="background:#F4A7A7;color:#7a2a2a;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">위험 증가</span>' if _gr > _ov * 1.1 else ('<span style="background:#D1FAE5;color:#065f46;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">위험 감소</span>' if _gr < _ov * 0.9 else '<span style="background:#E2E8F0;color:#475569;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">평균 수준</span>')
                    _reason_rows.append({'변수': _rf, '개인 값': str(_vl), '해당그룹 퇴직률': f"{_gr:.1f}%", '전체 평균 퇴직률': f"{_ov:.1f}%", '영향': _imp})
            else:
                _avg = float(df[_rf].mean())
                if _avg != 0:
                    _ratio = _emp_v / _avg
                    _sal = ['기본급','연봉','월급','급여']
                    _yr = ['근무연수','승진후경과연수']
                    _age = ['나이','연령']
                    _u = "만원" if any(k in _rf for k in _sal) else ("년" if any(k in _rf for k in _yr) else ("세" if any(k in _rf for k in _age) else ""))
                    _vs = f"{_emp_v:,.0f}{_u}" if _u else f"{_emp_v:,.1f}"
                    _as = f"{_avg:,.0f}{_u}" if _u else f"{_avg:,.1f}"
                    _imp = '<span style="background:#F4A7A7;color:#7a2a2a;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">위험 증가</span>' if _ratio < 0.7 or _ratio > 1.3 else '<span style="background:#E2E8F0;color:#475569;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">평균 수준</span>'
                    _reason_rows.append({'변수': _rf, '개인 값': _vs, '전체 평균': _as, '평균 대비': f"{_ratio*100:.0f}%", '영향': _imp})
        if _reason_rows:
            show_table_centered(pd.DataFrame(_reason_rows))
        else:
            st.info("예측 요인 분석 데이터가 부족합니다.")

        # =========================
        # 1) 동료 그룹 대비 퇴직 위험 비교
        # =========================
        st.subheader("구분 별 퇴직 위험 비교")

        emp_prob = pred_prob  # 개인 예측 확률
        overall_mean_prob = float(_all_proba_ind.mean())

        peer_cols = ['소속조직', '팀', '직무', '직책']
        peer_rows = []

        for col in peer_cols:
            if col in df.columns:
                try:
                    emp_val = emp_row[col].iloc[0]
                    peer_df = df[df[col] == emp_val]
                    if len(peer_df) > 0:
                        peer_proba = model.predict_proba(peer_df[X.columns])[:, 1].mean()

                        # 라벨 복원(있으면)
                        if col in label_encoders:
                            try:
                                display_val = label_encoders[col].inverse_transform([int(emp_val)])[0]
                            except Exception:
                                display_val = str(emp_val)
                        else:
                            display_val = str(emp_val)

                        diff = int(round((peer_proba - overall_mean_prob) * 100))  # %p, 정수

                        peer_rows.append({
                            '구분': col,
                            '개인값': display_val,
                            '동일그룹 인원수': len(peer_df),
                            '동일그룹 평균 퇴직위험(%)': f"{round(peer_proba * 100):d}%",
                            '전체 평균 대비 차이(p)': diff  # 예: -10, +5
                        })
                except Exception:
                    pass

        peer_df_disp = pd.DataFrame(peer_rows)
        if len(peer_df_disp) > 0:
            show_table_centered(peer_df_disp)
        else:
            st.info("동료 그룹(소속조직/팀/직무/직책) 기준 비교를 할 수 있는 데이터가 부족합니다.")

        # =========================
        # 2) 팀/소속조직 내 퇴직위험 순위 (바 차트)
        # =========================
        st.subheader("조직 내 퇴직위험 순위")

        df_with_proba = df.copy()
        df_with_proba['퇴직예측확률'] = _all_proba_ind

        _rank_has_chart = False
        for col in ['팀', '소속조직']:
            if col in df_with_proba.columns:
                try:
                    emp_val = emp_row[col].iloc[0]
                    same_grp = df_with_proba[df_with_proba[col] == emp_val].copy()
                    if len(same_grp) > 1:
                        same_grp = same_grp.sort_values('퇴직예측확률', ascending=False)
                        same_grp['rank'] = range(1, len(same_grp) + 1)

                        if '사원번호' in df_with_proba.columns and '사원번호' in emp_row.columns:
                            emp_id_val = emp_row['사원번호'].iloc[0]
                            my_idx = same_grp[same_grp['사원번호'] == emp_id_val].index
                        else:
                            my_idx = same_grp.index.intersection(emp_row.index)

                        if col in label_encoders:
                            try:
                                display_val = label_encoders[col].inverse_transform([int(emp_val)])[0]
                            except Exception:
                                display_val = str(emp_val)
                        else:
                            display_val = str(emp_val)

                        # 바 차트용 이름 및 색상
                        bar_names = []
                        for _, _r in same_grp.iterrows():
                            if '이름' in _r.index:
                                bar_names.append(str(_r['이름']))
                            elif '사원번호' in _r.index:
                                bar_names.append(str(int(_r['사원번호'])))
                            else:
                                bar_names.append(str(_r.name))
                        bar_colors = ['#EF4444' if idx in my_idx else COLORS['primary'] for idx in same_grp.index]

                        my_rank = int(same_grp.loc[my_idx, 'rank'].iloc[0]) if len(my_idx) > 0 else 0
                        n_grp = len(same_grp)

                        fig_rank = go.Figure(go.Bar(
                            x=bar_names, y=same_grp['퇴직예측확률'].values * 100,
                            marker_color=bar_colors,
                            text=[f"{v*100:.1f}%" for v in same_grp['퇴직예측확률'].values],
                            textposition='outside'
                        ))
                        fig_rank.update_layout(
                            title=f"{col} '{display_val}' 내 퇴직위험 순위 ({my_rank}위/{n_grp}명)",
                            xaxis_title='', yaxis_title='퇴직위험(%)', height=300
                        )
                        st.plotly_chart(set_font(fig_rank), use_container_width=True)
                        st.caption(f"🔴 빨간색 바가 해당 직원입니다. {col} '{display_val}' 내 {n_grp}명 중 {my_rank}위 (상위 {my_rank/n_grp*100:.1f}%)")
                        _rank_has_chart = True
                except Exception:
                    pass

        if not _rank_has_chart:
            st.info("팀/소속조직 기준으로 순위를 계산할 수 있는 데이터가 부족합니다.")

        # =========================
        # 3) 상위 변수별 프로필 비교 (레이더 차트 + 상세 테이블)
        # =========================
        if len(top_features) > 0:
            st.subheader("상위 변수별 프로필 비교")

            _r_vals, _r_avgs, _r_labels = [], [], []
            _num_rows = []   # 수치형 변수 상세 테이블용
            _cat_rows = []   # 범주형 변수 상세 테이블용

            for var in top_features:
                if var in label_encoders:
                    # 범주형 변수 처리
                    raw_val = emp_row[var].iloc[0]
                    _el = get_label(raw_val, var, label_encoders)
                    # 해당 범주의 퇴직률 계산
                    same_cat = df[df[var] == raw_val]
                    if len(same_cat) > 0 and '상태' in df.columns:
                        cat_turnover = same_cat['상태'].mean() * 100
                        overall_turnover = df['상태'].mean() * 100
                        diff_turnover = cat_turnover - overall_turnover
                        if diff_turnover > 0:
                            risk_tag = f"🟠 전체 대비 +{diff_turnover:.1f}%p 높음"
                        elif diff_turnover < -1:
                            risk_tag = f"🔵 전체 대비 {diff_turnover:.1f}%p 낮음"
                        else:
                            risk_tag = "🩶 전체 평균과 유사"
                        _cat_rows.append({
                            '변수': var,
                            '해당 직원': str(_el),
                            '해당 범주 퇴직률': f"{cat_turnover:.1f}%",
                            '전체 퇴직률': f"{overall_turnover:.1f}%",
                            '위험 수준': risk_tag
                        })
                    else:
                        _cat_rows.append({
                            '변수': var,
                            '해당 직원': str(_el),
                            '해당 범주 퇴직률': '-',
                            '전체 퇴직률': '-',
                            '위험 수준': '-'
                        })
                else:
                    # 수치형 변수 처리
                    _ev = float(emp_row[var].iloc[0])
                    _av = float(df[var].mean())
                    _std = float(df[var].std()) if df[var].std() > 0 else 1
                    _pct = int(round((df[var] <= _ev).mean() * 100))

                    if _av != 0:
                        ratio = _ev / _av * 100
                        _r_vals.append(ratio)
                        _r_avgs.append(100)
                        _r_labels.append(var)
                        diff_pct = ratio - 100
                        if diff_pct > 10:
                            direction = f"▲ 평균 대비 +{diff_pct:.0f}%"
                        elif diff_pct < -10:
                            direction = f"▼ 평균 대비 {diff_pct:.0f}%"
                        else:
                            direction = f"● 평균과 유사 ({diff_pct:+.0f}%)"
                    else:
                        direction = "-"

                    _num_rows.append({
                        '변수': var,
                        '개인값': round(_ev, 1),
                        '전체 평균': round(_av, 1),
                        '비교': direction,
                        '분위수': f"하위 {_pct}%"
                    })

            # --- 레이더 차트 ---
            if _r_labels:
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=_r_vals + [_r_vals[0]], theta=_r_labels + [_r_labels[0]],
                    fill='toself', name='해당 직원',
                    fillcolor='rgba(85, 72, 199, 0.2)',
                    line=dict(color=COLORS['secondary'], width=2)
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=_r_avgs + [_r_avgs[0]], theta=_r_labels + [_r_labels[0]],
                    fill='toself', name='전체 평균 (100%)',
                    fillcolor='rgba(72, 192, 216, 0.15)',
                    line=dict(color=COLORS['primary'], width=2, dash='dot')
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, max(max(_r_vals), 150) * 1.1])),
                    showlegend=True, height=400,
                    title='상위 변수별 프로필 (전체 평균=100% 기준)'
                )
                st.plotly_chart(set_font(fig_radar), use_container_width=True)
                st.caption("전체 평균을 100%로 놓았을 때 해당 직원의 상대적 수준입니다. 안쪽이면 평균 이하, 바깥이면 평균 이상.")

            # --- 수치형 변수 상세 비교 테이블 ---
            if _num_rows:
                st.markdown("##### 수치형 변수 상세 비교")
                show_table_centered(pd.DataFrame(_num_rows))

                # 인사이트 코멘트 자동 생성
                _insights = []
                for row in _num_rows:
                    var_name = row['변수']
                    comp = row['비교']
                    if '▲' in comp:
                        _insights.append(f"- **{var_name}**: 전체 평균보다 **높은 수준**입니다. ({row['개인값']} vs 평균 {row['전체 평균']}, {row['분위수']})")
                    elif '▼' in comp:
                        _insights.append(f"- **{var_name}**: 전체 평균보다 **낮은 수준**입니다. ({row['개인값']} vs 평균 {row['전체 평균']}, {row['분위수']})")
                    else:
                        _insights.append(f"- **{var_name}**: 전체 평균과 **유사한 수준**입니다. ({row['개인값']} vs 평균 {row['전체 평균']}, {row['분위수']})")
                if _insights:
                    with st.expander("수치형 변수 해석 보기", expanded=True):
                        st.markdown("\n".join(_insights))

            # --- 범주형 변수 상세 비교 테이블 ---
            if _cat_rows:
                st.markdown("##### 범주형 변수 상세 비교")
                show_table_centered(pd.DataFrame(_cat_rows))

                # 범주형 인사이트
                _cat_insights = []
                for row in _cat_rows:
                    var_name = row['변수']
                    val = row['해당 직원']
                    risk = row['위험 수준']
                    if '🟠' in risk:
                        _cat_insights.append(f"- **{var_name}** = '{val}' → 이 그룹은 퇴직률이 전체 평균보다 **높아** 주의가 필요합니다.")
                    elif '🔵' in risk:
                        _cat_insights.append(f"- **{var_name}** = '{val}' → 이 그룹은 퇴직률이 전체 평균보다 **낮은** 편입니다.")
                    else:
                        _cat_insights.append(f"- **{var_name}** = '{val}' → 이 그룹의 퇴직률은 전체 평균과 **유사**합니다.")
                if _cat_insights:
                    with st.expander("범주형 변수 해석 보기", expanded=True):
                        st.markdown("\n".join(_cat_insights))

        # =========================
        # 4) 주요 숫자 변수에서의 위치(분위수)
        # =========================
        st.subheader("주요 숫자 변수별 비교")

        num_candidates = ['근무연수', '나이', '기본급', '입사전이직횟수', '보유역량']
        rows_num = []

        for col in num_candidates:
            if col in df.columns:
                try:
                    series = pd.to_numeric(df[col], errors='coerce')
                    emp_val = float(pd.to_numeric(emp_row[col], errors='coerce').iloc[0])

                    # 전체 평균/개인값: 정수
                    mean_val = int(round(series.mean()))
                    emp_val_int = int(round(emp_val))

                    # 분위수(%): 정수 + % 기호
                    pct = int(round((series <= emp_val).mean() * 100))

                    rows_num.append({
                        '변수': col,
                        '개인값': emp_val_int,
                        '전체 평균': mean_val,
                        '분위수(%)': f"{pct}%"
                    })
                except Exception:
                    pass

        if rows_num:
            num_df = pd.DataFrame(rows_num)
            show_table_centered(num_df)
            st.caption("※ 분위수(%)는 '이 값 이하인 사람이 전체에서 차지하는 비율'입니다. 값이 낮을수록 하위, 높을수록 상위 위치를 의미합니다.")
        else:
            st.info("근무연수/급여 등 숫자형 변수 기준 분위수 정보를 계산할 수 없습니다.")

        # =========================
        # 5) 유사 퇴직자 프로필 매칭
        # =========================
        st.subheader("유사 퇴직자 프로필")
        _departed = df[df['상태'] == 1].copy()
        if len(_departed) > 0 and len(X.columns) > 0:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                _emp_feat = emp_row[X.columns].values.reshape(1, -1).astype(float)
                _dep_feat = _departed[X.columns].values.astype(float)
                _sim = cosine_similarity(_emp_feat, _dep_feat)[0]
                _departed['유사도'] = _sim
                _top_sim = _departed.sort_values('유사도', ascending=False).head(3)
                _sim_cols = ['사원번호','이름','소속조직','팀','직무','직책','퇴직사유']
                _sim_show = [c for c in _sim_cols if c in _top_sim.columns] + ['유사도']
                _sim_disp = _top_sim.copy()
                for _sc in ['소속조직','팀','직무','직책']:
                    if _sc in label_encoders and _sc in _sim_disp.columns:
                        _sim_disp[_sc] = label_encoders[_sc].inverse_transform(_sim_disp[_sc])
                _sim_disp['유사도'] = _sim_disp['유사도'].apply(lambda x: f"{x*100:.1f}%")
                show_table_centered(_sim_disp[_sim_show])
                st.caption("코사인 유사도 기반으로 이 직원과 가장 비슷한 조건의 퇴직자 3명입니다. 퇴직 패턴 참고용으로 활용하세요.")
            except Exception:
                st.info("유사 퇴직자를 계산할 수 없습니다.")
        else:
            st.info("퇴직자 데이터가 없어 유사 퇴직자를 표시할 수 없습니다.")

        # =========================
        # 6) 퇴직 위험 해석 및 조언 (고도화)
        # =========================
        st.subheader("퇴직 위험 해석 및 조언")
        tips = []
        for _tf in top_features[:6]:
            if _tf not in X.columns or _tf not in df.columns:
                continue
            try:
                if _tf in label_encoders:
                    _ev = emp_row[_tf].iloc[0]
                    _grp = df.groupby(_tf)['상태'].mean()
                    if _ev in _grp.index:
                        _gr = _grp[_ev]
                        _ov = df['상태'].mean()
                        if _gr > _ov * 1.3:
                            _vl = get_label(_ev, _tf, label_encoders)
                            tips.append(f"**{_tf}** '{_vl}' 그룹의 퇴직률({_gr*100:.1f}%)이 전체 평균({_ov*100:.1f}%)보다 높습니다. 해당 그룹 대상 **맞춤 리텐션 프로그램** 검토가 필요합니다.")
                else:
                    _ev = float(emp_row[_tf].iloc[0])
                    _av = float(df[_tf].mean())
                    if _av == 0:
                        continue
                    _ratio = _ev / _av
                    _sal = ['기본급','연봉','월급','급여']
                    _ot = ['연장근무','야근','초과근무']
                    _yr_s = ['근무연수']
                    _career = ['입사전이직횟수','이직횟수']
                    _promo = ['승진후경과연수']
                    if any(k in _tf for k in _sal) and _ratio < 0.85:
                        tips.append(f"**{_tf}**({_ev:,.0f}만원)이 평균({_av:,.0f}만원)보다 {(1-_ratio)*100:.0f}% 낮습니다. **보상 밴드 점검 및 시장 경쟁력 분석** 권장.")
                    elif any(k in _tf for k in _yr_s) and _ratio < 0.7:
                        tips.append(f"**{_tf}**({_ev:.0f}년)이 평균({_av:.0f}년)보다 짧아 정착 리스크. **온보딩 강화·멘토 배정·경력개발 면담** 필요.")
                    elif any(k in _tf for k in _ot) and _ratio > 1.15:
                        tips.append(f"**{_tf}** 빈도가 평균 대비 {(_ratio-1)*100:.0f}% 높아 소진(Burnout) 위험. **업무량 재조정·휴식 관리** 권장.")
                    elif any(k in _tf for k in _career) and _ratio > 1.3:
                        tips.append(f"**{_tf}**({_ev:.0f}회)이 평균({_av:.0f}회)보다 높아 이직 성향이 있습니다. **장기 인센티브·경력비전 제시** 검토.")
                    elif any(k in _tf for k in _promo) and _ratio > 1.3:
                        tips.append(f"**{_tf}**({_ev:.0f}년)이 평균({_av:.0f}년)보다 길어 승진 정체감이 우려됩니다. **승진 경로 논의·역할 확대** 필요.")
            except Exception:
                pass
        if not tips:
            tips.append("주요 위험 신호가 두드러지지 않습니다. 정기 케어와 경력개발 대화를 권장합니다.")
        for t in tips:
            st.markdown(f"""
            <div style="background-color: #48C0D8; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: white; font-size: 14px; margin: 0;">• {t}</p>
            </div>
            """, unsafe_allow_html=True)

        # =========================
        # 6) 상세 원본 데이터
        # =========================
        with st.expander("직원 상세 정보 보기"):
            row_disp = emp_row.copy()
            for c in label_encoders:
                if c in row_disp.columns:
                    try:
                        row_disp[c] = label_encoders[c].inverse_transform(row_disp[c])
                    except Exception:
                        pass
            # Transpose하여 항목명-값 형태로 표시
            detail_df = pd.DataFrame({
                '항목': row_disp.columns,
                '값': row_disp.iloc[0].values
            })
            show_table_centered(detail_df)