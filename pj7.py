# =========================
# HR Insight Dashboard — 엔트리 포인트
#
# 모듈 구성:
#   hr_styles.py      : 전역 CSS/JS, 컬러 팔레트, 차트 폰트
#   hr_data.py        : 파일 로딩, 전처리, 인코딩, 스키마 상수
#   hr_model.py       : XGBoost 학습, 성능 지표, OOF 예측 확률
#   hr_components.py  : 공용 UI 컴포넌트 (테이블/카드/다운로드 등)
#   view_overview.py  : ① 전체 현황
#   view_core.py      : ② 핵심인재 현황
#   view_individual.py: ③ 개인별 현황
# =========================
import streamlit as st
import numpy as np

# NumPy 2.0 compatibility shim
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

st.set_page_config(page_title="HR Insight Dashboard", layout="wide")

from hr_styles import inject_global_styles, inject_sidebar_toggle
from hr_data import load_and_preprocess
from hr_model import train_model
import view_overview
import view_core
import view_individual

inject_global_styles()
inject_sidebar_toggle()

# =========================
# 업로드
# =========================
st.sidebar.markdown("""
<div style="font-size: 13px; font-weight: 600; color: rgba(255,255,255,0.7); margin-bottom: 8px; letter-spacing: 1px; text-transform: uppercase; text-align: center;">
    데이터 업로드
</div>
""", unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader("회사 데이터 업로드 (csv/xlsx)", type=["csv", "xlsx"], label_visibility="collapsed")

# =========================
# 앱 시작
# =========================
if uploaded is None:
    st.info("사이드바에서 CSV 또는 Excel 파일을 업로드해주세요.")
    st.stop()

df, X, y, label_encoders, validation, load_err = load_and_preprocess(uploaded)
if load_err:
    st.error(load_err); st.stop()
if df is None:
    st.warning("데이터를 불러올 수 없습니다. 파일 형식을 확인해주세요."); st.stop()

model, metrics, feature_importance, all_proba = train_model(X, y)
if isinstance(metrics, dict) and "error" in metrics:
    st.error(metrics["error"]); st.stop()

# 상위 중요 변수 6개 사용
top_features = feature_importance.head(6).index.tolist() if feature_importance is not None and len(feature_importance) > 0 else []

# 페이지들이 공유하는 컨텍스트
ctx = {
    "df": df,
    "X": X,
    "y": y,
    "model": model,
    "metrics": metrics,
    "feature_importance": feature_importance,
    "top_features": top_features,
    "label_encoders": label_encoders,
    "all_proba": all_proba,        # 5-fold OOF 예측 확률 (df와 동일 인덱스)
    "validation": validation,
}

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

MENU_PAGES = {
    "전체 현황": view_overview,
    "핵심인재 현황": view_core,
    "개인별 현황": view_individual,
}

if "menu" not in st.session_state:
    st.session_state["menu"] = "전체 현황"
for item in MENU_PAGES:
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

# 선택된 페이지 렌더링
MENU_PAGES.get(menu, view_overview).render(ctx)
