# =========================
# 공용 UI 컴포넌트 / 표시용 유틸
# =========================
import html as _html
import io
import re

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.stats import chi2_contingency


def esc(v) -> str:
    """업로드 데이터에서 온 값을 HTML에 넣기 전 이스케이프 (HTML 인젝션 방지)"""
    return _html.escape(str(v))


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
    salary_like = ['기본급', '연봉', '급여', '월급']
    years_like = ['근무연수', '승진후경과연수']
    age_like = ['나이', '연령']
    if any(k in var for k in salary_like):
        return _fmt_range(left, right, unit="만원")
    elif any(k in var for k in years_like):
        return _fmt_range(left, right, unit="년")
    elif any(k in var for k in age_like):
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


def render_explanation(var_name, bucket_label, rate, overall, n=None, share=None, delta=None,
                       action="리텐션 정책 점검", explain_mode="헤드라인"):
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


def render_risk_group_card(var, group_label, rate, diff, n, total_employees, share,
                           accent_color="#48C0D8"):
    """'주의 그룹' 요약 카드 (전체 현황 페이지에서 범주형/숫자형 공용)"""
    if diff >= 3:
        risk_label = "고위험"
    elif diff >= 0:
        risk_label = "주의"
    else:
        risk_label = "낮은 위험"

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
            ">주의 그룹: {esc(var)} = {esc(group_label)}</span>
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
    tag_colors = {
        "red":   ("rgba(162,28,28,.12)", "#991f1f"),
        "amber": ("rgba(146,88,0,.12)",  "#7a4e00"),
        "blue":  ("rgba(20,80,150,.12)", "#0d4a8a"),
        "gray":  ("rgba(100,100,100,.1)", "#555555"),
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
                    f'background:{bg};color:{fc}">{esc(t["label"])}</span>'
                )

        incentive_td = f'<td style="padding:9px 10px;text-align:center">{esc(row.get("인센티브", ""))}</td>' if has_incentive else ""
        grade_td = f"""<td style="padding:9px 10px">
            <span style="padding:2px 7px;border-radius:4px;font-size:11px;font-weight:500;
              background:{grade_bg};color:{grade_color}">{esc(grade)}</span>
          </td>""" if has_grade else ""

        rows_html += f"""
        <tr class="main-row" onclick="toggle({i})" style="cursor:pointer">
          <td>
            <span id="chev-{i}" style="font-size:9px;color:#aaa;display:inline-block;transition:transform .2s">▶</span>
          </td>
          <td style="color:#888;white-space:nowrap">{esc(row.get("사원번호", ""))}</td>
          <td><strong style="color:#334155">{esc(row.get("이름", ""))}</strong></td>
          <td style="white-space:nowrap">{esc(row.get("소속조직", ""))}</td>
          <td style="white-space:nowrap">{esc(row.get("직책", ""))}</td>
          {grade_td}
          {incentive_td}
          <td style="font-weight:600;color:{risk_color};white-space:nowrap">{risk_text}</td>
          <td style="text-align:left">{tags_html}</td>
        </tr>
        <tr id="detail-{i}" class="detail-row" style="display:none">
          <td colspan="9" style="padding:0">
            <div style="padding:10px 12px 10px 44px;font-size:0.85rem;color:#64748B;border-bottom:1px solid #E2E8F0;text-align:left">
              <span style="font-size:0.75rem;font-weight:700;color:#94A3B8;letter-spacing:.04em;margin-right:6px">예측사유</span>{esc(reason_text)}
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


def show_core_talent_table(df):
    """접이식 핵심인재 테이블 렌더링 (행 수 기반 높이 자동 조정)"""
    row_count = len(df)
    table_height = min(max(400, row_count * 45 + 60), 800)
    components.html(build_core_talent_html(df), height=table_height, scrolling=True)


def show_table_centered(df, allow_html_cols=None):
    """
    Streamlit dataframe의 정렬 이슈를 해결하기 위해 HTML로 직접 렌더링합니다.
    allow_html_cols에 지정된 컬럼(배지 등 신뢰된 마크업)을 제외한 모든 셀은 이스케이프됩니다.
    """
    try:
        df_disp = df.fillna('-').copy()
        allow = set(allow_html_cols or [])
        for col in df_disp.columns:
            if col not in allow:
                df_disp[col] = df_disp[col].map(esc)

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


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """DataFrame을 엑셀 파일 바이트로 변환 (다운로드 버튼용)"""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


def excel_download_button(df: pd.DataFrame, filename: str, label: str = "엑셀 다운로드", key=None):
    """스타일 통일된 엑셀 다운로드 버튼"""
    try:
        data = df_to_excel_bytes(df)
        st.download_button(
            label=f"📥 {label}",
            data=data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=key,
        )
    except Exception as e:
        st.caption(f"엑셀 파일 생성 실패: {e}")
