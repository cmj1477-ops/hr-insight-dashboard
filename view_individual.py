# =========================
# 3) 개인별 현황 페이지
# =========================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import shap

from hr_styles import COLORS, set_font, add_pdf_button, RISK_HIGH, RISK_MID, risk_color
from hr_components import show_table_centered, get_label, esc


def _search_employee(df, label_encoders):
    """직원 검색 (사원번호 / 성명 부분 검색). 선택된 1명(df 1행)을 반환."""
    search_mode = st.radio("검색 방식 선택", ["사원번호", "성명"], horizontal=True)
    emp_row = None

    if search_mode == "사원번호":
        if '사원번호' not in df.columns:
            st.error("'사원번호' 컬럼이 없습니다.")
            return None
        # 문자/숫자 사번 모두 지원 (숫자 강제 변환 시 문자 사번에서 크래시 발생하므로 텍스트 검색 사용)
        emp_no_input = st.text_input("사원번호 입력", placeholder="예: 1001")
        if emp_no_input.strip():
            key = emp_no_input.strip()
            ids = df['사원번호'].astype(str).str.strip()
            matched = df[ids == key]
            if len(matched) == 0:
                # 숫자형 사번이 1001.0 처럼 저장된 경우 대비
                ids_num = pd.to_numeric(ids, errors='coerce')
                key_num = pd.to_numeric(pd.Series([key]), errors='coerce').iloc[0]
                if not pd.isna(key_num):
                    matched = df[ids_num == key_num]
            if len(matched) >= 1:
                if len(matched) > 1:
                    st.warning(f"동일 사원번호 {len(matched)}건이 있습니다. 첫 번째 행을 표시합니다. 데이터를 확인하세요.")
                emp_row = matched.head(1)
            else:
                st.info("일치하는 사원번호가 없습니다.")
        else:
            st.info("사원번호를 입력하면 예측 결과가 나타납니다.")
    else:
        if '이름' not in df.columns:
            st.error("'이름' 컬럼이 없습니다.")
            return None
        name_input = st.text_input("성명을 입력하세요 (일부만 입력해도 검색됩니다)")
        if name_input.strip():
            names = df['이름'].astype(str)
            matched = df[names.str.contains(name_input.strip(), na=False, regex=False)]
            if len(matched) == 1:
                emp_row = matched
            elif len(matched) > 1:
                st.warning(f"검색 결과 {len(matched)}명 — 동명이인이 있다면 사원번호를 꼭 확인하세요!")
                cols_to_show = [c for c in ['사원번호', '이름', '직무', '소속조직', '팀', '직책'] if c in matched.columns]
                matched_disp = matched.copy()
                for c in ['직무', '소속조직', '팀', '직책']:
                    if c in label_encoders and c in matched_disp.columns:
                        matched_disp[c] = label_encoders[c].inverse_transform(matched_disp[c])
                show_table_centered(matched_disp[cols_to_show].head(20))

                def _fmt_choice(i):
                    parts = []
                    for c in ['사원번호', '이름', '소속조직', '직책']:
                        if c in matched_disp.columns:
                            parts.append(str(matched_disp.loc[i, c]))
                    return " · ".join(parts)

                sel = st.selectbox(
                    "조회할 직원을 선택하세요",
                    options=list(matched.index),
                    format_func=_fmt_choice,
                    index=None,
                    placeholder="직원 선택",
                )
                if sel is not None:
                    emp_row = matched.loc[[sel]]
            else:
                st.info("일치하는 이름이 없습니다.")
    return emp_row


def _render_shap_factors(ctx, emp_row):
    """🆕 SHAP 기반 개인별 예측 요인 분석 — 모델이 실제로 반영한 기여도"""
    df = ctx['df']; X = ctx['X']
    model = ctx['model']; label_encoders = ctx['label_encoders']

    st.subheader("AI 모델이 본 개인별 예측 요인 (SHAP)")
    try:
        idx = emp_row.index[0]
        emp_X = X.loc[[idx]]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(emp_X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv).reshape(-1)

        shap_s = pd.Series(sv, index=X.columns)
        top = shap_s.reindex(shap_s.abs().sort_values(ascending=False).index).head(8)
        top = top.iloc[::-1]  # 가로 막대는 아래→위 순서

        bar_labels = []
        for var in top.index:
            raw_val = emp_row[var].iloc[0] if var in emp_row.columns else emp_X[var].iloc[0]
            val_label = get_label(raw_val, var, label_encoders)
            if isinstance(val_label, float):
                val_label = f"{val_label:,.1f}"
            bar_labels.append(f"{var} ({val_label})")

        bar_colors = [COLORS['warning'] if v > 0 else COLORS['primary'] for v in top.values]
        fig_shap = go.Figure(go.Bar(
            x=top.values,
            y=bar_labels,
            orientation='h',
            marker_color=bar_colors,
            text=[f"{v:+.2f}" for v in top.values],
            textposition='outside',
            cliponaxis=False,
        ))
        fig_shap.add_vline(x=0, line_dash="dot", line_color="#9CA3AF")
        _amax = float(np.abs(top.values).max()) if len(top) > 0 else 1.0
        fig_shap.update_layout(
            title="이 직원의 예측에 대한 변수별 기여도 (SHAP)",
            xaxis_title="← 퇴직 확률을 낮춤        |        퇴직 확률을 높임 →",
            yaxis_title="",
            height=360,
            showlegend=False,
        )
        fig_shap.update_xaxes(range=[-_amax * 1.35, _amax * 1.35], zeroline=False)
        st.plotly_chart(set_font(fig_shap), use_container_width=True)
        st.markdown("""
<div style="background:#F8FAFC; border:1px solid #E5E7EB; border-radius:8px; padding:14px 18px; font-size:13.5px; color:#475569; line-height:1.75;">
<b style="color:#334155;">SHAP 기여도란?</b><br>
그룹 평균 비교와 달리, <b>학습된 모델이 이 직원 한 명의 예측을 계산할 때 각 변수를 실제로 얼마나 반영했는지</b>를 분해한 값입니다.
<b>오른쪽(주황)</b> 막대는 이 직원의 해당 조건이 퇴직 예측을 끌어올린 요인,
<b>왼쪽(청록)</b> 막대는 끌어내린 요인입니다. 막대가 길수록 영향이 큽니다.
</div>
        """, unsafe_allow_html=True)
    except Exception:
        st.info("SHAP 기여도를 계산할 수 없습니다. (모델/데이터 구조 확인 필요)")


def _render_what_if(ctx, emp_row):
    """🆕 What-if 시뮬레이션 — 조건 변경 시 퇴직 확률 변화"""
    df = ctx['df']; X = ctx['X']
    model = ctx['model']; label_encoders = ctx['label_encoders']
    top_features = ctx['top_features']

    st.subheader("What-if 시뮬레이션")
    st.markdown("""
    <div style="background-color: #F8FAFC; padding: 12px 16px; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 12px;">
        <p style="margin: 0; font-size: 13px; color: #64748B; line-height: 1.6;">
        주요 변수의 값을 바꿔보면 <b>리텐션 조치(보상 조정, 승진, 근무형태 변경 등)의 기대 효과</b>를 확인할 수 있습니다.<br>
        조건을 조정한 뒤 <b>시뮬레이션 실행</b>을 누르세요.
        </p>
    </div>
    """, unsafe_allow_html=True)

    idx = emp_row.index[0]
    x_base = X.loc[[idx]].astype(float)

    sim_features = [f for f in top_features[:6] if f in X.columns]
    if not sim_features:
        st.info("시뮬레이션할 변수가 없습니다.")
        return

    new_vals = {}
    with st.form("what_if_form"):
        form_cols = st.columns(2)
        for i, var in enumerate(sim_features):
            with form_cols[i % 2]:
                cur = x_base[var].iloc[0]
                if var in label_encoders:
                    classes = list(label_encoders[var].classes_)
                    cur_idx = int(cur) if 0 <= int(cur) < len(classes) else 0
                    choice = st.selectbox(f"{var}", options=classes, index=cur_idx, key=f"wi_{var}")
                    new_vals[var] = int(classes.index(choice))
                else:
                    col_series = pd.to_numeric(df[var], errors='coerce').dropna()
                    v_min, v_max = float(col_series.min()), float(col_series.max())
                    if v_min >= v_max:
                        continue
                    is_int = bool(np.allclose(col_series % 1, 0))
                    if is_int:
                        new_vals[var] = st.slider(
                            f"{var}", min_value=int(v_min), max_value=int(v_max),
                            value=int(round(cur)), key=f"wi_{var}")
                    else:
                        new_vals[var] = st.slider(
                            f"{var}", min_value=v_min, max_value=v_max,
                            value=float(cur), key=f"wi_{var}")
        submitted = st.form_submit_button("▶ 시뮬레이션 실행", use_container_width=True)

    if submitted:
        try:
            x_new = x_base.copy()
            for var, val in new_vals.items():
                x_new[var] = float(val)
            base_p = float(model.predict_proba(x_base)[0][1])
            new_p = float(model.predict_proba(x_new)[0][1])
            delta = (new_p - base_p) * 100

            r1, r2, r3 = st.columns(3)
            r1.metric("현재 조건 (모델 기준)", f"{base_p*100:.1f}%")
            r2.metric("시뮬레이션 결과", f"{new_p*100:.1f}%",
                      delta=f"{delta:+.1f}%p", delta_color="inverse")
            if delta < -1:
                verdict, v_color = "위험 감소 효과가 있습니다", risk_color(0.0)
            elif delta > 1:
                verdict, v_color = "위험이 오히려 증가합니다", COLORS['warning']
            else:
                verdict, v_color = "변화가 크지 않습니다", "#94A3B8"
            with r3:
                st.markdown(f"""
                <div style="background-color: {v_color}; padding: 14px 15px; border-radius: 8px; text-align: center; margin-top: 4px;">
                    <span style="color: #FFFFFF; font-size: 14px; font-weight: 600;">{verdict}</span>
                </div>
                """, unsafe_allow_html=True)
            st.caption("※ 시뮬레이션은 학습된 모델 자체의 예측 기준이므로, 상단에 표시된 교차검증 기반 확률과 수치가 다소 다를 수 있습니다. 변화폭(%p)을 중심으로 해석하세요.")
        except Exception as e:
            st.info(f"시뮬레이션 계산에 실패했습니다: {e}")


def render(ctx):
    df = ctx['df']
    label_encoders = ctx['label_encoders']

    add_pdf_button()
    st.title("개인별 퇴직 예측")

    emp_row = _search_employee(df, label_encoders)

    if emp_row is not None and not emp_row.empty:
        render_report(ctx, emp_row)


def render_report(ctx, emp_row):
    df = ctx['df']; X = ctx['X']
    model = ctx['model']
    top_features = ctx['top_features']
    label_encoders = ctx['label_encoders']
    all_proba = ctx['all_proba']          # OOF(교차검증) 예측 확률 — df와 동일 인덱스

    emp_idx = emp_row.index[0]
    pred_prob = float(all_proba.loc[emp_idx])

    # 재직자 기준 분포 (퇴직자를 포함하면 분포가 오른쪽으로 쏠려 위치 비교가 왜곡됨)
    _active_mask = df['상태'] == 0
    _active_proba = all_proba[_active_mask]

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
                'bar': {'color': risk_color(pred_prob)},
                'steps': [
                    {'range': [0, 30], 'color': '#F0FBFC'},
                    {'range': [30, 70], 'color': '#FEF7E0'},
                    {'range': [70, 100], 'color': '#FCE7C8'}
                ],
                'threshold': {
                    'line': {'color': '#334155', 'width': 3},
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
            x=_active_proba.values * 100, nbinsx=30,
            marker_color=COLORS['primary'], opacity=0.7,
            name='재직자 분포'
        ))
        _vline_color = risk_color(pred_prob)
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
            title='재직자 전체 대비 위치'
        )
        st.plotly_chart(set_font(fig_hist), use_container_width=True)

    # 위험등급 배지
    if pred_prob >= RISK_HIGH:
        badge_color, badge_text = '#F59E0B', '고위험'
    elif pred_prob >= RISK_MID:
        badge_color, badge_text = '#7DD3FC', '중위험'
    else:
        badge_color, badge_text = '#48C0D8', '저위험'

    st.markdown(f"""
    <div style="background-color: {badge_color}; padding: 12px 15px; border-radius: 8px; margin-bottom: 20px; text-align: center;">
        <span style="color: #334155; font-size: 16px; font-weight: 600;">
            사원번호 {esc(emp_row.get('사원번호', pd.Series(['-'])).iloc[0])} /
            {esc(emp_row.get('이름', pd.Series(['-'])).iloc[0])} — {badge_text} (퇴직 확률: {pred_prob*100:.1f}%)
        </span>
    </div>
    """, unsafe_allow_html=True)

    if '상태' in emp_row.columns and int(emp_row['상태'].iloc[0]) == 1:
        st.caption("※ 이 직원은 이미 퇴직한 인원입니다. 표시된 확률은 재직 당시 조건 기준의 참고값입니다.")

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
                _imp = '<span style="background:#FEF3C7;color:#7a4e00;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">위험 증가</span>' if _gr > _ov * 1.1 else ('<span style="background:#E0F7FA;color:#0284C7;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">위험 감소</span>' if _gr < _ov * 0.9 else '<span style="background:#E2E8F0;color:#475569;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">평균 수준</span>')
                _reason_rows.append({'변수': _rf, '개인 값': str(_vl), '해당그룹 퇴직률': f"{_gr:.1f}%", '전체 평균 퇴직률': f"{_ov:.1f}%", '영향': _imp})
        else:
            _avg = float(df[_rf].mean())
            if _avg != 0:
                _ratio = _emp_v / _avg
                _sal = ['기본급', '연봉', '월급', '급여']
                _yr = ['근무연수', '승진후경과연수']
                _age = ['나이', '연령']
                _u = "만원" if any(k in _rf for k in _sal) else ("년" if any(k in _rf for k in _yr) else ("세" if any(k in _rf for k in _age) else ""))
                _vs = f"{_emp_v:,.0f}{_u}" if _u else f"{_emp_v:,.1f}"
                _as = f"{_avg:,.0f}{_u}" if _u else f"{_avg:,.1f}"
                _imp = '<span style="background:#FEF3C7;color:#7a4e00;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">위험 증가</span>' if _ratio < 0.7 or _ratio > 1.3 else '<span style="background:#E2E8F0;color:#475569;padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600">평균 수준</span>'
                _reason_rows.append({'변수': _rf, '개인 값': _vs, '전체 평균': _as, '평균 대비': f"{_ratio*100:.0f}%", '영향': _imp})
    if _reason_rows:
        show_table_centered(pd.DataFrame(_reason_rows), allow_html_cols=['영향'])
    else:
        st.info("예측 요인 분석 데이터가 부족합니다.")

    # =========================
    # 🆕 SHAP 기반 개인별 요인 분석
    # =========================
    _render_shap_factors(ctx, emp_row)

    # =========================
    # 🆕 What-if 시뮬레이션
    # =========================
    _render_what_if(ctx, emp_row)

    # =========================
    # 1) 동료 그룹 대비 퇴직 위험 비교
    # =========================
    st.subheader("구분 별 퇴직 위험 비교")

    overall_mean_prob = float(_active_proba.mean()) if len(_active_proba) > 0 else 0.0

    peer_cols = ['소속조직', '팀', '직무', '직책']
    peer_rows = []

    for col in peer_cols:
        if col in df.columns:
            try:
                emp_val = emp_row[col].iloc[0]
                # 동일 그룹의 재직자 기준 평균 위험
                peer_df = df[(df[col] == emp_val) & (df['상태'] == 0)]
                if len(peer_df) > 0:
                    peer_proba = float(all_proba[peer_df.index].mean())

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

    _rank_has_chart = False
    for col in ['팀', '소속조직']:
        if col in df.columns:
            try:
                emp_val = emp_row[col].iloc[0]
                # 재직자만 순위에 포함 (조회 대상 직원은 항상 포함)
                grp_mask = (df[col] == emp_val) & ((df['상태'] == 0) | (df.index == emp_idx))
                same_grp = df[grp_mask].copy()
                same_grp['퇴직예측확률'] = all_proba[same_grp.index]
                if len(same_grp) > 1:
                    same_grp = same_grp.sort_values('퇴직예측확률', ascending=False)
                    same_grp['rank'] = range(1, len(same_grp) + 1)

                    my_idx = same_grp.index.intersection([emp_idx])

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
                            bar_names.append(str(_r['사원번호']))
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
                    st.caption(f"강조된 바가 해당 직원입니다. {col} '{display_val}' 내 {n_grp}명 중 {my_rank}위 (상위 {my_rank/n_grp*100:.1f}%)")
                    _rank_has_chart = True
            except Exception:
                pass

    if not _rank_has_chart:
        st.info("팀/소속조직 기준으로 순위를 계산할 수 있는 데이터가 부족합니다.")

    # =========================
    # 3) 상위 변수별 프로필 비교 (위험 기여 막대 + 상세 테이블)
    # =========================
    if len(top_features) > 0:
        st.subheader("상위 변수별 프로필 비교")

        _num_rows = []   # 수치형 변수 상세 테이블용
        _cat_rows = []   # 범주형 변수 상세 테이블용
        _bar_rows = []   # 위험 기여 막대 (수치·범주형 통합)

        for var in top_features:
            if var in label_encoders:
                # 범주형 변수 처리
                raw_val = emp_row[var].iloc[0]
                _el = get_label(raw_val, var, label_encoders)
                same_cat = df[df[var] == raw_val]
                if len(same_cat) > 0 and '상태' in df.columns:
                    cat_turnover = same_cat['상태'].mean() * 100
                    overall_turnover = df['상태'].mean() * 100
                    diff_turnover = cat_turnover - overall_turnover
                    if diff_turnover > 0:
                        risk_tag = f"전체 대비 +{diff_turnover:.1f}%p 높음"
                    elif diff_turnover < -1:
                        risk_tag = f"전체 대비 {diff_turnover:.1f}%p 낮음"
                    else:
                        risk_tag = "전체 평균과 유사"
                    _cat_rows.append({
                        '변수': var,
                        '해당 직원': str(_el),
                        '해당 범주 퇴직률': f"{cat_turnover:.1f}%",
                        '전체 퇴직률': f"{overall_turnover:.1f}%",
                        '위험 수준': risk_tag
                    })

                    # 위험 기여도 (범주형): 해당 범주 퇴직률 - 전체 퇴직률 (%p)
                    try:
                        _bar_rows.append({
                            'label': f"{var} ({_el})",
                            'pp': float(diff_turnover),  # percentage points
                        })
                    except Exception:
                        pass
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
                _pct = int(round((df[var] <= _ev).mean() * 100))

                if _av != 0:
                    ratio = _ev / _av * 100
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

                # 위험 기여도 (수치형): 직원의 값을 5분위 binning → 그 구간의 평균 퇴직률 - 전체
                try:
                    if '상태' in df.columns and df[var].nunique() > 1:
                        # 5-quantile bin (qcut), duplicates='drop'로 중복 경계 처리
                        bins = pd.qcut(df[var], q=5, duplicates='drop')
                        df_b = pd.DataFrame({'b': bins, 's': df['상태'].values})
                        bin_rate = df_b.groupby('b', observed=False)['s'].mean()
                        # 직원의 값이 속하는 bin 찾기
                        emp_bin = pd.cut(
                            [_ev],
                            bins=bins.cat.categories,
                            include_lowest=True
                        )[0]
                        if pd.isna(emp_bin):
                            # 범위 밖이면 가장 가까운 구간 사용
                            emp_rate = bin_rate.iloc[-1] if _ev > df[var].max() else bin_rate.iloc[0]
                        else:
                            emp_rate = bin_rate.get(emp_bin, df['상태'].mean())
                        diff_pp = (float(emp_rate) - float(df['상태'].mean())) * 100
                        _bar_rows.append({
                            'label': f"{var} ({_ev:.1f})",
                            'pp': diff_pp,
                        })
                except Exception:
                    pass

        # --- 위험 기여도 가로 막대 (퍼센트 포인트 단위) ---
        if _bar_rows:
            _bar_df = pd.DataFrame(_bar_rows).sort_values('pp')
            _bar_colors = [COLORS['warning'] if p > 0 else COLORS['primary'] for p in _bar_df['pp']]
            _overall = float(df['상태'].mean() * 100) if '상태' in df.columns else 0.0
            fig_bar = go.Figure(go.Bar(
                x=_bar_df['pp'],
                y=_bar_df['label'],
                orientation='h',
                marker_color=_bar_colors,
                text=[f"{p:+.1f}%p" for p in _bar_df['pp']],
                textposition='outside',
                cliponaxis=False,
            ))
            fig_bar.add_vline(x=0, line_dash="dot", line_color="#9CA3AF")
            _amax = float(_bar_df['pp'].abs().max()) if len(_bar_df) > 0 else 5.0
            _xpad = max(_amax * 1.35, 5.0)
            fig_bar.update_layout(
                title=f"상위 변수별 위험 기여도 (전체 평균 퇴직률 {_overall:.1f}% 기준)",
                xaxis_title="← 평균보다 안전한 그룹        |        평균보다 위험한 그룹 →  (%p)",
                yaxis_title="",
                height=360,
                showlegend=False,
            )
            fig_bar.update_xaxes(range=[-_xpad, _xpad], zeroline=False)
            st.plotly_chart(set_font(fig_bar), use_container_width=True)
            st.markdown(f"""
<div style="background:#F8FAFC; border:1px solid #E5E7EB; border-radius:8px; padding:14px 18px; font-size:13.5px; color:#475569; line-height:1.75;">
<b style="color:#334155;">막대 읽는 법</b><br>
이 차트는 모델이 가장 중요하게 본 상위 6개 변수에 대해,
<b>이 직원이 속한 그룹의 평균 퇴직률이 전체 평균({_overall:.1f}%)보다 얼마나 높은지/낮은지</b>를 퍼센트 포인트(%p)로 보여줍니다.<br><br>
<b style="color:#7a4e00;">예: "평가등급 (CC) +18%p"</b> = CC 등급 직원들의 평균 퇴직률이 전체보다 18%p 더 높다는 뜻.
즉 이 직원의 CC 등급은 퇴직 가능성을 끌어올리는 요인입니다.<br>
<b style="color:#2A9BB0;">예: "근무연수 (14.2) −12%p"</b> = 비슷한 근속자의 평균 퇴직률이 전체보다 12%p 낮다는 뜻. 안정 요인입니다.<br><br>
<b>오른쪽(주황)</b>으로 길수록 그 변수에서 직원의 위치가 <b>퇴직 위험을 끌어올리는 그룹</b>,
<b>왼쪽(청록)</b>으로 길수록 <b>안전한 그룹</b>에 속한다는 의미입니다.
</div>
            """, unsafe_allow_html=True)

        # --- 주요 위험 / 안정 요인 요약 카드 ---
        if _bar_rows:
            _risk = sorted([r for r in _bar_rows if r['pp'] > 1.0], key=lambda r: -r['pp'])[:3]
            _safe = sorted([r for r in _bar_rows if r['pp'] < -1.0], key=lambda r: r['pp'])[:3]

            def _fmt_items(items, color):
                if not items:
                    return "<div style='color:#9CA3AF;font-size:13px;padding:6px 0;'>해당하는 요인이 없습니다.</div>"
                return "".join(
                    f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #F1F5F9;'>"
                    f"<span style='color:#334155;font-weight:500;'>{r['label']}</span>"
                    f"<span style='color:{color};font-weight:700;'>{r['pp']:+.1f}%p</span>"
                    f"</div>"
                    for r in items
                )

            _sc1, _sc2 = st.columns(2)
            with _sc1:
                st.markdown(f"""
                <div style='background:#FFFFFF;border:1px solid #E5E7EB;border-left:4px solid {COLORS['warning']};border-radius:8px;padding:16px 20px;'>
                    <div style='font-size:13px;color:#6B7280;letter-spacing:0.04em;margin-bottom:8px;'>주요 위험 요인 (퇴직 확률 ↑)</div>
                    {_fmt_items(_risk, COLORS['warning'])}
                </div>
                """, unsafe_allow_html=True)
            with _sc2:
                st.markdown(f"""
                <div style='background:#FFFFFF;border:1px solid #E5E7EB;border-left:4px solid {COLORS['primary']};border-radius:8px;padding:16px 20px;'>
                    <div style='font-size:13px;color:#6B7280;letter-spacing:0.04em;margin-bottom:8px;'>주요 안정 요인 (퇴직 확률 ↓)</div>
                    {_fmt_items(_safe, COLORS['primary'])}
                </div>
                """, unsafe_allow_html=True)

        # --- 상세 비교 테이블 (참고용 원본 수치) ---
        with st.expander("상세 비교 테이블 보기", expanded=False):
            if _num_rows:
                st.markdown("**수치형 변수**")
                show_table_centered(pd.DataFrame(_num_rows))
            if _cat_rows:
                st.markdown("**범주형 변수**")
                show_table_centered(pd.DataFrame(_cat_rows))

    # =========================
    # 4) 주요 숫자 변수에서의 위치(분위수)
    # =========================
    st.subheader("주요 숫자 변수별 비교")

    num_candidates = ['근무연수', '승진후경과연수', '나이', '기본급', '입사전이직횟수', '보유역량']
    rows_num = []

    # 변수별 적정 소수 자릿수 (불필요한 소수점 표시 방지)
    _int_vars = {'나이', '기본급', '입사전이직횟수', '보유역량'}

    for col in num_candidates:
        if col in df.columns:
            try:
                series = pd.to_numeric(df[col], errors='coerce').dropna()
                emp_val = float(pd.to_numeric(emp_row[col], errors='coerce').iloc[0])

                if col in _int_vars:
                    mean_str = f"{series.mean():.0f}"
                    emp_str = f"{emp_val:.0f}"
                else:
                    mean_str = f"{series.mean():.1f}"
                    emp_str = f"{emp_val:.1f}"

                # 분위수: 동률 보정 (less + equal/2 — Hyndman-Fan 정의)
                n = len(series)
                if n > 0:
                    less = float((series < emp_val).sum())
                    eq = float((series == emp_val).sum())
                    pct = int(round(((less + eq / 2.0) / n) * 100))
                else:
                    pct = 0

                rows_num.append({
                    '변수': col,
                    '개인값': emp_str,
                    '전체 평균': mean_str,
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
            _sim_cols = ['사원번호', '이름', '소속조직', '팀', '직무', '직책', '퇴직사유']
            _sim_show = [c for c in _sim_cols if c in _top_sim.columns] + ['유사도']
            _sim_disp = _top_sim.copy()
            for _sc in ['소속조직', '팀', '직무', '직책']:
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
                        _vl = esc(get_label(_ev, _tf, label_encoders))
                        tips.append(f"**{_tf}** '{_vl}' 그룹의 퇴직률({_gr*100:.1f}%)이 전체 평균({_ov*100:.1f}%)보다 높습니다. 해당 그룹 대상 **맞춤 리텐션 프로그램** 검토가 필요합니다.")
            else:
                _ev = float(emp_row[_tf].iloc[0])
                _av = float(df[_tf].mean())
                if _av == 0:
                    continue
                _ratio = _ev / _av
                _sal = ['기본급', '연봉', '월급', '급여']
                _ot = ['연장근무', '야근', '초과근무']
                _yr_s = ['근무연수']
                _career = ['입사전이직횟수', '이직횟수']
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
    # 7) 상세 원본 데이터
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
