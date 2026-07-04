# =========================
# 2) 핵심인재 현황 페이지
# =========================
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hr_styles import COLORS, TEXT_COLOR, set_font, add_pdf_button, RISK_HIGH, RISK_MID
from hr_data import clean_text_series, get_core_mask
from hr_components import (
    show_table_centered, show_core_talent_table, render_explanation,
    bucketize_numeric, humanize_interval_label, excel_download_button,
)


def render(ctx):
    df = ctx['df']; X = ctx['X']
    top_features = ctx['top_features']
    label_encoders = ctx['label_encoders']
    all_proba = ctx['all_proba']          # OOF(교차검증) 예측 확률 — df와 동일 인덱스

    add_pdf_button()
    st.title("핵심인재 퇴직예측")

    core_col = '핵심인재'
    if core_col not in df.columns:
        st.error("'핵심인재' 컬럼이 없습니다.")
        return

    # 🔹 1단계: 핵심인재 전체(재직+퇴직) 추출
    core_mask = get_core_mask(df, label_encoders, core_col)
    core_all = df.loc[core_mask]                          # 핵심인재 전체 (재직+퇴직)
    core_active = core_all[core_all['상태'] == 0].copy()  # 🔥 예측 대상: 재직 핵심인재만

    # 예측 확률은 페이지에서 한 번만 계산해 재사용
    core_proba = all_proba[core_active.index] if len(core_active) > 0 else pd.Series(dtype=float)

    total_core = len(core_all)
    core_left = int(core_all['상태'].sum())
    core_rate = (core_left / total_core * 100) if total_core > 0 else 0
    all_rate = df['상태'].mean() * 100

    core_pred_mean = float(core_proba.mean() * 100) if len(core_proba) > 0 else 0

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
            threshold_90 = core_proba.quantile(0.90)
            high_risk_count = int((core_proba >= threshold_90).sum())
            high_risk_rate = (high_risk_count / len(core_active)) * 100
        else:
            high_risk_count = 0
            high_risk_rate = 0

        # -----------------------------
        # 🔹 3) 핵심인재 퇴직자 중 주요 퇴직사유 — 퇴직한 핵심인재 기준
        # -----------------------------
        if '퇴직사유' in core_all.columns:
            reason_series = clean_text_series(core_all[core_all['상태'] == 1]['퇴직사유'])
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
    # 퇴직위험 등급 분포 + 조직별 히트맵
    # -----------------------------
    st.markdown("---")
    if len(core_active) > 0:
        _core_proba = core_proba.values

        # 등급 분류
        high_cnt = int((_core_proba >= RISK_HIGH).sum())
        mid_cnt = int(((_core_proba >= RISK_MID) & (_core_proba < RISK_HIGH)).sum())
        low_cnt = int((_core_proba < RISK_MID).sum())

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("퇴직위험 등급 분포")
            risk_labels = ['고위험 (≥70%)', '중위험 (30~70%)', '저위험 (<30%)']
            risk_values = [high_cnt, mid_cnt, low_cnt]
            risk_colors = ['#F59E0B', '#7DD3FC', '#48C0D8']
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
                    if v >= RISK_HIGH: return '#F59E0B'
                    elif v >= RISK_MID: return '#7DD3FC'
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
    # 핵심인재 고위험 Top 10
    # -----------------------------
    st.markdown("---")
    st.subheader("핵심인재 고위험군")
    if len(core_active) > 0:
        _top10_df = core_active.copy()
        _top10_df['퇴직예측확률'] = core_proba
        _top10 = _top10_df.sort_values('퇴직예측확률', ascending=False).head(10)

        top10_disp = _top10.copy()
        top10_base = ['사원번호', '이름', '소속조직', '팀', '직책', '직무', '평가등급']
        top10_show = [c for c in top10_base if c in top10_disp.columns] + ['예측퇴직위험']
        top10_disp['예측퇴직위험'] = top10_disp['퇴직예측확률'].apply(lambda x: f"{x*100:.1f}%")
        for c in ['소속조직', '팀', '직책', '직무', '평가등급']:
            if c in label_encoders and c in top10_disp.columns:
                top10_disp[c] = label_encoders[c].inverse_transform(top10_disp[c])

        # 위험 등급 컬럼 추가
        def _risk_badge(prob):
            if prob >= RISK_HIGH:
                return '<span style="color:#F59E0B;">●</span> 고위험'
            elif prob >= RISK_MID:
                return '<span style="color:#7DD3FC;">●</span> 중위험'
            else:
                return '<span style="color:#48C0D8;">●</span> 저위험'
        top10_disp['위험등급'] = _top10['퇴직예측확률'].apply(_risk_badge)
        top10_show.insert(-1, '위험등급')

        show_table_centered(top10_disp[top10_show], allow_html_cols=['위험등급'])
        st.caption("상위 10명은 AI 예측 기반 퇴직 확률이 가장 높은 핵심인재이며, 선제적 리텐션 조치가 필요합니다.")
    else:
        st.info("재직 중인 핵심인재가 없어 Top 10을 표시할 수 없습니다.")

    st.markdown("---")
    st.subheader("핵심인재 전체 리스트")

    # -----------------------------
    # 🔥 5) 핵심인재 전체 리스트 — 재직 핵심인재만 예측
    # -----------------------------
    if len(core_active) > 0:
        core_df_pred = core_active.copy()
        core_df_pred['퇴직예측확률'] = core_proba

        # ① 통계 캐시 생성 (전체 df 기준) — 확장 버전
        stats_cache = {}
        for f in top_features:
            if f in X.columns:
                if f in label_encoders:
                    grp = df.groupby(f)['상태'].agg(['mean', 'count'])
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
                    grp = df.groupby(f)['상태'].agg(['mean', 'count'])
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
                    salary_like = ['기본급', '연봉', '월급', '급여']
                    years_like = ['근무연수', '승진후경과연수']
                    age_like = ['나이', '연령']

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

                    lower_bad = ['기본급', '연봉', '월급', '급여', '만족도', '워라밸', '환경만족', '관계만족', '근무연수', '보유역량']
                    higher_bad = ['야근', '연장근무', '초과근무', '이직횟수', '입사전이직횟수', '통근거리', '거리', '승진후경과연수']

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

        base_cols = ['사원번호', '이름', '소속조직', '팀', '직책', '직무', '평가등급', '인센티브']
        final_cols = [c for c in base_cols if c in all_core.columns] + ['예측퇴직위험', '예측사유']

        disp = all_core.copy()
        disp = disp.rename(columns={'퇴직예측확률': '예측퇴직위험'})
        disp['예측퇴직위험'] = disp['예측퇴직위험'].apply(lambda x: f"{x*100:.1f}%")

        for c in ['소속조직', '팀', '직책', '직무', '평가등급', '인센티브']:
            if c in label_encoders and c in disp.columns:
                disp[c] = label_encoders[c].inverse_transform(disp[c])

        show_core_talent_table(disp[final_cols])
        st.caption("행을 클릭하면 예측사유 상세를 확인할 수 있습니다.")

        # 🆕 핵심인재 리스트 엑셀 다운로드
        excel_download_button(
            disp[final_cols],
            filename="핵심인재_퇴직위험_리스트.xlsx",
            label=f"핵심인재 리스트 다운로드 ({len(disp):,}명)",
            key="dl_core_list",
        )
    else:
        st.info("재직 중인 핵심인재가 없어 예측 리스트를 표시할 수 없습니다.")

    # -----------------------------
    # 핵심인재 퇴직 추이 + 퇴직사유 차트
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
                c_monthly = c_monthly.sort_values('년월')
                c_monthly['월라벨'] = c_monthly['년월'].dt.strftime('%y.%m')

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
            c_vc = clean_text_series(c_reason_df['퇴직사유'])
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
        if c not in ['사원번호', '이름', '상태', '퇴직일', '퇴직사유', '퇴직후이직처']
        and core_all[c].nunique() > 1
    ]

    for var in core_vars:
        # 범주형 (라벨 인코딩 포함)
        if var in label_encoders:
            df_plot = core_all.copy()
            df_plot[var + "_name"] = label_encoders[var].inverse_transform(core_all[var])
            by_var = df_plot.groupby(var + "_name")['상태'].mean() * 100
            x_axis = by_var.index

            fig = px.bar(
                x=x_axis, y=by_var.values,
                color=by_var.values,
                color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                title=f"핵심인재 {var}별 퇴직률(%)",
                labels={'x': var, 'y': '퇴직률(%)'}
            )
            fig.update_layout(height=260, showlegend=False)
            st.plotly_chart(set_font(fig), use_container_width=True)

            max_grp = by_var.idxmax()
            rate = float(by_var.max())
            n = int((df_plot[var + "_name"] == max_grp).sum())
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
                by_var = df_tmp.groupby('bin')['상태'].mean() * 100
                nice_labels = [humanize_interval_label(var, b) for b in by_var.index]

                fig = px.bar(
                    x=nice_labels, y=by_var.values,
                    color=by_var.values,
                    color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                    title=f"핵심인재 {var}별 퇴직률(%)",
                    labels={'x': '구간', 'y': '퇴직률(%)'}
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
