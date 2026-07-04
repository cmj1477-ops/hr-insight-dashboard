# =========================
# 1) 전체 현황 페이지
# =========================
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import shap

from hr_styles import COLORS, TEXT_COLOR, set_font, add_pdf_button
from hr_data import MISSING_TOKENS, CAT_COLS, clean_text_series, get_core_mask
from hr_components import (
    show_table_centered, render_risk_group_card,
    bucketize_numeric, humanize_interval_label, cramers_v,
    excel_download_button,
)


def render(ctx):
    df = ctx['df']; X = ctx['X']
    model = ctx['model']; metrics = ctx['metrics']
    feature_importance = ctx['feature_importance']
    top_features = ctx['top_features']
    label_encoders = ctx['label_encoders']
    all_proba = ctx['all_proba']          # OOF(교차검증) 예측 확률 — df와 동일 인덱스

    add_pdf_button()
    st.title("전체 현황")
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
            core_mask = get_core_mask(df, label_encoders)
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

    # 퇴직사유 요약 — 미입력/미기재 등 결측 토큰 제외
    reason_text = "데이터 없음"
    if '퇴직사유' in df.columns:
        try:
            vc = clean_text_series(df.loc[df['상태'] == 1, '퇴직사유'])
            reason_summary = (vc.value_counts(normalize=True).head(3) * 100).round(1)
            if len(reason_summary) > 0:
                reason_text = ", ".join([f"{idx}({val}%)" for idx, val in reason_summary.items()])
        except Exception:
            pass

    # 이직처 요약 — 미입력/미기재 등 결측 토큰 제외 (이직 인원수 과대집계 방지)
    dest_text = "데이터 없음"; num_move = 0
    try:
        moved_df = df[df['상태'] == 1]
        if '퇴직후이직처' in moved_df.columns:
            dest_series = clean_text_series(moved_df['퇴직후이직처'])
            num_move = int(len(dest_series))
            top_dest = dest_series.value_counts().head(3)
            if len(top_dest) > 0:
                dest_text = ", ".join([f"{idx} {val}명" for idx, val in top_dest.items()])
    except Exception:
        pass

    total_employees = len(df)
    left_count = int(df['상태'].sum())
    active_count = total_employees - left_count
    overall_rate = (left_count / total_employees) * 100 if total_employees else 0
    retention_rate = 100 - overall_rate

    # 재직자 기준 예측 퇴직위험 평균 (퇴직자를 포함하면 지표가 부풀려지므로 제외)
    _active_mask = df['상태'] == 0
    if _active_mask.sum() > 0:
        overall_pred_mean = float(all_proba[_active_mask].mean() * 100)
    else:
        overall_pred_mean = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("재직자/퇴직자", f"{active_count:,}명/{left_count:,}명")
    c2.metric("퇴직률", f"{overall_rate:.1f}%")
    c3.metric("재직률", f"{retention_rate:.1f}%")
    c4.metric("재직자 퇴직위험 평균", f"{overall_pred_mean:.1f}%")

    core_rate_disp = f"{core_rate:.1f}%" if not np.isnan(core_rate) else "-"
    st.markdown(f"""
> 🔹 전체 퇴직률 **{total_rate:.1f}%**, 핵심인재 퇴직률 **{core_rate_disp}**이며 월별 퇴직 추이는 **{trend}** 하고 있습니다.{"  "}
> 🔹 조직별 퇴직자는 {org_text} 입니다.{"  "}
> 🔹 퇴직 사유로는 **{reason_text}** 이며, 이직 인원 **{num_move}명**의 주요 이직처는 **{dest_text}** 입니다.{"  "}
> 🔹 해당 데이터는 사무기술직이 대상이며, 임원 및 계약직은 제외하였습니다.
""")

    a, b = st.columns([1, 2])
    with a:
        st.subheader("인원 현황 비율")
        cnts = df['상태'].value_counts()
        labels = ['재직(0)', '퇴직(1)']
        values = [cnts.get(0, 0), cnts.get(1, 0)]
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
                _shap_explainer = shap.TreeExplainer(model)
                _shap_values = _shap_explainer.shap_values(df[X.columns])
                if isinstance(_shap_values, list):
                    _shap_values = _shap_values[1]
                _shap_df = pd.DataFrame(_shap_values, columns=X.columns)
                # 방향 판정:
                # - mean(shap)은 클래스 가중치(scale_pos_weight) 때문에 전 변수가 음수로
                #   쏠릴 수 있어 방향 지표로 부적합
                # - 수치형: 변수값 ↔ SHAP값 상관의 부호 → "값이 높을수록/낮을수록 위험"
                # - 범주형: 범주별 SHAP 평균이 가장 높은 그룹 → 모델이 위험하게 본 그룹
                for feat in top_imp.index:
                    try:
                        if feat in label_encoders:
                            grp_shap = pd.DataFrame(
                                {'v': df[feat].values, 's': _shap_df[feat].values}
                            ).groupby('v')['s'].mean()
                            if len(grp_shap) > 0:
                                _shap_direction[feat] = ('cat', int(grp_shap.idxmax()))
                        else:
                            vals = df[feat].astype(float)
                            if vals.nunique() > 1 and _shap_df[feat].std() > 0:
                                corr = float(np.corrcoef(vals, _shap_df[feat])[0, 1])
                                _shap_direction[feat] = ('num', corr)
                    except Exception:
                        pass
            except Exception:
                pass

            _total_imp = feature_importance.sum()
            interpretation_rows = []
            for feat in top_imp.index:
                importance_pct = (top_imp[feat] / _total_imp * 100) if _total_imp > 0 else 0

                left_group = df[df['상태'] == 1][feat].mean()
                stay_group = df[df['상태'] == 0][feat].mean()

                feat_display = feat
                detail = ""
                data_high_cat = None
                if feat in label_encoders:
                    le = label_encoders[feat]
                    classes = list(le.classes_)
                    # 라벨 인코딩된 정수의 평균은 카테고리를 가리키지 않으므로
                    # 그룹별 실제 퇴직률(상태 평균)을 비교해서 최고/최저 범주를 찾는다
                    try:
                        grp = df.groupby(feat)['상태'].mean()
                        if len(grp) > 0:
                            high_cat_idx = int(grp.idxmax())
                            low_cat_idx = int(grp.idxmin())
                            high_cat = classes[high_cat_idx] if 0 <= high_cat_idx < len(classes) else str(high_cat_idx)
                            low_cat = classes[low_cat_idx] if 0 <= low_cat_idx < len(classes) else str(low_cat_idx)
                            data_high_cat = high_cat
                            detail = (
                                f"'{high_cat}' 그룹의 퇴직률이 가장 높고 "
                                f"({grp.max()*100:.1f}%), '{low_cat}' 그룹이 가장 낮음 "
                                f"({grp.min()*100:.1f}%)"
                            )
                        else:
                            detail = "범주별 퇴직률 산정 불가"
                    except Exception:
                        detail = "범주별 퇴직률 산정 불가"
                else:
                    if left_group > stay_group:
                        detail = f"값이 높을수록 퇴직 확률 상승 (퇴직자 평균: {left_group:.1f}, 재직자 평균: {stay_group:.1f})"
                    else:
                        detail = f"값이 낮을수록 퇴직 확률 상승 (퇴직자 평균: {left_group:.1f}, 재직자 평균: {stay_group:.1f})"

                # 방향 표시: SHAP 기반(우선) → 데이터 기반(폴백)
                _dir_info = _shap_direction.get(feat)
                if feat in label_encoders:
                    risk_cat = None
                    if _dir_info is not None and _dir_info[0] == 'cat':
                        try:
                            risk_cat = label_encoders[feat].inverse_transform([_dir_info[1]])[0]
                        except Exception:
                            risk_cat = None
                    if risk_cat is None:
                        risk_cat = data_high_cat
                    arrow = f"● '{risk_cat}' 그룹 위험" if risk_cat is not None else "-"
                else:
                    if _dir_info is not None and _dir_info[0] == 'num':
                        higher_risky = _dir_info[1] > 0
                    else:
                        higher_risky = left_group > stay_group
                    arrow = "▲ 값 높을수록 위험" if higher_risky else "▼ 값 낮을수록 위험"
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
                <b>방향</b>: 모델(SHAP 분석)이 판단한 위험 방향 — 어떤 값(▲높음/▼낮음)·그룹(●)일 때 퇴직 위험이 높아지는지<br>
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
                monthly = monthly.sort_values('년월')
                monthly['월라벨'] = monthly['년월'].dt.strftime('%y.%m')

                max_cnt = int(monthly['퇴직자 수'].max()) if len(monthly) > 0 else 0
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
            vc = clean_text_series(reason_df['퇴직사유'])
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
        dest = clean_text_series(moved_df['퇴직후이직처'])
        dest_counts = dest.value_counts(dropna=True)
        if dest_counts.sum() > 0:
            dest_df = dest_counts.head(15).reset_index()
            dest_df.columns = ['이직처', '건수']
            # Bar: Unified Blue Gradient
            fig_dest = px.bar(dest_df, x='건수', y='이직처', orientation='h',
                              labels={'건수': '건수', '이직처': '이직처'},
                              color='건수', color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                              text='건수')
            x_max = float(dest_df['건수'].max()) if len(dest_df) > 0 else 0
            fig_dest.update_traces(textposition='outside', cliponaxis=False)
            fig_dest.update_layout(height=320, showlegend=False,
                                   xaxis=dict(range=[0, x_max * 1.15 if x_max > 0 else 1]))
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
                df_plot[var + "_name"] = label_encoders[var].inverse_transform(df[var])
                group_rates = df_plot.groupby(var + "_name")['상태'].mean() * 100
                x_vals = group_rates.index
                # Bar: Unified Blue Gradient
                fig_bar = px.bar(x=x_vals, y=group_rates.values, color=group_rates.values,
                                 color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                                 title=f"{var} 별 퇴직률(%)", labels={'x': var, 'y': '퇴직률(%)'})
                fig_bar.update_layout(height=300, showlegend=False)
                st.plotly_chart(set_font(fig_bar), use_container_width=True)

                max_grp = group_rates.idxmax()
                rate = float(group_rates.max())
                n = int((df_plot[var + "_name"] == max_grp).sum())
                share = n / len(df_plot) * 100
                diff = rate - overall_rate  # 전체 평균 퇴직률 대비 차이 (%p)

                render_risk_group_card(var, f"'{max_grp}'", rate, diff, n, total_employees, share)
            else:
                try:
                    bins = bucketize_numeric(df[var], bins="quartile")
                    df_tmp = pd.DataFrame({var: df[var], 'bin': bins, '상태': df['상태']})
                    group_rates = df_tmp.groupby('bin')['상태'].mean() * 100
                    nice_labels = [humanize_interval_label(var, b) for b in group_rates.index]

                    # Bar: Unified Blue Gradient
                    fig_bar = px.bar(x=nice_labels, y=group_rates.values,
                                     color=group_rates.values, color_continuous_scale=[(0, '#A5E6F3'), (1, COLORS['primary'])],
                                     title=f"{var} 별 퇴직률(%)", labels={'x': '구간', 'y': '퇴직률(%)'})
                    fig_bar.update_layout(height=300, showlegend=False)
                    st.plotly_chart(set_font(fig_bar), use_container_width=True)

                    idxmax = group_rates.idxmax()
                    rate = float(group_rates.max())
                    bucket_label = humanize_interval_label(var, idxmax)
                    n = int((df_tmp['bin'] == idxmax).sum())
                    share = n / len(df_tmp) * 100
                    diff = rate - overall_rate  # 전체 평균 대비 차이 (%p)

                    render_risk_group_card(var, bucket_label, rate, diff, n, total_employees, share)
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
        cv_df = pd.DataFrame(cat_vs, columns=['변수', '관련도']).sort_values('관련도', ascending=False)
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
    for col in ['소속조직', '팀', '직무', '직위', '직책']:
        if col in df.columns:
            dept_col = col; break
    if dept_col:
        # 재직자 기준으로 조직별 평균 위험 산출
        df_pred = df[df['상태'] == 0].copy()
        df_pred['퇴직예측확률'] = all_proba[df_pred.index]
        dept_risk = df_pred.groupby(dept_col)['퇴직예측확률'].mean().sort_values(ascending=False)
        top5 = dept_risk.head(5).reset_index().rename(columns={'퇴직예측확률': '평균 퇴직위험(%)'})
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
                if pd.api.types.is_numeric_dtype(df[var]) and var not in label_encoders:
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
                        grp_rate_dept = df[df[dept_col] == top_group_val].groupby(var)['상태'].mean()
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
    active_pred = df[df['상태'] == 0].copy()
    active_pred['퇴직예측확률'] = all_proba[active_pred.index]

    if len(active_pred) == 0:
        st.info("재직 중인 직원이 없어 개인별 예측 순위를 표시할 수 없습니다.")
    else:
        rank_all = active_pred.sort_values('퇴직예측확률', ascending=False)
        disp_cols = [c for c in ['사원번호', '이름', '직무', '소속조직', '팀', '직책'] if c in df.columns] + ['퇴직예측확률']
        rank_disp = rank_all[disp_cols].rename(columns={'퇴직예측확률': '퇴직위험확률'})
        rank_disp['퇴직위험확률'] = rank_disp['퇴직위험확률'].apply(lambda x: f"{x*100:.1f}%")
        for c in ['직무', '소속조직', '팀', '직책']:
            if c in label_encoders and c in rank_disp.columns:
                rank_disp[c] = label_encoders[c].inverse_transform(rank_disp[c])
        show_table_centered(rank_disp.head(10))
        # 🆕 재직자 전체 위험 리스트 엑셀 다운로드
        excel_download_button(
            rank_disp,
            filename="재직자_퇴직위험_리스트.xlsx",
            label=f"재직자 전체 위험 리스트 다운로드 ({len(rank_disp):,}명)",
            key="dl_overview_rank",
        )

    st.markdown("---")
    with st.expander("모델 설명 및 신뢰도", expanded=False):

        # ----- 섹션 전용 스타일 -----
        st.markdown("""
        <style>
        .mdl-badge-card {
            border-radius: 12px;
            padding: 24px 28px;
            color: #FFFFFF;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 8px;
        }
        .mdl-badge-label {
            font-size: 13px;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            opacity: 0.85;
            margin-bottom: 6px;
        }
        .mdl-badge-value {
            font-size: 32px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 4px;
        }
        .mdl-badge-sub {
            font-size: 14px;
            opacity: 0.92;
        }
        .mdl-metric-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 10px;
        }
        .mdl-metric-name {
            font-size: 14px;
            color: #6B7280;
            margin-bottom: 4px;
            font-weight: 500;
        }
        .mdl-metric-val {
            font-size: 22px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 8px;
        }
        .mdl-metric-bar-bg {
            height: 8px;
            background: #F3F4F6;
            border-radius: 4px;
            overflow: hidden;
        }
        .mdl-metric-bar-fill {
            height: 100%;
            border-radius: 4px;
        }
        .mdl-metric-desc {
            font-size: 12px;
            color: #6B7280;
            margin-top: 6px;
        }
        .mdl-cm-card {
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .mdl-cm-num {
            font-size: 36px;
            font-weight: 800;
            line-height: 1.1;
        }
        .mdl-cm-label {
            font-size: 14px;
            font-weight: 600;
            margin-top: 6px;
            letter-spacing: 0.02em;
        }
        .mdl-cm-desc {
            font-size: 12px;
            opacity: 0.9;
            margin-top: 4px;
        }
        .mdl-notice {
            background: transparent;
            padding: 16px 4px 4px 4px;
            color: #475569;
            font-size: 14px;
            line-height: 1.7;
            margin-top: 16px;
        }
        .mdl-notice b {
            color: #334155;
            font-weight: 700;
        }
        </style>
        """, unsafe_allow_html=True)

        _acc = metrics['accuracy']
        _f1 = metrics['f1']
        _roc = metrics['roc_auc']
        _pr = metrics['pr_auc']
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        _total_test = tn + fp + fn + tp

        # 신뢰도 등급: 단일 split은 변동성이 크므로 5-fold CV 평균을 우선 사용
        _cv_roc = metrics.get('cv_mean', {}).get('roc_auc', float('nan'))
        _cv_std = metrics.get('cv_std_roc', float('nan'))
        _n_total = metrics.get('n_total', 0)

        if isinstance(_cv_roc, float) and not np.isnan(_cv_roc):
            _grade_roc = _cv_roc
            _roc_source = f"5-Fold CV 평균 ROC AUC {_cv_roc:.2f} (±{_cv_std:.2f})"
        else:
            _grade_roc = _roc
            _roc_source = f"ROC AUC {_roc:.2f}"

        _light_cyan = "#7DD3FC"
        if _grade_roc >= 0.9:
            _grade, _grade_color, _grade_msg = "매우 높음", COLORS["primary"], "실무 의사결정에 자신 있게 활용 가능합니다."
        elif _grade_roc >= 0.8:
            _grade, _grade_color, _grade_msg = "높음", _light_cyan, "보조 지표로 적극 활용할 수 있는 수준입니다."
        elif _grade_roc >= 0.7:
            _grade, _grade_color, _grade_msg = "보통", COLORS["secondary"], "참고 자료로 활용하되 정성적 판단을 병행하세요."
        else:
            _grade, _grade_color, _grade_msg = "낮음", COLORS["warning"], "데이터 보강 후 재학습이 권장됩니다."

        # ===== 1) 용어와 원리 =====
        st.markdown("### 용어와 원리")
        st.markdown("""
**XGBoost 모델이란?**

이 대시보드의 예측 엔진은 **XGBoost(eXtreme Gradient Boosting)**라는 머신러닝 알고리즘입니다.

- **수많은 작은 의사결정 나무(Decision Tree)를 순차적으로 쌓아 올리는** 방식입니다.
  앞선 나무가 틀린 부분을 다음 나무가 보완하면서 점점 정답에 가까워집니다.
- 직원 한 명에 대해 "근무연수 5년 미만인가?", "평가등급이 BB 이하인가?", "서울 근무인가?" 같은
  **수십~수백 개의 질문을 조합**해서 최종적으로 퇴직 확률을 계산합니다.
- 변수 간 **복잡한 상호작용**(예: "젊은 + 서울 + 연장근무"가 동시에 있을 때 위험 급증)을
  자동으로 학습하기 때문에, 단순 통계로는 보이지 않는 패턴까지 잡아냅니다.
- 결측치(빈 값)에도 강하고, 캐글(Kaggle) 등 데이터 경진대회에서 가장 많이 우승해 온
  **검증된 모델**이라 HR·금융·마케팅 등 실무에서 폭넓게 쓰입니다.

**이 모델은 무엇을 하나요?**

직원들의 근속연수, 직무, 조직, 급여 등 **여러 정보를 종합**해서,
"이 직원이 앞으로 퇴직할 가능성이 높은지 낮은지"를 **확률(%)**로 알려줍니다.
예를 들어 퇴직 위험 72%라면, 과거에 비슷한 조건의 직원 100명 중 약 72명이 퇴직했다는 의미입니다.

**표시되는 확률은 어떻게 계산되나요?**

대시보드의 개인/그룹 위험도는 **5-fold 교차검증(Out-of-Fold) 예측값**입니다.
즉, 각 직원의 확률은 "그 직원을 학습에 사용하지 않은 모델"이 예측한 값이라,
학습 데이터를 그대로 재예측할 때 생기는 과대낙관(overfitting 착시)이 제거되어 있습니다.

**영향도(%)는 무엇인가요?**

모델이 퇴직 여부를 판단할 때 **어떤 정보를 얼마나 중요하게 봤는지**를 비율로 나타낸 것입니다.
예: 핵심인재 여부 38%, 근속연수 15% → 모델이 퇴직을 예측할 때 핵심인재 여부를 가장 많이 참고했다는 뜻입니다.

**상관계수 / 관련도는 무엇인가요?**

- **상관계수**(-1 ~ +1): 숫자형 변수와 퇴직의 관계입니다.
  - +0.5 이상이면 "그 값이 높을수록 퇴직이 많다"
  - -0.5 이하면 "그 값이 높을수록 퇴직이 적다"
  - 0에 가까우면 퇴직과 별 관련이 없습니다.
- **관련도**(0 ~ 1): 범주형 변수(직무, 조직 등)와 퇴직의 관계입니다.
  - 1에 가까울수록 "어떤 그룹이냐에 따라 퇴직 비율 차이가 크다"는 뜻입니다.

**모델 작동 원리**

- 이 대시보드는 **XGBoost 기반 이진 분류 모델**로, 각 직원의 특성을 입력받아 **퇴직(1) / 재직(0) 확률**을 예측합니다.
- 직무, 조직, 연장근무, 보상수준 등 여러 변수가 서로 섞여 작용하는 패턴을 함께 학습합니다.
- 경기, 조직개편, 경영전략 변화처럼 데이터에 없는 외부 요인은 반영하지 못하므로, **HR/리더의 정성적 판단과 함께 쓰는 보조 도구**로 보는 것이 적절합니다.

**주의할 점**

- 이 모델은 **과거 데이터의 패턴**을 학습한 것이므로, 미래를 100% 맞추지는 못합니다.
- 경기 변동, 조직개편, 개인 사정 등 **데이터에 없는 요인은 반영되지 않습니다.**
- "퇴직 위험이 높다" = "반드시 퇴직한다"가 아니라, **"관심을 가지고 살펴볼 필요가 있다"**는 신호입니다.
- HR담당자와 리더의 **정성적 판단과 함께 보조 도구로 활용**하는 것을 권장합니다.
        """)

        st.markdown("---")

        # ===== 2) 신뢰도 =====
        st.markdown("### 신뢰도")
        st.markdown(f"""
        <div class="mdl-badge-card" style="background: linear-gradient(135deg, {_grade_color} 0%, {_grade_color}CC 100%);">
            <div class="mdl-badge-label">모델 신뢰도 등급</div>
            <div class="mdl-badge-value">{_grade}</div>
            <div class="mdl-badge-sub">{_roc_source} · {_grade_msg}</div>
        </div>
        """, unsafe_allow_html=True)

        if _n_total > 0 and _n_total < 1000:
            st.markdown(f"""
            <div class="mdl-notice">
                현재 학습 표본은 <b>{_n_total}명</b>입니다. 표본이 1,000명 미만이면 신뢰도 평가 자체가 변동성이 큽니다.
                동일 데이터로 재학습 시 등급이 한 단계 오르내릴 수 있으므로, 가능하면 표본을 확대해 학습하시기를 권장합니다.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### 핵심 지표")

        _gauges = [
            ("정확도 (Accuracy)", _acc, "전체 예측 중 맞춘 비율"),
            ("F1 Score", _f1, "놓치지 않으면서 과잉경보를 줄인 균형 지표"),
            ("ROC AUC", _roc, "재직자와 퇴직자를 얼마나 잘 구분하는지"),
            ("PR AUC", _pr, "퇴직자 비중이 적을 때의 정밀 예측력"),
        ]

        _c1, _c2 = st.columns(2)
        for _i, (_name, _val, _desc) in enumerate(_gauges):
            with (_c1 if _i % 2 == 0 else _c2):
                _pct = max(0, min(1, _val)) * 100
                st.markdown(f"""
                <div class="mdl-metric-card">
                    <div class="mdl-metric-name">{_name}</div>
                    <div class="mdl-metric-val">{_val:.2f}<span style="font-size:14px;color:#9CA3AF;font-weight:500;">  /  1.00</span></div>
                    <div class="mdl-metric-bar-bg">
                        <div class="mdl-metric-bar-fill" style="width:{_pct}%; background:{COLORS['primary']};"></div>
                    </div>
                    <div class="mdl-metric-desc">{_desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ===== 3) 세부사항 =====
        st.markdown("### 세부사항")
        st.markdown("##### 혼동행렬 (테스트 표본 기준)")
        st.caption(f"테스트 데이터 총 {_total_test}명을 대상으로 모델의 예측이 실제 결과와 얼마나 일치했는지 보여줍니다.")

        _cards = [
            (tp, "사전 포착", "실제 퇴직자를 퇴직으로 정확히 예측", COLORS["primary"]),
            (tn, "정상 재직", "실제 재직자를 재직으로 정확히 예측", _light_cyan),
            (fn, "놓친 퇴직자", "실제 퇴직자를 재직으로 잘못 예측", COLORS["warning"]),
            (fp, "과잉 경보", "실제 재직자를 퇴직으로 잘못 예측", COLORS["secondary"]),
        ]

        _cc = st.columns(4)
        for _col, (_num, _lbl, _d, _bg) in zip(_cc, _cards):
            _txt = "#334155" if _bg == _light_cyan else "#FFFFFF"
            with _col:
                st.markdown(f"""
                <div class="mdl-cm-card" style="background:{_bg}; color:{_txt};">
                    <div class="mdl-cm-num">{_num}</div>
                    <div class="mdl-cm-label">{_lbl}</div>
                    <div class="mdl-cm-desc">{_d}</div>
                </div>
                """, unsafe_allow_html=True)

        _total_quit = tp + fn
        _catch_rate = (tp / _total_quit * 100) if _total_quit > 0 else 0
        _precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        st.markdown(f"""
        <div class="mdl-notice">
            테스트 표본의 실제 퇴직자 <b>{_total_quit}명</b> 중 <b>{tp}명({_catch_rate:.0f}%)</b>을 사전에 포착했고,
            고위험 예측 <b>{tp + fp}명</b> 중 <b>{tp}명({_precision:.0f}%)</b>이 실제 퇴직자였습니다.
        </div>
        """, unsafe_allow_html=True)
