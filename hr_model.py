# =========================
# 모델 학습(XGBoost) + 중요도 + OOF 예측
# =========================
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score
)


@st.cache_resource(show_spinner=True)
def train_model(X, y):
    """XGBoost 학습.

    반환: (model, metrics, feature_importance, all_proba)
    - all_proba: 5-fold 교차검증의 out-of-fold 예측 확률(pd.Series, X와 동일 인덱스).
      학습에 쓰인 데이터를 그대로 재예측하면 과대낙관되므로,
      대시보드에 표시되는 개인/그룹 위험도는 이 값을 사용한다.
    """
    if y.nunique() < 2:
        return None, {"error": "타깃 클래스가 한 종류만 있습니다. 재직(0)/퇴직(1)이 모두 포함되도록 데이터를 확인하세요."}, None, None

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

    # Early stopping용 내부 validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # 공통 XGBoost 하이퍼파라미터
    xgb_params = dict(
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
        scale_pos_weight=scale_pos_weight,
    )

    # 최종 모델: early stopping으로 자동 종료
    model = xgb.XGBClassifier(
        n_estimators=800,
        early_stopping_rounds=30,
        **xgb_params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # 예측
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 5-fold 교차검증: out-of-fold 확률 + fold별 ROC AUC를 한 번에 계산
    oof = np.full(len(y), np.nan)
    fold_aucs = []
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, te_idx in skf.split(X, y):
            fold_model = xgb.XGBClassifier(n_estimators=300, **xgb_params)
            fold_model.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False)
            p = fold_model.predict_proba(X.iloc[te_idx])[:, 1]
            oof[te_idx] = p
            try:
                fold_aucs.append(roc_auc_score(y.iloc[te_idx], p))
            except Exception:
                pass
        cv_roc_mean = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        cv_roc_std = float(np.std(fold_aucs)) if fold_aucs else float("nan")
    except Exception:
        cv_roc_mean = float("nan")
        cv_roc_std = float("nan")

    # OOF 계산 실패(폴드 분할 불가 등) 시 in-sample 예측으로 폴백
    if np.isnan(oof).any():
        oof_fallback = model.predict_proba(X)[:, 1]
        oof = np.where(np.isnan(oof), oof_fallback, oof)
        oof_is_insample = True
    else:
        oof_is_insample = False

    all_proba = pd.Series(oof, index=X.index, name='퇴직예측확률')

    # 성능 지표 계산
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "cv_mean": {"accuracy": np.nan, "f1": np.nan, "roc_auc": cv_roc_mean},
        "cv_std_roc": cv_roc_std,
        "n_total": int(len(X)),
        "best_iteration": int(getattr(model, "best_iteration", model.n_estimators) or model.n_estimators),
        "oof_is_insample": oof_is_insample,
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

    return model, metrics, feature_importance, all_proba
