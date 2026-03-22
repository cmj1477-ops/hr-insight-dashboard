import pandas as pd
import numpy as np

# Load data
file_path = '/Users/minju/Desktop/phyton/HR_퇴직예측_샘플데이터_100명_latest.csv'
df = pd.read_csv(file_path)

# Normalize columns if needed (based on pj7.py logic)
# Assuming the CSV has standard names or '재직'/'퇴직' in '상태' column if it exists, 
# but the user is asking about "Predicted" risk, which implies the model's output.
# However, the explanation should be based on the input features that *cause* the model to predict high risk.

# Target group: Management Division
target_group = df[df['소속조직'] == '경영본부']
rest_group = df[df['소속조직'] != '경영본부']

# Redirect output to file
import sys
import contextlib
with open('analysis_result.txt', 'w') as f, contextlib.redirect_stdout(f):
    
    print(f"Management Division Count: {len(target_group)}")
    print(f"Rest of Company Count: {len(rest_group)}")

    # Analyze Numerical Columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n--- Numerical Feature Comparison (Mean) ---")
    print(f"{'Feature':<20} | {'Mgmt Div':<10} | {'Rest':<10} | {'Diff':<10}")
    print("-" * 60)

    significant_factors = []

    for col in numeric_cols:
        if col in ['사원번호', '상태']: continue # Skip ID and Target
        
        mean_target = target_group[col].mean()
        mean_rest = rest_group[col].mean()
        diff = mean_target - mean_rest
        
        print(f"{col:<20} | {mean_target:<10.2f} | {mean_rest:<10.2f} | {diff:<10.2f}")
        
        # Heuristic for "significant" difference (just for highlighting)
        if col in ['직무만족도', '환경만족도', '관계만족도', '워라밸', '성과평가', '동료피드백']:
            if diff < -0.2: significant_factors.append(f"Lower {col} ({mean_target:.1f} vs {mean_rest:.1f})")
        elif col in ['야근시간', '통근거리', '입사전이직횟수']:
            if diff > 1: significant_factors.append(f"Higher {col} ({mean_target:.1f} vs {mean_rest:.1f})")
        elif col in ['연봉(만원)']:
            if diff < -300: significant_factors.append(f"Lower {col} ({mean_target:.0f} vs {mean_rest:.0f})")

    # Analyze Categorical Columns (Mode or Distribution)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print("\n--- Categorical Feature Comparison (Top Value) ---")
    for col in categorical_cols:
        if col in ['이름', '소속조직', '퇴직일', '퇴직사유']: continue
        
        top_target = target_group[col].mode()[0] if not target_group.empty else "N/A"
        top_rest = rest_group[col].mode()[0] if not rest_group.empty else "N/A"
        
        print(f"{col:<20} | Mgmt: {top_target} | Rest: {top_rest}")

    print("\n--- Potential Reasons for High Risk ---")
    for factor in significant_factors:
        print(f"- {factor}")
