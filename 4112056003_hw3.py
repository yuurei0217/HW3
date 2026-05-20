import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="CRISP-DM Regression Demo", layout="wide")

st.title("🍷 Red Wine Quality Prediction - CRISP-DM Pipeline")
st.markdown("本應用程式展示了 **CRISP-DM (Cross-Industry Standard Process for Data Mining)** 流程，結合 `scikit-learn` 完整 Pipeline，預測紅酒品質。")

# --- Phase 1: Business Understanding ---
st.header("1. Business Understanding (商業理解)")
st.info("目標：根據紅酒的化學測試結果 (如酸度、糖分、pH值等)，建立多元線性回歸模型來預測其品質 (Quality)。這能協助釀酒師自動化品鑑流程並優化配方。")

# --- Phase 2: Data Understanding ---
st.header("2. Data Understanding (資料理解)")
@st.cache_data
def load_data():
    # Kaggle 資料集 (yasserh/wine-quality-dataset) 源自 UCI
    # 因 Kaggle 需要 API key 認證，此處直接從開源的 UCI 相同資料集網址載入以確保 Web App 順暢運作
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    return pd.read_csv(url, sep=';')

st.markdown("**資料來源:** [Kaggle - Wine Quality Dataset (yasserh)](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)")
df = load_data()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw Data (前5筆)")
    st.dataframe(df.head())
with col2:
    st.subheader("Data Summary")
    st.write(f"資料集大小： {df.shape[0]} 筆樣本, {df.shape[1]} 個特徵")
    st.write("各特徵基本統計量：")
    st.dataframe(df.describe().T[['mean', 'std', 'min', 'max']])

st.subheader("Exploratory Data Analysis (EDA)")
if st.checkbox("顯示相關係數熱力圖 (Correlation Heatmap)"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- Phase 3: Data Preparation ---
st.header("3. Data Preparation (資料準備)")
st.write("在此階段，我們將切割訓練集與測試集 (80/20)，並透過 `sklearn` 的 Pipeline 進行資料標準化 (StandardScaler)。")

X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"- 訓練集大小: {X_train.shape}")
st.write(f"- 測試集大小: {X_test.shape}")

# --- Phase 4: Modeling ---
st.header("4. Modeling (建立模型與特徵選擇)")

st.write("使用 `sklearn.pipeline.Pipeline` 打包：")
st.code("""
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', RFE(estimator=LinearRegression(), n_features_to_select=n_features)),
    ('regressor', LinearRegression())
])
""", language='python')

# 讓使用者選擇特徵數量
n_features = st.slider("選擇要保留的特徵數量 (RFE - Recursive Feature Elimination)", min_value=3, max_value=11, value=7)

# 建立 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', RFE(estimator=LinearRegression(), n_features_to_select=n_features)),
    ('regressor', LinearRegression())
])

# 訓練模型
pipeline.fit(X_train, y_train)

# 取得被選中的特徵
rfe_step = pipeline.named_steps['feature_selection']
selected_features = X.columns[rfe_step.support_].tolist()

st.success(f"模型訓練完成！保留的 {n_features} 個特徵為：\n" + ", ".join(selected_features))

# --- Phase 5: Evaluation ---
st.header("5. Evaluation (模型評估)")
y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("MSE", f"{mse:.4f}")
col2.metric("RMSE", f"{rmse:.4f}")
col3.metric("MAE", f"{mae:.4f}")
col4.metric("R² Score", f"{r2:.4f}")

# 預測圖
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted with 95% Prediction Interval
# 使用 1.96 * RMSE 作為 95% 預測區間的近似值
pi_margin = 1.96 * rmse
results_df = pd.DataFrame({
    'Actual': y_test.values, 
    'Predicted': y_pred,
    'Lower': y_pred - pi_margin,
    'Upper': y_pred + pi_margin
})
results_df = results_df.sort_values(by='Actual').reset_index(drop=True)

ax1.plot(results_df.index, results_df['Actual'], 'o', color='darkred', label='Actual', alpha=0.6)
ax1.plot(results_df.index, results_df['Predicted'], '-', color='royalblue', label='Predicted')
ax1.fill_between(results_df.index, results_df['Lower'], results_df['Upper'], color='royalblue', alpha=0.15, label='95% Prediction Interval')
ax1.set_title('Actual vs Predicted Quality')
ax1.set_xlabel('Samples (Sorted)')
ax1.set_ylabel('Quality')
ax1.legend()

# Residual Plot
residuals = y_test - y_pred
sns.residplot(x=y_pred, y=residuals, lowess=True, color="darkred", line_kws={"color": "royalblue"}, ax=ax2)
ax2.set_title('Residuals vs Predicted')
ax2.set_xlabel('Predicted Quality')
ax2.set_ylabel('Residuals')
ax2.axhline(0, color='gray', linestyle='--')

st.pyplot(fig)

# --- Phase 6: Deployment ---
st.header("6. Deployment (互動預測)")
st.write("透過側邊欄 (Sidebar) 調整紅酒的化學特徵，模型將即時預測紅酒品質。")

st.sidebar.header("🍷 輸入紅酒特徵")
user_input = {}
for col in X.columns:
    # 根據資料集的 min 和 max 設定 slider
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())
    user_input[col] = st.sidebar.slider(col, min_val, max_val, mean_val)

input_df = pd.DataFrame([user_input])

# 預測
prediction = pipeline.predict(input_df)[0]

st.sidebar.markdown("---")
st.sidebar.subheader("預測品質 (Quality)")
st.sidebar.markdown(f"<h1 style='text-align: center; color: #ff4b4b;'>{prediction:.2f}</h1>", unsafe_allow_html=True)
if prediction >= 6.5:
    st.sidebar.success("這是一瓶高品質的好酒！🍾")
elif prediction >= 5.5:
    st.sidebar.warning("這是一瓶普通品質的酒。🍷")
else:
    st.sidebar.error("這瓶酒的品質可能較差。📉")
