# AIoT HW3 - Red Wine Quality Prediction 🍷

本專案使用 **CRISP-DM** 流程，透過多元線性回歸 (Multiple Linear Regression) 模型，根據理化測試指標預測紅酒的品質 (Quality)。專案不僅包含靜態分析與完整的特徵選擇，還建構了具有互動資料視覺化與機器學習 Pipeline 的 Streamlit Web 應用程式。

## 🌟 專案成果與亮點 (Highlights)
1. **完整的 sklearn Pipeline**：整合了 `StandardScaler` (資料標準化)、`RFE` (遞迴特徵消除法) 以及 `LinearRegression`，實現了自動化的機器學習工作流。
2. **優異的模型表現**：Kaggle 社群針對此資料集的線性回歸基礎 MSE 約落在 0.40~0.42，而本專案經由特徵篩選後，模型測試 **MSE 達到 0.3913**，精準度超越一般基準。
3. **專業的殘差分析視覺化**：不只輸出常見的實際 vs. 預測對比圖，更繪製了帶有 LOWESS 平滑線的殘差圖 (Residual Plot)，有效檢驗模型誤差分佈。
4. **即時互動 Web App**：將模型部署為 Streamlit 應用程式，讓使用者能透過介面拉桿即時調整 11 項化學成分，獲得模型對於紅酒品質的即時評分預測。

## 📊 CRISP-DM 流程摘要

1. **Business Understanding (商業理解)**: 自動化紅酒品質預測，協助釀酒師洞察關鍵化學成分並優化釀造配方。
2. **Data Understanding (資料理解)**: 使用 Kaggle 上的 [Wine Quality Dataset (yasserh)](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)，包含化學自變數與品質目標變數。透過相關係數熱力圖 (Heatmap) 進行了探索性資料分析 (EDA)。
3. **Data Preparation (資料準備)**: 將資料以 80/20 比例分割為訓練集與測試集，並透過 `StandardScaler` 進行特徵縮放。
4. **Modeling (建立模型與特徵選擇)**: 採用「向後淘汰法 (Backward Elimination)」與「RFE」進行特徵篩選，剔除冗餘特徵，防止過度擬合並精簡模型架構。
5. **Evaluation (模型評估)**: 
   * **MSE**: `0.3913`
   * **RMSE**: `0.6255`
   * **MAE**: `0.5043`
   * **$R^2$**: `0.4013`
6. **Deployment (模型部署)**: 輸出 `wine_quality_model.pkl`，並成功使用 Streamlit 封裝成互動式前端網頁部署至雲端。

## 🚀 如何在本地端執行

**1. 安裝所需套件**
```bash
pip install -r requirements.txt
```

**2. 執行靜態分析與圖表輸出 (Option A)**
```bash
python static_analysis.py
```
*(執行後會在 `output/` 資料夾生成模型檔案與預測圖表)*

**3. 啟動 Streamlit 互動式 Web App (Option B - 主程式)**
```bash
streamlit run 4112056003_hw3.py
```
*(網頁啟動後將在瀏覽器 `http://localhost:8501/` 開啟)*
