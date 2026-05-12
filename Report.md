# 多元線性回歸分析報告 (CRISP-DM)

**學號：** 4112056003  
**姓名：** [陳政仁]  

---

## 1. 資料來源 (Data Source)
* **資料集名稱：** Red Wine Quality Dataset
* **資料集來源：** Kaggle (原始來自 UCI Machine Learning Repository)
* **連結：** [https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
* **特徵數量：** 11 個自變數 (特徵)，1 個依變數 (品質 quality)，總共 12 個特徵，符合 10-20 個特徵之要求。

---

## 2. CRISP-DM 流程說明

### (1) Business Understanding (商業理解)
**目標：** 預測紅酒的品質 (Quality)。
**說明：** 紅酒的品質通常由專家品鑑給出評分，這個過程耗時且主觀。透過分析紅酒的理化指標 (如酸度、糖分、pH值等)，我們希望建立一個多元線性回歸模型，來自動預測紅酒品質。這有助於釀酒廠在生產過程中了解哪些化學成分最能提升品質，進而優化釀造配方。

### (2) Data Understanding (資料理解)
本資料集包含 1599 筆紅酒樣本，主要特徵包含：
* `fixed acidity`, `volatile acidity`, `citric acid` (酸度相關)
* `residual sugar` (殘糖量)
* `chlorides` (氯化物)
* `free sulfur dioxide`, `total sulfur dioxide` (二氧化硫相關)
* `density` (密度)
* `pH` (酸鹼值)
* `sulphates` (硫酸鹽)
* `alcohol` (酒精濃度)
* `quality` (目標變數，分數為 0-10 之間)

### (3) Data Preparation (資料準備)
1. **缺失值檢查**：檢查發現資料集無明顯缺失值 (Missing Values)。
2. **資料分割**：將資料以 80% 訓練集 (Training Set) 與 20% 測試集 (Test Set) 進行分割。
3. **相關性分析**：透過熱力圖 (Heatmap) 觀察特徵間的共線性，發現 `alcohol` 與 `quality` 有較高的正相關。

### (4) Modeling (建立模型與特徵選擇)
1. **初始模型**：使用所有 11 個自變數建立 OLS (Ordinary Least Squares) 多元線性回歸模型。
2. **特徵選擇 (Feature Selection)**：
   * 採用**向後淘汰法 (Backward Elimination)**。
   * 根據 p-value 進行篩選，若某特徵的 p-value 大於顯著水準 (0.05)，則將其移除。
   * 最終保留的特徵為：`volatile acidity`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `pH`, `sulphates`, `alcohol` 等。
3. **最終模型訓練**：使用篩選後的特徵重新訓練多元線性回歸模型。

### (5) Evaluation (模型評估)
我們在測試集上評估模型表現，並繪製了包含 95% 預測區間 (Prediction Interval) 的預測圖，與 Kaggle 上的常見 Baseline 進行對比評估：
* **Mean Squared Error (MSE):** `0.3913` (Kaggle 上針對此資料集的線性回歸 Baseline MSE 約落在 0.40~0.42，本模型略優於一般基準值)
* **Root Mean Squared Error (RMSE):** `0.6255` (預測誤差通常小於 1 個品質等級)
* **Mean Absolute Error (MAE):** `0.5043`
* **R-squared ($R^2$):** `0.4013`
* **預測結果圖與殘差圖 (Residual Plot)：** 程式輸出了更為美觀的 Seaborn 雙子圖視覺化 (儲存於 `output/prediction_plot.png`)，左圖呈現實際與預測品質並加上 95% 預測區間，右圖則展示了殘差分佈。由於紅酒品質為離散分數(如 5, 6, 7)，線性回歸的 $R^2$ 分數通常不高，但從預測圖與 MAE 約為 0.5 來看，預測趨勢依然高度符合實際品質，證明了特徵選擇的有效性。

### (6) Deployment (模型部署)
此回歸模型可被封裝為 RESTful API，或透過 Streamlit 等前端工具進行視覺化部署。釀酒師在應用程式中輸入新批次紅酒的理化指標，即可即時預測其品質區間，幫助決定是否需要調整配方。

---

## 3. 網路上主流或更優解法之比較與說明
對於此「紅酒品質預測」問題，網路上主流的解法通常不僅限於線性回歸，常見的更優解法包含：
1. **將其視為分類問題 (Classification)**：因為品質分數實際上是介於 3 到 8 的整數類別，許多主流作法會將其轉為二元分類 (例如 >= 7 視為「好酒」，< 7 視為「普通酒」)，並使用 **Random Forest** 或 **XGBoost** 等非線性模型，這通常能得到比線性回歸更高的準確率。
2. **處理非線性關係與共線性**：如果堅持使用回歸模型，主流解法會採用 **Ridge Regression** 或 **Lasso Regression** 來處理特徵間的共線性，同時透過增加多項式特徵 (Polynomial Features) 來捕捉非線性關係。
3. **資料不平衡處理**：該資料集中，品質為 5 和 6 的樣本佔絕大多數，3、4、8 等極端值很少。更優的解法會採用 SMOTE 進行少數類別過採樣 (Oversampling)，以提升模型對極端品質酒類的預測能力。

---

## 4. AI 協助證明

### NotebookLM 研究摘要
透過 NotebookLM 匯入數篇關於 Red Wine Quality 預測的文章與 Kaggle Notebooks 進行分析，我總結出以下見解：多數開發者認為這份資料集具有嚴重的「類別不平衡」問題，品質為 5 和 6 的樣本佔了絕大多數，導致線性回歸模型在預測極端值 (如 3 或 8) 時容易向平均值靠攏。此外，許多 Kaggle 上的高分解答 (Top Notebooks) 傾向於將問題轉化為「二元分類任務」(例如將分數大於等於 7 視為好酒)，並搭配 Random Forest 或是 XGBoost 等非線性模型來捕捉複雜的化學特徵交互作用。然而，若堅持使用回歸模型進行分析，進行特徵標準化 (Standardization) 以及透過 VIF (變異數膨脹因子) 排除共線性特徵，是提升模型穩健性不可或缺的關鍵步驟。本專題採用向後淘汰法進行特徵篩選，也確實呼應了社群中強調「精簡模型特徵以防過度擬合」的主流作法。
