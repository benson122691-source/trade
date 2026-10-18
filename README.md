# 跨資產投資模擬器

學生專題原型：比較台股 ETF、美股 ETF、比特幣與黃金 ETF 的歷史投資模擬，並以多模型比較產生未來報酬情境估算。

## 啟動

需要 Python 3.10 以上與網路連線。

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 第一版功能

- 0050.TW、SPY、BTC-USD、GLD 四個代表標的。
- 海外資產以 USD/TWD 匯率換算為新台幣。
- 一次投入與定期定額歷史模擬，並顯示累積報酬、年化波動與最大回撤。
- Random Forest 與 Ridge 在訓練資料內的時間順序驗證段選模，再以保留測試段評估，並和簡單基準比較誤差。
- Streamlit 互動儀表板顯示歷史資產價值走勢、各模型誤差、期末金額中央估算與經驗情境區間圖。

## 機器學習模型

模型程式獨立放在 `model.py`，Streamlit 介面也從這個模組呼叫。它以新台幣計價的調整後價格建立特徵，預測未來 H 筆有效價格觀測的對數報酬，再轉成一般報酬率。不同資產的觀測日曆可能不同。

特徵包括 5、20、60 日動能，20、60 日波動度，以及價格相對 20、60 日均線的差距。候選模型包括 Random Forest Regressor（250 棵樹、`min_samples_leaf=5`、`max_features=0.8`）和標準化後的 Ridge Regression（`alpha=1.0`）。

模型以前 80% 樣本作開發資料，後 20% 樣本保留作最終測試。開發資料內再做時間順序驗證以選擇模型，訓練區間末端會保留 H 筆間隔，避免標籤報酬期間和驗證或測試區間重疊。回傳的測試 MAE、RMSE 以對數報酬為單位，並和零報酬及訓練期平均報酬基準比較。上下情境是測試期實際報酬的第 10 與第 90 百分位，**不是信賴區間或收益保證**。

獨立使用模型時，準備有日期索引的 pandas Series：

```python
import pandas as pd
from model import forecast_return

prices = pd.read_csv("prices.csv", parse_dates=["Date"])
prices = prices.set_index("Date")["Close"]

result = forecast_return(prices, horizon=20)
print(result.to_dict())
```

`prices.csv` 至少要有 `Date` 和 `Close` 欄位，`Close` 必須是正數價格。資料不足時，函式會回傳清楚的錯誤訊息。長預測期間需要較長歷史資料，因為模型還要保留測試區間和訓練／測試隔離區。

## 資料來源與限制

價格及 USD/TWD 匯率由 Yahoo Finance 透過 `yfinance` 套件取得。海外資產與匯率資料可能缺漏或修訂。歷史回測採調整後收盤價，但尚未納入交易手續費、稅負、滑價、ETF 配息稅務或匯兌成本。模型區間不是信賴區間，也不代表未來結果；未來情境以總投入金額在目前一次投入估算，即使歷史模擬選擇定期定額亦然。

目前模型輸出是專題原型，尚未證明能穩定優於簡單基準。H 日報酬標籤在每日預測起點之間會重疊，因此 holdout 誤差具有時間相依性，應以描述性指標解讀。正式研究還要記錄資料下載日期，並做不同時間區間的穩健性檢查。

本專題用於教育與方法比較，不是投資建議。使用前請確認資料來源的使用條款與標的代碼。

