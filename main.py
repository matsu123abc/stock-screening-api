from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Any
import os
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime
from azure.storage.blob import BlobServiceClient
import io
import json

app = FastAPI()

# ============================================================
# ① ScreeningRequest
# ============================================================
class ScreeningRequest(BaseModel):
    symbols: List[str]

# ============================================================
# ② BlobCSVRequest
# ============================================================
class BlobCSVRequest(BaseModel):
    blob_filename: str


# ============================================================
# /api/screening（一次スクリーニング）
# ============================================================
@app.post("/api/screening")
async def screening(body: ScreeningRequest):
    try:
        results = []

        for symbol in body.symbols:
            try:
                df = yf.Ticker(symbol).history(period="6mo")
                if df.empty:
                    continue

                df["EMA20"] = df["Close"].ewm(span=20).mean()
                df["EMA50"] = df["Close"].ewm(span=50).mean()
                df["EMA200"] = df["Close"].ewm(span=200).mean()

                df["H-L"] = df["High"] - df["Low"]
                df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
                df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
                df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
                df["ATR"] = df["TR"].rolling(window=14).mean()

                latest = df.iloc[-1]

                results.append({
                    "symbol": symbol,
                    "close": float(latest["Close"]),
                    "ema20": float(latest["EMA20"]),
                    "ema50": float(latest["EMA50"]),
                    "ema200": float(latest["EMA200"]),
                    "atr": float(latest["ATR"]),
                })

            except Exception:
                logging.exception(f"Error processing {symbol}")

        return {"results": results}

    except Exception as e:
        logging.exception("screening error")
        return {"error": str(e)}


# ============================================================
# /api/screening_from_blob（①-B BLOB CSV 実行）
# ============================================================
@app.post("/api/screening_from_blob")
async def screening_from_blob(body: BlobCSVRequest):
    try:
        connect_str = os.getenv("AzureWebJobsStorage")
        blob_service = BlobServiceClient.from_connection_string(connect_str)

        blob_container = "block-data"
        blob_name = body.blob_filename

        blob_client = blob_service.get_blob_client(
            container=blob_container,
            blob=blob_name
        )

        csv_text = blob_client.download_blob().readall().decode("utf-8")
        df_csv = pd.read_csv(io.StringIO(csv_text))

        required_cols = ["コード", "銘柄名", "市場"]
        for col in required_cols:
            if col not in df_csv.columns:
                return {"error": f"CSV に '{col}' 列がありません"}

        symbols = [f"{code}.T" for code in df_csv["コード"]]

        screening_request = ScreeningRequest(symbols=symbols)
        screening_result = await screening(screening_request)
        results = screening_result["results"]

        result_container = os.getenv("RESULT_CONTAINER", "screening-results")
        today = datetime.now().strftime("%Y-%m-%d")
        output_blob_name = f"{today}/screening_{today}.json"

        result_blob = blob_service.get_blob_client(
            container=result_container,
            blob=output_blob_name
        )

        json_text = json.dumps(results, ensure_ascii=False, indent=2)
        result_blob.upload_blob(json_text, overwrite=True)

        return {
            "saved_to": output_blob_name,
            "results": results
        }

    except Exception as e:
        logging.exception("screening_from_blob error")
        return {"error": str(e)}


# ============================================================
# /（UI を返す） ← HTML + JS を main.py に埋め込み
# ============================================================
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>Stock AI Screening Viewer</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 6px; }
  </style>
</head>
<body>

<h2>Stock AI Screening Viewer</h2>

<h3>①-B BLOB の CSV を選択して実行</h3>

<select id="blobCsvList">
  <option value="prime_001-050.csv">prime_001-050.csv</option>
  <option value="prime_051-100.csv">prime_051-100.csv</option>
  <option value="prime_101-150.csv">prime_101-150.csv</option>
  <option value="prime_151-200.csv">prime_151-200.csv</option>
  <option value="prime_201-250.csv">prime_201-250.csv</option>
  <option value="prime_251-300.csv">prime_251-300.csv</option>
  <option value="prime_301-350.csv">prime_301-350.csv</option>
  <option value="prime_351-400.csv">prime_351-400.csv</option>
  <option value="prime_401-450.csv">prime_401-450.csv</option>
  <option value="prime_451-500.csv">prime_451-500.csv</option>
  <option value="prime_501-550.csv">prime_501-550.csv</option>
  <option value="prime_551-600.csv">prime_551-600.csv</option>
  <option value="prime_601-650.csv">prime_601-650.csv</option>
  <option value="prime_651-700.csv">prime_651-700.csv</option>
  <option value="prime_701-750.csv">prime_701-750.csv</option>
  <option value="prime_751-800.csv">prime_751-800.csv</option>
  <option value="prime_801-850.csv">prime_801-850.csv</option>
  <option value="prime_851-900.csv">prime_851-900.csv</option>
  <option value="prime_901-950.csv">prime_901-950.csv</option>
  <option value="prime_951-1000.csv">prime_951-1000.csv</option>
  <option value="prime_1001-1050.csv">prime_1001-1050.csv</option>
  <option value="prime_1051-1100.csv">prime_1051-1100.csv</option>
  <option value="prime_1101-1150.csv">prime_1101-1150.csv</option>
  <option value="prime_1151-1200.csv">prime_1151-1200.csv</option>
  <option value="prime_1201-1250.csv">prime_1201-1250.csv</option>
  <option value="prime_1251-1300.csv">prime_1251-1300.csv</option>
  <option value="prime_1301-1350.csv">prime_1301-1350.csv</option>
  <option value="prime_1351-1400.csv">prime_1351-1400.csv</option>
  <option value="prime_1401-1450.csv">prime_1401-1450.csv</option>
  <option value="prime_1451-1500.csv">prime_1451-1500.csv</option>
  <option value="prime_1501-1550.csv">prime_1501-1550.csv</option>
  <option value="prime_1551-1600.csv">prime_1551-1600.csv</option>
</select>

<button onclick="runBlobCSV()">BLOB CSV で実行</button>

<hr>

<h3>② 結果表示</h3>
<div id="loading"></div>
<div id="result"></div>

<script>
const API_BASE = "";
const RESULT_BLOB_BASE = "https://stockai20260214.blob.core.windows.net/results/";

async function runBlobCSV() {
  const filename = document.getElementById("blobCsvList").value;

  document.getElementById("loading").innerText =
    `BLOB CSV (${filename}) を実行中…`;

  try {
    const response = await fetch(
      `/api/screening_from_blob`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ blob_filename: filename })
      }
    );

    const result = await response.json();

    if (!result.saved_to) {
      document.getElementById("loading").innerText = "エラー";
      alert(JSON.stringify(result));
      return;
    }

    document.getElementById("loading").innerText = "完了！";
    loadResultJson(result.saved_to);

  } catch (e) {
    document.getElementById("loading").innerText = "通信エラー";
  }
}

async function loadResultJson(path) {
  const url = RESULT_BLOB_BASE + path;
  const res = await fetch(url);
  const json = await res.json();

  renderTable(json);
}

function renderTable(data) {
  if (!data || data.length === 0) {
    document.getElementById("result").innerHTML = "<p>結果なし</p>";
    return;
  }

  let html = "<table><tr><th>symbol</th><th>close</th><th>ema20</th><th>ema50</th><th>ema200</th><th>atr</th></tr>";

  for (const r of data) {
    html += `<tr>
      <td>${r.symbol}</td>
      <td>${r.close}</td>
      <td>${r.ema20}</td>
      <td>${r.ema50}</td>
      <td>${r.ema200}</td>
      <td>${r.atr}</td>
    </tr>`;
  }

  html += "</table>";
  document.getElementById("result").innerHTML = html;
}
</script>

</body>
</html>
"""


# ============================================================
# 完了
# ============================================================
