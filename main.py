from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Any
import os
import logging
import yfinance as yf
from openai import AzureOpenAI
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import io
import json

app = FastAPI()

def safe_float(x):
    try:
        if hasattr(x, "iloc"):
            return float(x.iloc[0])
        return float(x)
    except Exception:
        return None

def ema(series, span):
    return series.ewm(span=span).mean()

def calc_atr(df, window=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def gpt_score(symbol, name, price, market_cap,
              drop_rate, reversal_rate, reversal_strength,
              ema20, ema50, slope_ema20,
              atr, volume, vol_ma20, volume_ratio):

    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    prompt = f"""
あなたは短期トレードの専門家です。
以下の銘柄について、短期的な期待度を 0〜100 点でスコアリングし、
さらに「買い」「様子見」「避ける」のいずれかで売買判断を行い、
最後に **短期トレード視点での根拠を3〜5行で詳しくコメント** してください。

返答は必ず次の JSON 形式のみで返してください。
JSON の前後に説明文や文章を一切付けないこと。

{{
  "score": 数値,
  "judgement": "買い/様子見/避ける",
  "comment": "短期トレード視点での根拠を3〜5行で詳しく記述"
}}

銘柄コード: {symbol}
株価: {price}
時価総額(億円): {market_cap}

下落率: {drop_rate:.2f}%
反転率: {reversal_rate:.2f}%
反転強度: {reversal_strength:.2f}

EMA20: {ema20}
EMA50: {ema50}
EMA20の傾き: {slope_ema20:.2f}

出来高: {volume}
出来高20日平均: {vol_ma20}
出来高急増率: {volume_ratio:.2f}

ATR(14): {atr}
"""

    try:
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        raw = res.choices[0].message.content.strip()

        # --- JSON 抽出（安全版） ---
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        json_text = raw[json_start:json_end]

        # --- 余計な文字を除去 ---
        json_text = json_text.replace("```json", "").replace("```", "").strip()

        return json.loads(json_text)

    except Exception as e:
        return {
            "score": 0,
            "judgement": "エラー",
            "comment": f"GPTエラー: {str(e)}"
        }


def process_symbol(symbol, company_name, market, log, python_condition=None):
    try:
        df = yf.download(symbol, period="180d", interval="1d")

        if df is None or df.empty:
            return None

        try:
            ticker = yf.Ticker(symbol)
            fi = getattr(ticker, "fast_info", None)
            mc = None

            if fi is not None:
                mc = fi.get("market_cap", None)

            if mc is None:
                info = ticker.info
                mc = info.get("marketCap", None)

            market_cap = int(mc / 100000000) if mc else None
        except:
            market_cap = None

        df["EMA20"] = ema(df["Close"], 20)
        df["EMA50"] = ema(df["Close"], 50)
        df["EMA200"] = ema(df["Close"], 200)
        df["ATR"] = calc_atr(df)
        df["vol_ma20"] = df["Volume"].rolling(window=20).mean()

        if len(df) < 25:
            return None

        ema20_now = df["EMA20"].iloc[-1]
        ema20_prev = df["EMA20"].iloc[-5]

        slope_prev = safe_float(df["EMA20"].iloc[-6] - df["EMA20"].iloc[-11])
        slope_now = safe_float(ema20_now - ema20_prev)

        if not (slope_prev < 0 and slope_now > 0):
            return None

        first_reversal_date = None
        last_reversal_date = None

        for i in range(11, len(df)):
            slope_prev_i = safe_float(df["EMA20"].iloc[i-6] - df["EMA20"].iloc[i-11])
            slope_now_i  = safe_float(df["EMA20"].iloc[i]   - df["EMA20"].iloc[i-5])

            if slope_prev_i < 0 and slope_now_i > 0:
                if first_reversal_date is None:
                    first_reversal_date = df.index[i].strftime("%Y-%m-%d")
                last_reversal_date = df.index[i].strftime("%Y-%m-%d")

        recent = df.tail(120)
        peak_price = safe_float(recent["High"].max())
        bottom_price = safe_float(recent["Low"].min())

        latest = df.iloc[-1]
        close_price = safe_float(latest["Close"])

        drop_rate = safe_float((bottom_price / peak_price - 1) * 100) if peak_price else None
        reversal_rate = safe_float((close_price / bottom_price - 1) * 100) if bottom_price else None

        if drop_rate and drop_rate != 0:
            reversal_strength = safe_float(reversal_rate / abs(drop_rate))
        else:
            reversal_strength = None

        ema20 = safe_float(latest["EMA20"])
        ema50 = safe_float(latest["EMA50"])
        ema200 = safe_float(latest["EMA200"])
        atr = safe_float(latest["ATR"])

        vol_ma20 = safe_float(latest["vol_ma20"])
        volume = safe_float(latest["Volume"])
        volume_ratio = volume / vol_ma20 if vol_ma20 and vol_ma20 > 0 else 0

        short_score = (
            (reversal_strength or 0) * 0.4 +
            (volume_ratio or 0) * 0.2 +
            (slope_now or 0) * 0.2 +
            (drop_rate or 0) * 0.1 -
            (atr or 0) * 0.1
        )

        gpt = gpt_score(
            symbol, company_name, close_price, market_cap,
            drop_rate, reversal_rate, reversal_strength,
            ema20, ema50, slope_now,
            atr, volume, vol_ma20, volume_ratio
        )

        return {
            "symbol": symbol,
            "company_name": company_name,
            "market": market,
            "close": close_price,
            "EMA20": ema20,
            "EMA50": ema50,
            "EMA200": ema200,
            "ATR": atr,
            "drop_rate": drop_rate,
            "reversal_rate": reversal_rate,
            "reversal_strength": reversal_strength,
            "market_cap": market_cap,
            "slope_ema20": slope_now,
            "volume_ratio": volume_ratio,
            "first_reversal_date": first_reversal_date,
            "last_reversal_date": last_reversal_date,
            "short_score": short_score,
            "gpt_score": gpt.get("score"),
            "gpt_judgement": gpt.get("judgement"),
            "gpt_comment": gpt.get("comment"),

            # 二次スクリーニング指標
            "drop_from_high_pct": drop_rate,
            "rebound_from_low_pct": reversal_rate,
            "ema20_vs_ema50": safe_float(ema20 - ema50),
            "ema50_vs_ema200": safe_float(ema50 - ema200),
            "price_vs_ema20_pct": safe_float((close_price / ema20 - 1) * 100) if ema20 else None,
            "vol_vs_ma20": volume_ratio,
            "atr_ratio": safe_float(atr / close_price) if close_price else None,

            "passed_python_condition": True
        }

    except Exception:
        return None

class ScreeningFromBlobRequest(BaseModel):
    blob_filename: str

@app.post("/api/screening_from_blob")
async def screening_from_blob(req: ScreeningFromBlobRequest):

    blob_filename = req.blob_filename

    blob_service = BlobServiceClient.from_connection_string(
        os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    )
    container = blob_service.get_container_client("results")

    input_blob = container.get_blob_client(blob_filename)
    csv_bytes = input_blob.download_blob().readall()
    df = pd.read_csv(io.BytesIO(csv_bytes))

    results = []

    for i, row in df.iterrows():
        symbol = row["symbol"]
        company_name = row.get("company_name", "")
        market = row.get("market", "")

        r = process_symbol(
            symbol=symbol,
            company_name=company_name,
            market=market,
            log=lambda msg: None
        )

        if r:
            results.append(r)

    result_blob_path = f"{blob_filename.replace('.csv', '')}_result.json"
    result_blob = container.get_blob_client(result_blob_path)
    result_blob.upload_blob(
        json.dumps(results, ensure_ascii=False),
        overwrite=True
    )

    return {
        "saved_to": result_blob_path
    }

# =====  PART2 =====
from typing import List, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os, io, json, logging
from datetime import datetime
import pandas as pd
import yfinance as yf
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI

app = FastAPI()

# =========================
# ScreeningRequest
# =========================
class ScreeningRequest(BaseModel):
    symbols: List[str]


# =========================
# ★ screening()：ログなし版
# =========================
@app.post("/api/screening")
async def screening(body: ScreeningRequest):
    try:
        symbols = body.symbols
        results = []

        for symbol in symbols:
            try:
                r = process_symbol(
                    symbol=symbol,
                    company_name="",
                    market="",
                    log=lambda msg: None,  # ログ無効化
                    python_condition=None
                )
                if r is not None:
                    results.append(r)
            except Exception as e:
                print(f"[ERROR] screening {symbol}: {e}")

        return {"results": results}

    except Exception as e:
        logging.exception("screening error")
        return {"error": str(e)}


# =========================
# BlobCSVRequest
# =========================
class BlobCSVRequest(BaseModel):
    blob_filename: str


# =========================
# ★ screening_from_blob（ログなし高速版）
# =========================
@app.post("/api/screening_from_blob")
async def screening_from_blob(body: BlobCSVRequest):

    try:
        # Azure Blob 接続
        connect_str = os.getenv("AzureWebJobsStorage")
        blob_service = BlobServiceClient.from_connection_string(connect_str)

        # 入力 CSV のコンテナ
        input_container = blob_service.get_container_client("block-data")

        # 結果保存コンテナ（results に統一）
        result_container = blob_service.get_container_client("results")

        blob_name = body.blob_filename

        # CSV 読み込み
        blob_client = input_container.get_blob_client(blob_name)
        csv_text = blob_client.download_blob().readall().decode("utf-8")
        df_csv = pd.read_csv(io.StringIO(csv_text))

        required_cols = ["コード", "銘柄名", "市場"]
        for col in required_cols:
            if col not in df_csv.columns:
                return {
                    "saved_to": None,
                    "error": f"CSV に '{col}' 列がありません"
                }

        # 銘柄コードを .T に変換
        symbols = [f"{code}.T" for code in df_csv["コード"]]

        results = []

        for i, symbol in enumerate(symbols):
            r = process_symbol(
                symbol=symbol,
                company_name=df_csv.loc[i, "銘柄名"],
                market=df_csv.loc[i, "市場"],
                log=lambda msg: None  # ログ無効化
            )

            if r:
                results.append(r)

        # 結果 JSON 保存
        today = datetime.now().strftime("%Y-%m-%d")
        output_blob_name = f"{today}/screening_{today}.json"

        result_blob = result_container.get_blob_client(output_blob_name)
        result_blob.upload_blob(
            json.dumps(results, ensure_ascii=False, indent=2),
            overwrite=True
        )

        return {
            "saved_to": output_blob_name
        }

    except Exception as e:
        logging.exception("screening_from_blob error")
        return {
            "saved_to": None,
            "error": str(e)
        }


# =========================
# explain_symbol（企業説明）
# =========================
@app.get("/api/explain_symbol")
async def explain_symbol(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        company_name = info.get("shortName") or info.get("longName") or symbol
        summary = info.get("longBusinessSummary")

        if not summary:
            summary = "企業情報（longBusinessSummary）が取得できませんでした。"

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        prompt = f"""
あなたはプロの株式アナリストです。
以下の企業情報をもとに、この企業が「何をしている会社か」「主力事業」「強み」「特徴」を
投資家向けに分かりやすく説明してください。

【企業名】
{company_name}

【企業情報（Yahoo Finance）】
{summary}
"""

        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        explanation = res.choices[0].message.content.strip()

        return JSONResponse(
            {
                "symbol": symbol,
                "company": company_name,
                "explanation": explanation
            },
            status_code=200
        )

    except Exception as e:
        logging.exception("explain_symbol error")
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# second_screening（二次スクリーニング）
# =========================
class SecondScreeningRequest(BaseModel):
    results: List[Any]

@app.post("/second_screening")
async def second_screening(body: SecondScreeningRequest):
    try:
        results = body.results

        filtered = []
        for r in results:
            if (
                (r.get("drop_from_high_pct") or 0) < -20 and
                (r.get("rebound_from_low_pct") or 0) > 25 and
                (r.get("ema20_vs_ema50") or 0) > 5.0 and
                (r.get("ema50_vs_ema200") or 0) > 10.0 and
                (r.get("price_vs_ema20_pct") or 0) > 2 and
                (r.get("vol_vs_ma20") or 0) > 1.0 and
                (r.get("atr_ratio") or 0) > 1
            ):
                filtered.append(r)

        return JSONResponse(
            {"second_screening": filtered, "count": len(filtered)},
            status_code=200
        )

    except Exception as e:
        logging.exception("second_screening error")
        return JSONResponse({"error": str(e)}, status_code=500)


# =========================
# third_screening（三次スクリーニング）
# =========================
@app.post("/third_screening")
async def third_screening(body: dict):

    try:
        symbols = body.get("symbols", [])

        if not symbols:
            return JSONResponse(
                {"error": "symbols が空です"},
                status_code=400
            )

        results = []

        for sym in symbols:
            ticker = yf.Ticker(sym)
            info = ticker.info

            fundamentals = {
                "売上高": info.get("totalRevenue"),
                "営業利益率": info.get("operatingMargins"),
                "純利益率": info.get("profitMargins"),
                "EPS": info.get("trailingEps"),
                "PER": info.get("trailingPE"),
                "PBR": info.get("priceToBook"),
                "ROE": info.get("returnOnEquity"),
                "売上成長率": info.get("revenueGrowth"),
                "利益成長率": info.get("earningsGrowth"),
                "フリーCF": info.get("freeCashflow"),
                "負債総額": info.get("totalDebt"),
                "現金": info.get("totalCash"),
            }

            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )

            prompt = f"""
あなたはプロの株式アナリストです。
以下の企業業績データをもとに、企業の強み・弱み・リスク・総合評価を簡潔に説明してください。

銘柄: {sym}
業績データ:
{fundamentals}
"""

            ai_res = client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            analysis = ai_res.choices[0].message.content.strip()

            results.append({
                "symbol": sym,
                "fundamentals": fundamentals,
                "analysis": analysis
            })

        return JSONResponse(
            {"results": results},
            status_code=200
        )

    except Exception as e:
        logging.exception("third_screening error")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

# ===== PART3   =====
@app.get("/", response_class=HTMLResponse)
def index():
    return r"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stock AI Screening Viewer</title>
  <style>
    body { font-family: sans-serif; margin: 20px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
    th, td { border: 1px solid #ccc; padding: 6px; }
    th { background: #eee; }
    .chart-link { font-size: 20px; text-decoration: none; }
    @media screen and (max-width: 600px) {
      body { font-size: 18px; padding: 10px; }
      select, button { width: 100%; font-size: 20px; padding: 14px; margin-top: 10px; }
      h2, h3 { font-size: 22px; margin-top: 24px; }
      table { font-size: 16px; width: 100%; display: block; overflow-x: auto; white-space: nowrap; }
      th, td { padding: 10px; }
    }
  </style>
</head>
<body>

<h2>Stock AI Screening Viewer</h2>

<h3>①-B BLOB の CSV を選択して実行</h3>

<select id="blobCsvList" onchange="loadCsvSymbols(this.value)">
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

<h3>選択したCSVの銘柄一覧</h3>
<div id="csvSymbolsBox">
  <button id="csvToggleBtn" type="button" onclick="toggleCsvSymbols()">銘柄一覧を表示</button>
  <div id="csvSymbols" style="display:none; margin-top:10px;"></div>
</div>

<h3>② 結果表示</h3>
<div id="loading"></div>

<h3>主要データ</h3>
<div id="mainTable"></div>

<h3>AI コメント一覧</h3>
<div id="aiTable"></div>

<h3>②-B 二次スクリーニング結果</h3>
<button onclick="runSecondScreening()">二次スクリーニングを実行</button>
<div id="secondTable"></div>

<h3>②-C 二次スクリーニング指標一覧</h3>
<div id="indicatorTable"></div>

<h3>③ 三次スクリーニング（企業業績 × AI 分析）</h3>
<button onclick="runThirdScreening()">三次スクリーニングを実行</button>
<div id="thirdTable"></div>

<h3>4. 結果をHTMLとして保存</h3>
<button onclick="downloadHtml()">この画面をHTML保存</button>

<script>
/* Debug marker */
console.log("PART3 script start");

/* 基本定数 */
const RESULT_BLOB_BASE = "https://stockai20260214.blob.core.windows.net/results/";
let latestResults = [];
let latestSecond = [];

/* トグル表示 */
function toggleCsvSymbols() {
  var box = document.getElementById("csvSymbols");
  var btn = document.getElementById("csvToggleBtn");
  if (!box) return;
  if (box.style.display === "none" || box.style.display === "") {
    box.style.display = "block";
    if (btn) btn.textContent = "銘柄一覧を閉じる";
  } else {
    box.style.display = "none";
    if (btn) btn.textContent = "銘柄一覧を表示";
  }
}

/* CSV を読み込み、銘柄一覧を表示する */
async function loadCsvSymbols(filename) {
  console.log("loadCsvSymbols called:", filename);
  var target = document.getElementById("csvSymbols");
  if (!filename) {
    if (target) target.innerHTML = "<p>ファイルが選択されていません。</p>";
    return;
  }

  var url = "https://stockai20260214.blob.core.windows.net/block-data/" + filename;

  try {
    var res = await fetch(url);
    if (!res.ok) throw new Error("HTTP " + res.status);
    var text = await res.text();

    // BOM を除去（先頭が BOM の場合）
    if (text && text.length > 0 && text.charCodeAt(0) === 0xFEFF) {
      text = text.slice(1);
    }

    // 行分割（CR を手動で取り除く）
    var rawLines = text.split("\n");
    var lines = [];
    for (var i = 0; i < rawLines.length; i++) {
      var ln = rawLines[i];
      if (ln.length > 0 && ln.charAt(ln.length - 1) === '\r') {
        ln = ln.slice(0, -1);
      }
      ln = ln.trim();
      if (ln.length > 0) lines.push(ln);
    }

    if (lines.length === 0) {
      if (target) target.innerHTML = "<p>CSV が空です。</p>";
      return;
    }

    // 簡易 CSV パーサ（引用符対応）
    function parseCsvLine(line) {
      var cols = [];
      var cur = "";
      var inQuotes = false;
      for (var j = 0; j < line.length; j++) {
        var ch = line.charAt(j);
        if (ch === '"') {
          if (inQuotes && j + 1 < line.length && line.charAt(j+1) === '"') {
            cur += '"';
            j++;
          } else {
            inQuotes = !inQuotes;
          }
        } else if (ch === ',' && !inQuotes) {
          cols.push(cur);
          cur = "";
        } else {
          cur += ch;
        }
      }
      cols.push(cur);
      // trim each
      for (var k = 0; k < cols.length; k++) cols[k] = cols[k].trim();
      return cols;
    }

    var headers = parseCsvLine(lines[0]);
    // ヘッダー名を柔軟に検出
    var codeIndex = -1, nameIndex = -1, marketIndex = -1;
    for (var h = 0; h < headers.length; h++) {
      var hh = headers[h].replace(/\uFEFF/g, "").trim().toLowerCase();
      if (hh === "コード" || hh === "code" || hh === "symbol") codeIndex = h;
      if (hh === "銘柄名" || hh === "名称" || hh === "name") nameIndex = h;
      if (hh === "市場" || hh === "market") marketIndex = h;
    }

    if (codeIndex === -1 || nameIndex === -1) {
      if (target) target.innerHTML = "<p>CSV ヘッダーに「コード」「銘柄名」が見つかりません。</p>";
      return;
    }

    var html = "<p>銘柄数: " + (lines.length - 1) + " 件</p><ul>";
    for (var i2 = 1; i2 < lines.length; i2++) {
      var cols = parseCsvLine(lines[i2]);
      var code = cols[codeIndex] || "";
      var name = cols[nameIndex] || "";
      var market = (marketIndex !== -1) ? (cols[marketIndex] || "") : "";
      var link = code ? "https://finance.yahoo.co.jp/quote/" + encodeURIComponent(code) : "#";
      html += "<li><a href=\"" + link + "\" target=\"_blank\" rel=\"noopener noreferrer\">" + escapeHtml(code) + " : " + escapeHtml(name) + "</a>";
      if (market) html += " （" + escapeHtml(market) + "）";
      html += "</li>";
    }
    html += "</ul>";

    if (target) target.innerHTML = html;

    // 自動展開
    if (target && (target.style.display === "none" || target.style.display === "")) {
      target.style.display = "block";
      var btn = document.getElementById("csvToggleBtn");
      if (btn) btn.textContent = "銘柄一覧を閉じる";
    }

  } catch (e) {
    console.error("loadCsvSymbols error:", e);
    if (target) target.innerHTML = "<p>CSV 読み込みエラー</p>";
  }
}

/* XSS 対策のための簡易エスケープ */
function escapeHtml(s) {
  if (!s) return "";
  s = String(s);
  s = s.replace(/&/g, "&amp;");
  s = s.replace(/</g, "&lt;");
  s = s.replace(/>/g, "&gt;");
  s = s.replace(/"/g, "&quot;");
  s = s.replace(/'/g, "&#39;");
  return s;
}

/* HTML 保存 */
function downloadHtml() {
  var html = document.documentElement.outerHTML;
  var blob = new Blob([html], { type: "text/html" });
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  var now = new Date();
  var y = now.getFullYear();
  var m = String(now.getMonth() + 1).padStart(2, "0");
  var d = String(now.getDate()).padStart(2, "0");
  var hh = String(now.getHours()).padStart(2, "0");
  var mm = String(now.getMinutes()).padStart(2, "0");
  a.href = url;
  a.download = "stock_screening_" + y + m + d + "_" + hh + mm + ".html";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* BLOB 実行（ダミー呼び出し） */
async function runBlobCSV() {
  var filename = document.getElementById("blobCsvList").value;
  document.getElementById("loading").innerText = "BLOB CSV (" + filename + ") を実行中…";
  try {
    var response = await fetch("/api/screening_from_blob", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ blob_filename: filename })
    });
    var result = await response.json();
    if (!result.saved_to) {
      document.getElementById("loading").innerText = "エラー";
      alert(JSON.stringify(result));
      return;
    }
    document.getElementById("loading").innerText = "完了！";
    loadResultJson(result.saved_to);
  } catch (e) {
    console.error("runBlobCSV error:", e);
    document.getElementById("loading").innerText = "通信エラー";
  }
}

/* 結果 JSON 読み込み */
async function loadResultJson(path) {
  var url = RESULT_BLOB_BASE + path;
  var res = await fetch(url);
  var json = await res.json();
  latestResults = json;
  renderMainTable(json);
  renderAiTable(json);
  renderIndicatorTable(json);
}

/* 一次スクリーニング結果表示 */
function renderMainTable(data) {
  if (!data || data.length === 0) {
    document.getElementById("mainTable").innerHTML = "<p>スクリーニング通過銘柄なし</p>";
    return;
  }
  var html = "<table><tr>"
    + "<th>symbol</th><th>company</th><th>market</th><th>close</th>"
    + "<th>最初の反転日</th><th>最新の反転日</th><th>short_score</th>"
    + "<th>judgement</th><th>chart</th><th>説明</th></tr>";
  for (var i = 0; i < data.length; i++) {
    var r = data[i];
    html += "<tr>"
      + "<td>" + escapeHtml(r.symbol) + "</td>"
      + "<td>" + escapeHtml(r.company_name || "") + "</td>"
      + "<td>" + escapeHtml(r.market || "") + "</td>"
      + "<td>" + escapeHtml(r.close) + "</td>"
      + "<td>" + escapeHtml(r.first_reversal_date || "") + "</td>"
      + "<td>" + escapeHtml(r.last_reversal_date || "") + "</td>"
      + "<td>" + escapeHtml(r.short_score) + "</td>"
      + "<td>" + escapeHtml(r.gpt_judgement) + "</td>"
      + "<td><a class=\"chart-link\" href=\"https://finance.yahoo.co.jp/quote/" + encodeURIComponent(r.symbol) + "\" target=\"_blank\">📈</a></td>"
      + "<td><a href=\"/api/explain_symbol?symbol=" + encodeURIComponent(r.symbol) + "\" target=\"_blank\">説明</a></td>"
      + "</tr>";
  }
  html += "</table>";
  document.getElementById("mainTable").innerHTML = html;
}

/* AI コメント一覧 */
function renderAiTable(data) {
  if (!data || data.length === 0) {
    document.getElementById("aiTable").innerHTML = "<p>AI コメントなし</p>";
    return;
  }
  var html = "<table><tr><th>symbol</th><th>company</th><th>AI コメント</th><th>説明</th></tr>";
  for (var i = 0; i < data.length; i++) {
    var r = data[i];
    html += "<tr>"
      + "<td>" + escapeHtml(r.symbol) + "</td>"
      + "<td>" + escapeHtml(r.company_name || "") + "</td>"
      + "<td>" + escapeHtml(r.gpt_comment || "") + "</td>"
      + "<td><a href=\"/api/explain_symbol?symbol=" + encodeURIComponent(r.symbol) + "\" target=\"_blank\">説明</a></td>"
      + "</tr>";
  }
  html += "</table>";
  document.getElementById("aiTable").innerHTML = html;
}

/* 二次スクリーニング */
async function runSecondScreening() {
  if (!latestResults || latestResults.length === 0) {
    alert("一次スクリーニング結果がありません。");
    return;
  }
  var response = await fetch("/second_screening", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ results: latestResults })
  });
  var data = await response.json();
  latestSecond = data.second_screening;
  renderSecondTable(latestSecond);
}

/* 二次スクリーニング結果 */
function renderSecondTable(data) {
  if (!data || data.length === 0) {
    document.getElementById("secondTable").innerHTML = "<p>二次スクリーニング結果（0 件）</p>";
    return;
  }
  var html = "<p>二次スクリーニング結果（" + data.length + " 件）</p>";
  html += "<table><tr><th>symbol</th><th>company</th><th>market</th><th>close</th><th>short_score</th><th>mid_score</th><th>judgement</th><th>chart</th><th>説明</th></tr>";
  for (var i = 0; i < data.length; i++) {
    var r = data[i];
    html += "<tr>"
      + "<td>" + escapeHtml(r.symbol) + "</td>"
      + "<td>" + escapeHtml(r.company_name || "") + "</td>"
      + "<td>" + escapeHtml(r.market || "") + "</td>"
      + "<td>" + escapeHtml(r.close) + "</td>"
      + "<td>" + escapeHtml(r.short_score) + "</td>"
      + "<td>" + escapeHtml(r.mid_score) + "</td>"
      + "<td>" + escapeHtml(r.gpt_judgement) + "</td>"
      + "<td><a class=\"chart-link\" href=\"https://finance.yahoo.co.jp/quote/" + encodeURIComponent(r.symbol) + "\" target=\"_blank\">📈</a></td>"
      + "<td><a href=\"/api/explain_symbol?symbol=" + encodeURIComponent(r.symbol) + "\" target=\"_blank\">説明</a></td>"
      + "</tr>";
  }
  html += "</table>";
  document.getElementById("secondTable").innerHTML = html;
}

/* 二次スクリーニング指標一覧 */
function renderIndicatorTable(data) {
  if (!data || data.length === 0) {
    document.getElementById("indicatorTable").innerHTML = "<p>二次スクリーニング指標一覧（0 件）</p>";
    return;
  }
  var html = "<p>二次スクリーニング指標一覧（" + data.length + " 件）</p>";
  html += "<table><tr><th>symbol</th><th>drop_from_high_pct</th><th>rebound_from_low_pct</th><th>ema20_vs_ema50</th><th>ema50_vs_ema200</th><th>price_vs_ema20_pct</th><th>vol_vs_ma20</th><th>atr_ratio</th></tr>";
  for (var i = 0; i < data.length; i++) {
    var r = data[i];
    html += "<tr>"
      + "<td>" + escapeHtml(r.symbol) + "</td>"
      + "<td>" + escapeHtml(r.drop_from_high_pct) + "</td>"
      + "<td>" + escapeHtml(r.rebound_from_low_pct) + "</td>"
      + "<td>" + escapeHtml(r.ema20_vs_ema50) + "</td>"
      + "<td>" + escapeHtml(r.ema50_vs_ema200) + "</td>"
      + "<td>" + escapeHtml(r.price_vs_ema20_pct) + "</td>"
      + "<td>" + escapeHtml(r.vol_vs_ma20) + "</td>"
      + "<td>" + escapeHtml(r.atr_ratio) + "</td>"
      + "</tr>";
  }
  html += "</table>";
  document.getElementById("indicatorTable").innerHTML = html;
}

/* 三次スクリーニング */
async function runThirdScreening() {
  if (!latestResults || latestResults.length === 0) {
    alert("一次スクリーニング結果がありません。");
    return;
  }
  var symbols = [];
  for (var i = 0; i < latestResults.length; i++) symbols.push(latestResults[i].symbol);
  var response = await fetch("/third_screening", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbols: symbols })
  });
  var data = await response.json();
  renderThirdTable(data.results);
}

/* 三次スクリーニング結果 */
function renderThirdTable(data) {
  if (!data || data.length === 0) {
    document.getElementById("thirdTable").innerHTML = "<p>三次スクリーニング結果（0 件）</p>";
    return;
  }
  var html = "<p>三次スクリーニング結果（" + data.length + " 件）</p>";
  html += "<table><tr><th>symbol</th><th>analysis</th><th>fundamentals</th></tr>";
  for (var i = 0; i < data.length; i++) {
    var r = data[i];
    html += "<tr>"
      + "<td>" + escapeHtml(r.symbol) + "</td>"
      + "<td>" + escapeHtml(r.analysis) + "</td>"
      + "<td><pre>" + escapeHtml(JSON.stringify(r.fundamentals, null, 2)) + "</pre></td>"
      + "</tr>";
  }
  html += "</table>";
  document.getElementById("thirdTable").innerHTML = html;
}

console.log("PART3 script end");
</script>

</body>
</html>

"""
