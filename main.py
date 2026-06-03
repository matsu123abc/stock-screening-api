from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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

def process_symbol(symbol: str, company_name: str, market: str, log, python_condition: str):
    try:
        # --- 株価データ取得（history を使用） ---
        df = yf.Ticker(symbol).history(period="6mo")

        if df.empty:
            log(f"[WARN] {symbol} の株価データが取得できませんでした")
            return None

        # --- EMA 計算 ---
        df["EMA20"] = df["Close"].ewm(span=20).mean()
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        df["EMA200"] = df["Close"].ewm(span=200).mean()

        # --- 直近高値からの下落率 ---
        recent_high = df["High"].rolling(window=60).max().iloc[-1]
        drop_from_high_pct = (df["Close"].iloc[-1] - recent_high) / recent_high * 100

        # --- 直近安値からの反発率 ---
        recent_low = df["Low"].rolling(window=60).min().iloc[-1]
        rebound_from_low_pct = (df["Close"].iloc[-1] - recent_low) / recent_low * 100

        # --- EMA の位置関係 ---
        ema20_vs_ema50 = (df["EMA20"].iloc[-1] - df["EMA50"].iloc[-1]) / df["EMA50"].iloc[-1] * 100
        ema50_vs_ema200 = (df["EMA50"].iloc[-1] - df["EMA200"].iloc[-1]) / df["EMA200"].iloc[-1] * 100
        price_vs_ema20_pct = (df["Close"].iloc[-1] - df["EMA20"].iloc[-1]) / df["EMA20"].iloc[-1] * 100

        # --- 出来高比 ---
        df["vol_ma20"] = df["Volume"].rolling(window=20).mean()
        vol_vs_ma20 = df["Volume"].iloc[-1] / df["vol_ma20"].iloc[-1]

        # --- ATR 計算 ---
        df["H-L"] = df["High"] - df["Low"]
        df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
        df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
        df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
        df["ATR"] = df["TR"].rolling(window=14).mean()
        atr_ratio = df["ATR"].iloc[-1] / df["Close"].iloc[-1] * 100

        # --- 条件式の評価 ---
        local_vars = {
            "drop_from_high_pct": drop_from_high_pct,
            "rebound_from_low_pct": rebound_from_low_pct,
            "ema20_vs_ema50": ema20_vs_ema50,
            "ema50_vs_ema200": ema50_vs_ema200,
            "price_vs_ema20_pct": price_vs_ema20_pct,
            "vol_vs_ma20": vol_vs_ma20,
            "atr_ratio": atr_ratio,
        }

        try:
            passed = eval(python_condition, {}, local_vars)
        except Exception as e:
            log(f"[ERROR] 条件式の評価に失敗: {e}")
            return None

        if not passed:
            return None

        # --- GPT コメント生成（後で Step4-C で実装） ---
        gpt_comment = None

        result = {
            "symbol": symbol,
            "company_name": company_name,
            "market": market,
            "drop_from_high_pct": drop_from_high_pct,
            "rebound_from_low_pct": rebound_from_low_pct,
            "ema20_vs_ema50": ema20_vs_ema50,
            "ema50_vs_ema200": ema50_vs_ema200,
            "price_vs_ema20_pct": price_vs_ema20_pct,
            "vol_vs_ma20": vol_vs_ma20,
            "atr_ratio": atr_ratio,
            "gpt_comment": gpt_comment,
        }

        log(f"[OK] {symbol} screening passed")
        return result

    except Exception as e:
        log(f"[ERROR] {symbol} processing failed: {e}")
        return None

class BlobCSVRequest(BaseModel):
    blob_filename: str

@app.post("/api/screening_from_blob")
async def screening_from_blob(body: BlobCSVRequest):
    try:
        # --- Blob 接続 ---
        connect_str = os.getenv("AzureWebJobsStorage")
        blob_service = BlobServiceClient.from_connection_string(connect_str)

        blob_container = "block-data"
        blob_name = body.blob_filename

        # --- Blob から CSV を読み込む ---
        blob_client = blob_service.get_blob_client(
            container=blob_container,
            blob=blob_name
        )

        csv_text = blob_client.download_blob().readall().decode("utf-8")

        # --- pandas で CSV を DataFrame 化 ---
        df_csv = pd.read_csv(io.StringIO(csv_text))

        # 必須列チェック
        required_cols = ["コード", "銘柄名", "市場"]
        for col in required_cols:
            if col not in df_csv.columns:
                return {"error": f"CSV に '{col}' 列がありません"}

        # 銘柄リスト抽出
        symbols = [f"{code}.T" for code in df_csv["コード"]]

        # --- screening API に渡す ---
        screening_request = ScreeningRequest(symbols=symbols)
        screening_result = await screening(screening_request)
        results = screening_result["results"]

        # ============================================================
        # ★★★ screening 結果を Blob に保存（Functions 版の後半処理）★★★
        # ============================================================
        result_container = os.getenv("RESULT_CONTAINER", "screening-results")

        today = datetime.now().strftime("%Y-%m-%d")
        output_blob_name = f"{today}/screening_{today}.json"

        result_blob = blob_service.get_blob_client(
            container=result_container,
            blob=output_blob_name
        )

        json_text = json.dumps(results, ensure_ascii=False, indent=2)
        result_blob.upload_blob(json_text, overwrite=True)

        # ============================================================

        return {
            "saved_to": output_blob_name,
            "results": results
        }

    except Exception as e:
        logging.exception("screening_from_blob error")
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "USA FastAPI is running"}

# =========================
# Step2: explain_symbol API
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

【出力形式】
- 企業の概要（何をしている会社か）
- 主力事業
- 強み
- リスク（分かる範囲で）
- 投資家向けの総合コメント（200〜300文字）
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
# Step3: second_screening API
# =========================

class SecondScreeningRequest(BaseModel):
    results: List[Any]

@app.post("/api/second_screening")
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
# Step4-A: screening API（計算のみ）
# =========================

class ScreeningRequest(BaseModel):
    symbols: List[str]

@app.post("/api/screening")
async def screening(body: ScreeningRequest):
    try:
        symbols = body.symbols
        results = []

        for symbol in symbols:
            try:
                # 180日分の株価データ（history を使用）
                df = yf.Ticker(symbol).history(period="6mo")

                if df.empty:
                    continue

                # EMA 計算
                df["EMA20"] = df["Close"].ewm(span=20).mean()
                df["EMA50"] = df["Close"].ewm(span=50).mean()
                df["EMA200"] = df["Close"].ewm(span=200).mean()

                # ATR 計算
                df["H-L"] = df["High"] - df["Low"]
                df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
                df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
                df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
                df["ATR"] = df["TR"].rolling(window=14).mean()

                latest = df.iloc[-1]

                result = {
                    "symbol": symbol,
                    "close": float(latest["Close"]),
                    "ema20": float(latest["EMA20"]),
                    "ema50": float(latest["EMA50"]),
                    "ema200": float(latest["EMA200"]),
                    "atr": float(latest["ATR"]),
                }

                results.append(result)

            except Exception as e:
                logging.exception(f"Error processing {symbol}")

        # ★ JSONResponse をやめて dict を返す
        return {"results": results}

    except Exception as e:
        logging.exception("screening error")
        return {"error": str(e)}
