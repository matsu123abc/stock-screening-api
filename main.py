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

app = FastAPI()

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
                # 180日分の株価データ
                end = datetime.now()
                start = end - timedelta(days=180)
                df = yf.download(symbol, start=start, end=end)

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

        return JSONResponse({"results": results}, status_code=200)

    except Exception as e:
        logging.exception("screening error")
        return JSONResponse({"error": str(e)}, status_code=500)
