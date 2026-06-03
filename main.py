from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import logging
import yfinance as yf
from openai import AzureOpenAI

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
        # --- 企業情報を Yahoo Finance から取得 ---
        ticker = yf.Ticker(symbol)
        info = ticker.info

        company_name = info.get("shortName") or info.get("longName") or symbol
        summary = info.get("longBusinessSummary")

        if not summary:
            summary = "企業情報（longBusinessSummary）が取得できませんでした。"

        # --- Azure OpenAI に要約させる ---
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
