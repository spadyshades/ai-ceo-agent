"""Live financial data tool using yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yfinance as yf


@dataclass
class FinancialSnapshot:
    ticker: str
    name: str
    currency: str
    current_price: float | None
    previous_close: float | None
    day_change_pct: float | None
    week_52_high: float | None
    week_52_low: float | None
    market_cap: float | None
    pe_ratio: float | None
    dividend_yield: float | None
    summary: str


def get_snapshot(ticker: str = "BMW.DE") -> FinancialSnapshot:
    """Fetch a live financial snapshot for a stock ticker."""
    stock = yf.Ticker(ticker)
    info: dict[str, Any] = stock.info or {}

    current = info.get("currentPrice") or info.get("regularMarketPrice")
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

    day_change = None
    if current and prev_close and prev_close > 0:
        day_change = round(((current - prev_close) / prev_close) * 100, 2)

    market_cap = info.get("marketCap")
    pe = info.get("trailingPE") or info.get("forwardPE")
    div_yield = info.get("dividendYield")
    if div_yield:
        div_yield = round(div_yield * 100, 2)

    high_52 = info.get("fiftyTwoWeekHigh")
    low_52 = info.get("fiftyTwoWeekLow")
    name = info.get("shortName") or info.get("longName") or ticker
    currency = info.get("currency", "EUR")

    cap_str = ""
    if market_cap:
        if market_cap >= 1e12:
            cap_str = f"{market_cap / 1e12:.1f}T"
        elif market_cap >= 1e9:
            cap_str = f"{market_cap / 1e9:.1f}B"
        else:
            cap_str = f"{market_cap / 1e6:.0f}M"

    parts = [f"{name} ({ticker}): {currency} {current or 'N/A'}"]
    if day_change is not None:
        direction = "up" if day_change >= 0 else "down"
        parts.append(f"{direction} {abs(day_change):.2f}% from previous close")
    if high_52 and low_52:
        parts.append(f"52-week range: {low_52}-{high_52}")
    if cap_str:
        parts.append(f"Market cap: {cap_str}")
    if pe:
        parts.append(f"P/E: {pe:.1f}")
    if div_yield:
        parts.append(f"Dividend yield: {div_yield:.2f}%")

    return FinancialSnapshot(
        ticker=ticker,
        name=name,
        currency=currency,
        current_price=current,
        previous_close=prev_close,
        day_change_pct=day_change,
        week_52_high=high_52,
        week_52_low=low_52,
        market_cap=market_cap,
        pe_ratio=pe,
        dividend_yield=div_yield,
        summary=". ".join(parts),
    )
