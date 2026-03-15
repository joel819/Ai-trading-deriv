"""
Deriv AI Trading Bot
Connects to Deriv demo account via WebSocket
Tests multiple strategies simultaneously
Uses AI to analyse market conditions
Records all trades for performance analysis
"""

import asyncio
import json
import os
import time
import websockets
import re
from openai import OpenAI, AsyncOpenAI
from datetime import datetime
from collections import deque

# ─── Configuration ────────────────────────────────────────────────────────────
DERIV_TOKEN = os.environ.get("DERIV_DEMO_TOKEN")
DERIV_APP_ID = os.environ.get("DERIV_APP_ID", "1089")  # 1089 is Deriv test app ID
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

# Synthetic indices to trade
SYMBOLS = [
    "R_10",   # Volatility 10
    "R_25",   # Volatility 25
    "R_50",   # Volatility 50
    "R_75",   # Volatility 75
    "R_100",  # Volatility 100
]

# Stake per trade in USD
STAKE = 1.0

# Maximum loss per day before bot stops
MAX_DAILY_LOSS = 20.0

# Minimum AI confidence to place trade
MIN_AI_CONFIDENCE = 7

# Deriv WebSocket URL — uses your registered App ID
WS_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"

# ─── Trade Logger ─────────────────────────────────────────────────────────────
class TradeLogger:
    def __init__(self):
        self.trades = []
        self.daily_pnl = 0.0
        self.session_start = datetime.now()

    def log_trade(self, symbol, strategy, direction, stake, result, profit):
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "strategy": strategy,
            "direction": direction,
            "stake": stake,
            "result": result,
            "profit": profit,
        }
        self.trades.append(trade)
        self.daily_pnl += profit
        print(f"[TRADE] {symbol} | {strategy} | {direction} | "
              f"{result} | P&L: ${profit:.2f} | Daily: ${self.daily_pnl:.2f}")

    def get_strategy_stats(self, strategy):
        strategy_trades = [t for t in self.trades if t["strategy"] == strategy]
        if not strategy_trades:
            return {"trades": 0, "wins": 0, "win_rate": 0, "pnl": 0}
        wins = len([t for t in strategy_trades if t["result"] == "win"])
        pnl = sum(t["profit"] for t in strategy_trades)
        return {
            "trades": len(strategy_trades),
            "wins": wins,
            "win_rate": float(f"{wins / len(strategy_trades) * 100:.1f}"),
            "pnl": float(f"{pnl:.2f}")
        }

    def print_summary(self):
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        for strategy in ["RSI", "TREND", "REVERSAL", "MOMENTUM", "BREAKOUT"]:
            stats = self.get_strategy_stats(strategy)
            if stats["trades"] > 0:
                print(f"{strategy}: {stats['trades']} trades | "
                      f"Win rate: {stats['win_rate']}% | "
                      f"P&L: ${stats['pnl']}")
        print(f"\nTotal Daily P&L: ${self.daily_pnl:.2f}")
        print("="*50 + "\n")


# ─── Price History ─────────────────────────────────────────────────────────────
class PriceHistory:
    def __init__(self, symbol, maxlen=100):
        self.symbol = symbol
        self.prices = deque(maxlen=maxlen)
        self.timestamps = deque(maxlen=maxlen)

    def add(self, price, timestamp):
        self.prices.append(price)
        self.timestamps.append(timestamp)

    def get_prices(self):
        return list(self.prices)

    def latest(self):
        return self.prices[-1] if self.prices else None

    def ready(self, min_candles=20):
        return len(self.prices) >= min_candles


# ─── Strategies ────────────────────────────────────────────────────────────────
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = prices[-i] - prices[-i-1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(f"{rsi:.2f}")


def calculate_sma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def strategy_rsi(prices):
    """RSI overbought/oversold strategy"""
    rsi = calculate_rsi(prices)
    if rsi is None:
        return None
    if rsi < 30:
        return "CALL"  # Oversold — expect rise
    if rsi > 70:
        return "PUT"   # Overbought — expect fall
    return None


def strategy_trend(prices):
    """Simple moving average trend following"""
    sma_fast = calculate_sma(prices, 5)
    sma_slow = calculate_sma(prices, 20)
    if sma_fast is None or sma_slow is None:
        return None
    
    # Asserting types to satisfy linter
    assert isinstance(sma_fast, float)
    assert isinstance(sma_slow, float)
    
    if sma_fast > sma_slow * 1.0005:
        return "CALL"  # Uptrend
    if sma_fast < sma_slow * 0.9995:
        return "PUT"   # Downtrend
    return None


def strategy_reversal(prices):
    """Mean reversion strategy"""
    if len(prices) < 20:
        return None
    sma = calculate_sma(prices, 20)
    current = prices[-1]
    deviation = (current - sma) / sma * 100
    if deviation < -0.5:
        return "CALL"  # Price too far below mean
    if deviation > 0.5:
        return "PUT"   # Price too far above mean
    return None


def strategy_momentum(prices):
    """Momentum breakout strategy"""
    if len(prices) < 10:
        return None
    recent = prices[-5:]
    older = prices[-10:-5]
    recent_avg = sum(recent) / len(recent)
    older_avg = sum(older) / len(older)
    change = (recent_avg - older_avg) / older_avg * 100
    if change > 0.3:
        return "CALL"  # Strong upward momentum
    if change < -0.3:
        return "PUT"   # Strong downward momentum
    return None


def strategy_breakout(prices):
    """Bollinger band breakout"""
    if len(prices) < 20:
        return None
    sma = calculate_sma(prices, 20)
    if sma is None:
        return None
    
    # Asserting types to satisfy linter
    assert isinstance(sma, float)
    
    variance = sum((p - sma) ** 2 for p in prices[-20:]) / 20
    std = variance ** 0.5
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    current = prices[-1]
    if current < lower:
        return "CALL"  # Below lower band — expect bounce up
    if current > upper:
        return "PUT"   # Above upper band — expect bounce down
    return None


STRATEGIES = {
    "RSI": strategy_rsi,
    "TREND": strategy_trend,
    "REVERSAL": strategy_reversal,
    "MOMENTUM": strategy_momentum,
    "BREAKOUT": strategy_breakout,
}


# ─── AI Market Analyser ────────────────────────────────────────────────────────
async def get_ai_confidence(symbol, direction, prices, strategy) -> int:
    """Ask NVIDIA AI to evaluate the trade before placing it"""
    if not NVIDIA_API_KEY:
        return 7  # Default confidence if no API key

    try:
        client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
        recent_prices = prices[-20:]
        price_change = ((recent_prices[-1] - recent_prices[0]) /
                       recent_prices[0] * 100)
        rsi = calculate_rsi(prices) or "N/A"

        prompt = f"""You are a trading analyst for synthetic indices on Deriv.
Analyse this trade setup and give a confidence score.

Symbol: {symbol}
Strategy: {strategy}
Proposed direction: {direction}
Recent price change: {price_change:.3f}%
RSI: {rsi}
Last 5 prices: {recent_prices[-5:]}

Return ONLY a JSON object with no markdown or backticks:
{{"confidence": <integer 1-10>, "reasoning": "one sentence"}}

Score 1-4 means DO NOT trade.
Score 5-6 means weak signal.
Score 7-8 means good signal.
Score 9-10 means very strong signal."""

        msg = await client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = re.sub(r"```json|```", "", msg.choices[0].message.content).strip()
        result = json.loads(raw)
        confidence = result.get("confidence", 5)
        reasoning = result.get("reasoning", "")
        print(f"[AI] {symbol} | {strategy} | {direction} | "
              f"Confidence: {confidence}/10 | {reasoning}")
        return confidence
    except Exception as e:
        print(f"[AI ERROR] {e}")
        return 5


# ─── Deriv WebSocket Client ────────────────────────────────────────────────────
class DerivBot:
    def __init__(self):
        self.ws = None
        self.authorized = False
        self.price_history = {s: PriceHistory(s) for s in SYMBOLS}
        self.logger = TradeLogger()
        self.pending_contracts = {}
        self.req_id = 1

    def next_id(self):
        self.req_id += 1
        return self.req_id

    async def send(self, payload):
        if self.ws:
            await self.ws.send(json.dumps(payload))

    async def authorize(self):
        token = os.environ.get("DERIV_DEMO_TOKEN")
        if not token:
            print("[ERROR] No token provided. Set DERIV_DEMO_TOKEN variable.")
            return
        
        token_str = str(token)
        token_preview = token_str[:4] + "****"  # type: ignore
        print(f"[AUTH] Attempting auth with token: {token_preview}")
        await self.send({
            "authorize": token_str,
            "req_id": self.next_id()
        })

    async def subscribe_ticks(self, symbol):
        await self.send({
            "ticks": symbol,
            "subscribe": 1,
            "req_id": self.next_id()
        })

    async def place_trade(self, symbol, direction, strategy):
        """Place a trade on Deriv"""
        if self.logger.daily_pnl <= -MAX_DAILY_LOSS:
            print(f"[RISK] Daily loss limit reached. Bot stopped.")
            return

        prices = self.price_history[symbol].get_prices()

        # Get AI confidence before trading
        ai_res = await get_ai_confidence(
            symbol, direction, prices, strategy
        )
        confidence: int = int(ai_res)  # type: ignore

        if confidence < MIN_AI_CONFIDENCE:
            print(f"[SKIP] {symbol} | {strategy} | "
                  f"AI confidence too low: {confidence}/10")
            return

        contract_type = "CALL" if direction == "CALL" else "PUT"

        await self.send({
            "buy": 1,
            "price": STAKE,
            "parameters": {
                "amount": STAKE,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": 5,
                "duration_unit": "t",  # 5 ticks
                "symbol": symbol,
            },
            "req_id": self.next_id()
        })

        print(f"[ORDER] {symbol} | {strategy} | {direction} | "
              f"${STAKE} | AI: {confidence}/10")

    async def analyse_and_trade(self, symbol):
        """Run all strategies and trade if signal found"""
        history = self.price_history[symbol]
        if not history.ready():
            return

        prices = history.get_prices()

        for strategy_name, strategy_fn in STRATEGIES.items():
            signal = strategy_fn(prices)
            if signal:
                await self.place_trade(symbol, signal, strategy_name)
                await asyncio.sleep(30)  # Wait 30s between trades
                break  # One trade at a time per symbol

    async def handle_message(self, message):
        data = json.loads(message)
        msg_type = data.get("msg_type")

        if msg_type == "authorize":
            if data.get("error"):
                error_msg = data['error']['message']
                error_code = data['error'].get('code', 'unknown')
                print(f"[ERROR] Auth failed: {error_msg}")
                print(f"[ERROR] Error code: {error_code}")
                print(f"[ERROR] Make sure you are using a VIRTUAL/DEMO account token")
                print(f"[ERROR] Go to app.deriv.com, switch to Virtual account,")
                print(f"[ERROR] then go to Settings > API Token to create a new token")
                return
            self.authorized = True
            account = data["authorize"]
            print(f"[AUTH] Connected as {account.get('email', 'Demo Account')}")
            print(f"[AUTH] Balance: ${account.get('balance', 0):.2f}")

            # Subscribe to all symbols
            for symbol in SYMBOLS:
                await self.subscribe_ticks(symbol)
                await asyncio.sleep(0.5)

        elif msg_type == "tick":
            tick = data["tick"]
            symbol = tick["symbol"]
            price = float(tick["quote"])
            timestamp = tick["epoch"]

            self.price_history[symbol].add(price, timestamp)

            # Analyse every 10 ticks
            if len(self.price_history[symbol].prices) % 10 == 0:
                await self.analyse_and_trade(symbol)

        elif msg_type == "buy":
            if data.get("error"):
                print(f"[ERROR] Trade failed: {data['error']['message']}")
            else:
                contract = data["buy"]
                contract_id = contract.get("contract_id")
                print(f"[PLACED] Contract {contract_id} | "
                      f"${contract.get('buy_price', STAKE):.2f}")

        elif msg_type == "proposal_open_contract":
            contract = data.get("proposal_open_contract", {})
            if contract.get("is_expired") or contract.get("is_sold"):
                profit = float(contract.get("profit", 0))
                result = "win" if profit > 0 else "loss"
                symbol = contract.get("underlying", "UNKNOWN")
                direction = contract.get("contract_type", "UNKNOWN")
                self.logger.log_trade(
                    symbol, "AUTO", direction,
                    STAKE, result, profit
                )

        elif msg_type == "error":
            print(f"[ERROR] {data.get('error', {}).get('message', 'Unknown error')}")

    async def run(self):
        print("="*50)
        print("DERIV AI TRADING BOT STARTING")
        print(f"Symbols: {', '.join(SYMBOLS)}")
        print(f"Stake per trade: ${STAKE}")
        print(f"Max daily loss: ${MAX_DAILY_LOSS}")
        print(f"Min AI confidence: {MIN_AI_CONFIDENCE}/10")
        print("="*50)

        summary_counter = 0

        async with websockets.connect(WS_URL) as ws:
            # Set internal ws after connection
            self.ws = ws  # type: ignore
            await self.authorize()

            async for message in ws:
                await self.handle_message(message)

                # Print summary every 100 messages
                summary_counter += 1
                if summary_counter % 100 == 0:
                    self.logger.print_summary()


# ─── Entry Point ──────────────────────────────────────────────────────────────
async def main():
    if not DERIV_TOKEN:
        print("[ERROR] DERIV_DEMO_TOKEN environment variable not set")
        return
    if not DERIV_APP_ID:
        print("[ERROR] DERIV_APP_ID environment variable not set")
        return
    if not NVIDIA_API_KEY:
        print("[WARNING] NVIDIA_API_KEY not set. Bot will use default confidence score of 7.")

    bot = DerivBot()

    while True:
        try:
            await bot.run()
        except websockets.exceptions.ConnectionClosed:
            print("[RECONNECT] Connection lost. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[ERROR] {e}. Restarting in 10 seconds...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
