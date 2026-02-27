# app/services/math_engine.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger("MathEngine")


class MathEngine:
    @staticmethod
    def _safe_float(val: Any) -> float:
        """核心防护函数：消除 NaN, Infinity 并转为 Python 标准 float，避免入库 JSON 崩溃"""
        if pd.isna(val) or np.isinf(val):
            return 0.0
        return float(val)

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> dict:
        """
        工业级指数技术指标计算引擎 (MA, Volatility, RSI, MACD)
        """
        if df is None or df.empty or 'close' not in df.columns:
            logger.warning("DataFrame 为空或缺少 'close' 列，无法计算大盘指标。")
            return {}

        if len(df) < 26:
            logger.warning("数据长度不足26个交易日，MACD等长线指标计算结果可能不准确。")

        df = df.copy()
        df = df.sort_values('date')
        close = df['close']

        try:
            # 1. 均线系统 (MA20)
            df['ma20'] = close.rolling(window=20, min_periods=1).mean()

            # 2. 年化波动率 (20日窗口)
            returns = close.pct_change()
            df['volatility'] = returns.rolling(window=20, min_periods=2).std() * np.sqrt(252)

            # 3. RSI (14日 - Wilder's Smoothing / EWM)
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)
            avg_gain = gain.ewm(com=14 - 1, min_periods=14).mean()
            avg_loss = loss.ewm(com=14 - 1, min_periods=14).mean()
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # 4. MACD (12, 26, 9)
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd_dif = exp1 - exp2
            macd_dea = macd_dif.ewm(span=9, adjust=False).mean()
            macd_hist = (macd_dif - macd_dea) * 2

            df['macd_dif'] = macd_dif
            df['macd_dea'] = macd_dea
            df['macd_hist'] = macd_hist

            latest = df.iloc[-1]

            return {
                "date": str(latest['date'])[:10],
                "close": MathEngine._safe_float(latest.get('close', 0.0)),
                "ma20": MathEngine._safe_float(latest.get('ma20', 0.0)),
                "volatility": MathEngine._safe_float(latest.get('volatility', 0.0)),
                "rsi": MathEngine._safe_float(latest.get('rsi_14', 50.0)),
                "macd": MathEngine._safe_float(latest.get('macd_hist', 0.0)),
                "macd_dif": MathEngine._safe_float(latest.get('macd_dif', 0.0)),
                "macd_dea": MathEngine._safe_float(latest.get('macd_dea', 0.0))
            }
        except Exception as e:
            logger.error(f"大盘技术指标数学计算过程发生异常: {e}")
            return {}

    @staticmethod
    def calculate_fund_metrics(df: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        🌟 核心升级：基金横向风控评价指标引擎
        (计算 Sharpe Ratio, Max Drawdown, Momentum, Annual Return)
        """
        if df is None or df.empty or '单位净值' not in df.columns:
            return {}

        try:
            # 强制转换为数值格式，剔除脏数据 (比如遇到空缺停牌日)
            df['单位净值'] = pd.to_numeric(df['单位净值'], errors='coerce')
            df = df.dropna(subset=['单位净值']).copy()

            # 数据点少于30天（次新基金），量化特征无参考价值，直接抛弃
            if len(df) < 30:
                raise ValueError("有效净值数据点少于30天，无法进行回溯计算")

            returns = df['单位净值'].pct_change().dropna()
            if returns.empty:
                return {}

            # 1. 计算年化收益与波动率
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)

            # 2. 计算夏普比率
            if annual_volatility != 0:
                sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            else:
                sharpe_ratio = 0.0

            # 3. 计算最大回撤 (从顶点下跌的最大幅度)
            cumulative_returns = (1 + returns).cumprod()
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()

            # 4. 计算区间真实动量 (直接取这段数据首尾的涨跌幅，比累计求和更精准)
            first_nav = df['单位净值'].iloc[0]
            last_nav = df['单位净值'].iloc[-1]
            momentum = (last_nav / first_nav) - 1 if first_nav != 0 else 0.0

            return {
                "annual_return": MathEngine._safe_float(annual_return),
                "sharpe_ratio": MathEngine._safe_float(sharpe_ratio),
                "max_drawdown": MathEngine._safe_float(max_drawdown),
                "momentum_1y": MathEngine._safe_float(momentum)
            }
        except Exception as e:
            logger.warning(f"基金量化特征提取失败/跳过: {e}")
            return {}