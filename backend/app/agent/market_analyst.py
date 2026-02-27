# app/agents/market_analyst.py

import json
import math
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

# LangChain & Pydantic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Internal Services & Models
from ..services.akshare_data import AkshareService
from ..services.math_engine import MathEngine
from ..models import MarketIndicator, MarketNews
from ..database import SessionLocal
from ..config import settings

logger = logging.getLogger("MarketPipeline")


# ==========================================
# 1. 定义 Pydantic 输出模型 (彻底剥离情绪打分，专注宏观研判)
# ==========================================
class MarketSentimentSchema(BaseModel):
    """用于强制 LLM 输出的结构定义 (纯宏观研判)"""
    macro_theme: str = Field(
        ...,
        description="一句话总结当前市场的核心宏观交易主题（如：'强预期弱现实下的政策博弈'、'流动性充裕驱动的科技行情'）"
    )
    top_sector: str = Field(
        ...,
        description="基于已精标新闻流与量化动能，当前最值得配置的具体行业板块（如：'红利低波'、'人工智能'、'纯债'）"
    )
    market_regime: str = Field(
        ...,
        description="市场状态判定。必须从 ['Risk-On (风险偏好提升)', 'Risk-Off (避险降仓)', 'Oscillation (震荡市)'] 中选择"
    )
    reasoning: str = Field(
        ...,
        description="详细逻辑分析：必须结合量价指标、情绪均值以及【情绪离散度/分歧度】进行解释，不超过300字"
    )


# ==========================================
# 2. 市场分析 Pipeline 类 (Quant + 预处理 NLP 融合)
# ==========================================
class MarketPipeline:
    def __init__(self):
        # 宏观分析需要一定的发散总结能力，temperature 设为 0.2 兼顾稳定与泛化
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            temperature=0.2,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )

        # 绑定结构化输出
        self.structured_llm = self.llm.with_structured_output(MarketSentimentSchema)

        # 定义需要追踪的宽基指数
        self.tracking_indices = {
            "沪深300": "sh000300",
            "中证500": "sh000905",
            "创业板指": "sz399006"
        }

    # ----------------------------------------------------------------
    # [核心模块 A]: 场内量化分析 (Quantitative Engine)
    # ----------------------------------------------------------------
    def _detect_market_regime(self, indicators_map: dict) -> dict:
        """
        纯量化多因子判定市场动能与状态 (Multi-Factor Regime Detection)
        基于：趋势(MA)、动量(MACD)、风险(Volatility)
        """
        trend_score = 0
        momentum_score = 0
        high_vol_signals = 0
        total_indices = len([k for k, v in indicators_map.items() if v])

        if total_indices == 0:
            return {"quant_regime": "Unknown", "bull_score": "0/0", "leading_style": "未知"}

        for name, ind in indicators_map.items():
            if not ind: continue

            # 因子 1：趋势 (Trend) - 价格站上 20 日均线
            if ind.get('close', 0) > ind.get('ma20', float('inf')):
                trend_score += 1

            # 因子 2：动量 (Momentum) - MACD 柱状图大于 0
            if ind.get('macd', 0) > 0:
                momentum_score += 1

            # 因子 3：风险 (Risk) - 年化波动率超过 22%
            if ind.get('volatility', 0) > 0.22:
                high_vol_signals += 1

        # 综合量化规则引擎 (Rule-based Engine)
        total_bull_score = trend_score + momentum_score
        max_possible_score = total_indices * 2
        bullish_ratio = total_bull_score / max_possible_score if max_possible_score > 0 else 0

        if bullish_ratio >= 0.6 and high_vol_signals < 2:
            quant_regime = "Bullish (稳健多头)"
        elif bullish_ratio >= 0.6 and high_vol_signals >= 2:
            quant_regime = "Volatile Bullish (高波多头/逼空)"
        elif bullish_ratio <= 0.3:
            quant_regime = "Bearish (空头避险)"
        else:
            quant_regime = "Oscillating (震荡市)"

        valid_inds = {k: v for k, v in indicators_map.items() if v}
        leading_style = max(valid_inds.items(), key=lambda x: x[1].get('rsi', 0))[0] if valid_inds else "无明显主线"

        return {
            "quant_regime": quant_regime,
            "trend_score": f"{trend_score}/{total_indices}",
            "momentum_score": f"{momentum_score}/{total_indices}",
            "high_volatility_warnings": high_vol_signals,
            "leading_style": leading_style,
            "raw_data": indicators_map
        }

    def _analyze_quantitative_data(self, db: Session) -> dict:
        """拉取行情数据 -> 计算技术指标 -> 执行量化状态机 -> 存库"""
        logger.info(">>> [Quant Engine] 开始计算多宽基指数技术面与因子状态...")
        indicators_map = {}

        for index_name, index_code in self.tracking_indices.items():
            df = AkshareService.get_index_daily(index_code)
            if not df.empty:
                ind = MathEngine.calculate_indicators(df)
                if not ind: continue

                indicators_map[index_name] = ind

                existing_ind = db.query(MarketIndicator).filter(
                    MarketIndicator.date == ind['date'],
                    MarketIndicator.index_code == index_code
                ).first()

                if not existing_ind:
                    db_ind = MarketIndicator(
                        date=ind['date'],
                        index_code=index_code,
                        ma20=ind['ma20'],
                        rsi_14=ind['rsi'],
                        volatility=ind['volatility'],
                        macd=ind.get('macd', 0.0)
                    )
                    db.add(db_ind)

        db.commit()
        return self._detect_market_regime(indicators_map)

    def update_quant_indicators(self):
        """仅在后台更新大盘量化指标，不调用 LLM (极度节省 Token)"""
        db: Session = SessionLocal()
        try:
            # 只运行量化计算并存入 MarketIndicator 数据库
            self._analyze_quantitative_data(db)
            logger.info(">>> [Quant Engine] 后台量化指标已刷新存库完毕。")
        except Exception as e:
            db.rollback()
            logger.error(f"!!! [Quant Engine] 后台指标刷新异常: {e}")
        finally:
            db.close()

    # ----------------------------------------------------------------
    # [核心模块 B]: 场外定性分析 (接收小 Agent 的精标数据 + 高级数学清洗)
    # ----------------------------------------------------------------
    def _analyze_qualitative_news(self, db: Session) -> Tuple[str, float, float]:
        """【核心重构】: 引入指数时间衰减 (EMA思想) 与 情绪离散度 (Dispersion) 计算"""
        logger.info(">>> [NLP Engine] 从底层池提取过去 24 小时已精标的宏观新闻流，并计算时间衰减分与情绪离散度...")

        time_threshold = datetime.now() - timedelta(hours=24)

        # 确保只拿小 Agent 已经打过分的新闻
        recent_news = db.query(MarketNews).filter(
            MarketNews.fetch_time >= time_threshold,
            MarketNews.sentiment_score.isnot(None)
        ).order_by(MarketNews.fetch_time.desc()).limit(30).all()

        if not recent_news:
            logger.warning(">>> [NLP Engine] 本地暂无已打分新闻，发送空文本让 LLM 纯靠量价研判...")
            return "近期暂无重大新闻驱动", 5.0, 0.0

        current_time = datetime.now()
        weighted_scores = []
        weights = []

        for n in recent_news:
            # 处理时区一致性，防止 offset-naive 和 offset-aware 报错
            news_time = n.fetch_time.replace(tzinfo=None) if n.fetch_time.tzinfo else n.fetch_time
            delta_hours = max(0.0, (current_time - news_time).total_seconds() / 3600.0)

            # 🌟 核心算法 1：引入半衰期 (Half-life = 6小时)，越旧的新闻权重越低
            w = math.exp(-math.log(2) / 6.0 * delta_hours)
            weights.append(w)
            weighted_scores.append(n.sentiment_score * w)

        sum_w = sum(weights)
        if sum_w == 0:
            return "计算异常兜底", 5.0, 0.0

        # 🌟 核心算法 2：计算加权平均情绪得分
        avg_score = sum(weighted_scores) / sum_w

        # 🌟 核心算法 3：计算情绪加权方差与标准差 (衡量多空分歧度/离散度)
        variance = sum(w * (n.sentiment_score - avg_score) ** 2 for w, n in zip(weights, recent_news)) / sum_w
        sentiment_std = math.sqrt(variance)

        # 将带有分数和关联板块的新闻整理成硬核报告
        formatted_lines = [f"- [{n.sentiment_score}分] {n.title} (指向板块: {n.related_sector})" for n in recent_news]

        # 🚨 终极风控：系统硬性预警插入
        if sentiment_std > 2.5:
            warning_msg = f"⚠️ 【系统量化预警】当前24小时情绪标准差高达 {sentiment_std:.2f}！市场处于极端割裂状态，多空博弈惨烈，随时可能诱发剧烈变盘！"
            formatted_lines.insert(0, warning_msg)

        news_text = "\n".join(formatted_lines)

        return news_text, round(avg_score, 2), round(sentiment_std, 2)

    # ----------------------------------------------------------------
    # [中枢枢纽]: 主执行流水线 (Orchestrator)
    # ----------------------------------------------------------------
    def run_analysis(self) -> Dict[str, Any]:
        """完整主控流程"""
        today_str = datetime.now().strftime("%Y-%m-%d")
        db: Session = SessionLocal()

        try:
            # 1. 提取量化特征
            quant_context = self._analyze_quantitative_data(db)

            # 2. 提取预处理过的新闻文本、平均分和离散度
            news_text, avg_news_score, sentiment_std = self._analyze_qualitative_news(db)

            logger.info(">>> [Fusion Engine] AI 启动高级共振推理 (加入离散度感知)...")

            # 3. 构建减负后的高级 Prompt (注入离散度参数)
            prompt = ChatPromptTemplate.from_template(
                """作为顶级买方宏观策略基金经理，请结合以下【量价因子数据】与【经过时间衰减计算的新闻流】，分析核心矛盾与方向。

                【1. 量价监控面板 (硬逻辑底座)】
                - 系统测算大势: {quant_regime}
                - 趋势支撑度: {trend_score} | 动量强度: {momentum_score}
                - 过去24小时情绪均分: {avg_news_score}/10 (平滑衰减后)
                - 🚨 情绪离散度/分歧度 (Standard Deviation): {sentiment_std} (若>2.5代表多空极其割裂，必须收缩战线防御！)
                - 当前资金最青睐的风格: {leading_style}

                【2. 过去24小时核心新闻与打分流水】
                {news_content}

                【深度推理指令】
                1. 你的任务是基于上述资料输出最终的 macro_theme, top_sector, market_regime 和 reasoning。
                2. 交叉验证：高分新闻（利好）是否得到了量价动能的印证？若情绪分歧极大({sentiment_std}>2.5)，必须在推理中体现对冲思维。
                3. 风险兜底：如果量价显示为 Risk-Off 或者 均分极低，必须在推荐板块中优先配置“纯债”或“黄金”。
                4. 请严格按照 JSON 格式输出。
                """
            )

            # 执行 LLM 推理
            chain = prompt | self.structured_llm
            analysis_result: MarketSentimentSchema = chain.invoke({
                "quant_regime": quant_context.get('quant_regime', 'Unknown'),
                "trend_score": quant_context.get('trend_score', '0/0'),
                "momentum_score": quant_context.get('momentum_score', '0/0'),
                "avg_news_score": avg_news_score,
                "sentiment_std": sentiment_std,
                "leading_style": quant_context.get('leading_style', '未知'),
                "news_content": news_text
            })

            logger.info(f">>> [MarketPipeline] 分析闭环完成！研判状态: {analysis_result.market_regime}")
            logger.info(f"    - 宏观主题: {analysis_result.macro_theme}")
            logger.info(f"    - 推荐配置: {analysis_result.top_sector}")

            return {
                "quant_metrics": quant_context,
                "news_analysis": analysis_result.model_dump(),
                "timestamp": today_str
            }

        except Exception as e:
            db.rollback()
            logger.error(f"!!! [MarketPipeline] 研判系统致命错误: {str(e)}")
            return {
                "error": str(e),
                "quant_metrics": {"quant_regime": "Bearish (Risk-Off)", "leading_style": "避险"},
                "news_analysis": {
                    "macro_theme": "系统风控触发",
                    "top_sector": "纯债/货币基金 (绝对防御)",
                    "market_regime": "Risk-Off",
                    "reasoning": "后台引擎故障降级机制触发，当前强制输出防御性避险指令。"
                },
                "timestamp": today_str
            }
        finally:
            db.close()