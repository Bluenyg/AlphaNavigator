# app/agents/inv_advisor.py

import os
import json
import logging
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timezone
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from sqlalchemy.orm import Session
from ..models import PortfolioItem, User, InvestmentAdvice, MarketIndicator, FundQuantFeature, TransactionLog
from ..database import SessionLocal
from ..config import settings

logger = logging.getLogger("AdvisorEngine_LangGraph_Production")


# ==========================================
# 1. Pydantic 结构化输出模型 (约束各个 Agent 的输出)
# ==========================================

# ❌ 删除了 FactorWeights，剥离大模型对核心数学权重的计算权限，改用底层代码硬算。

class SectorAgentOutput(BaseModel):
    top_sectors: List[str] = Field(..., description="当前最看好的 2-3 个 A股/债券 细分板块名称。")
    sector_logic: str = Field(...,
                              description="面向高净值客户的宏观分析。用极具专业感、流畅的投研语言解释逻辑。严禁在文本中暴露权重数字(如0.7)、'量化状态为'、'因子'等生硬术语，字数控制在150字内。")
    # 🌟 factor_weights 字段已移除，由底层系统在状态流转时注入


# 🌟 核心升级 1：重构基金推荐模型，强迫输出胜率逻辑与组合角色
class FundRecommendation(BaseModel):
    code: str = Field(..., description="基金代码")
    name: str = Field(..., description="基金名称")

    position_role: str = Field(...,
                               description="该基金在组合中扮演的角色，仅限：进攻核心 / 卫星增强 / 防守底仓 / 轮动交易仓")
    alpha_source: str = Field(...,
                              description="明确未来收益的 Alpha 核心驱动力，如：行业景气度提升 / 风格切换 / 利率下行 / 政策驱动 / 估值修复")

    buy_strategy: str = Field(...,
                              description="具体的买入建仓姿势。必须依赖当前市场状态(Regime)给出建议。例如：'市场顶部震荡，切勿追高，等趋势确认再买' 或 '防守期，可直接买入底仓'。控制在15个字以内。")
    holding_period_months: int = Field(...,
                                       description="建议持有的时间（以月为单位，严格依据资产类别与当前所处周期调整）。")

    reasoning: str = Field(...,
                           description="深度推荐理由：把量化数据翻译成投资优势，且必须说明这只基金是如何填补或对冲客户当前持仓缺口的。")


class FundAgentOutput(BaseModel):
    recommended_funds: List[FundRecommendation] = Field(..., description="最终优选出的 3 只左右的基金组合")


class PortfolioAdjustment(BaseModel):
    code: str = Field(..., description="已有持仓的基金代码")
    name: str = Field(..., description="已有持仓的基金名称")
    action: str = Field(..., description="操作指令，仅限 'HOLD'(持有), 'REDUCE'(减仓), 'CLEAR'(清仓), 'ADD'(加仓) 之一")
    action_details: str = Field(...,
                                description="具体操作建议：例如'逢高减仓50%'、'清仓落袋为安'、'锁仓等待7天惩罚期结束'、'逢低分批定投'")
    holding_advice: str = Field(...,
                                description="预期持有周期：例如'建议继续持有1-3个月'、'长期底仓锁死1年以上'、'短期规避手续费'")
    reasoning: str = Field(...,
                           description="调仓理由：必须是一段极具深度的专业分析，包含基本面剖析和交易摩擦考量。")


class PortfolioAgentOutput(BaseModel):
    overall_strategy: str = Field(...,
                                  description="账户整体贝塔(Beta)与阿尔法(Alpha)战略叙述，口吻需像一位顶尖私人银行家。")
    adjustments: List[PortfolioAdjustment] = Field(..., description="对现有持仓的操作建议列表。")


# ==========================================
# 2. LangGraph 状态定义 (State/Memory Context)
# ==========================================
class AdvisorState(TypedDict):
    username: str
    portfolio: List[Dict[str, Any]]
    market_data: Dict[str, Any]
    quant_regime: str

    regime_persistence_days: int
    target_allocation: Dict[str, str]
    sector_quant_scores: str

    top_sectors: List[str]
    sector_logic: str
    factor_weights: Dict[str, float]  # 🌟 现在由 _get_dynamic_factor_weights 确定性推导

    candidate_funds: List[Dict[str, Any]]
    recommended_funds: List[dict]
    overall_strategy: str
    adjustments: List[dict]
    error: str


# ==========================================
# 3. 本地量化资产筛选器 (🌟 引入截面 Z-Score 与均值回归惩罚)
# ==========================================
class FundQuantScreener:
    def get_candidates_for_sectors(self, sectors: List[str], quant_regime: str, weights: Dict[str, float]) -> List[
        Dict]:
        db: Session = SessionLocal()
        candidates = []
        is_risk_off = "Risk-Off" in quant_regime or "Bearish" in quant_regime or "震荡" in quant_regime

        try:
            w_s = weights.get("w_sharpe", 0.4)
            w_d = weights.get("w_drawdown", 0.4)
            w_m = weights.get("w_momentum", 0.2)

            for sector in sectors:
                if is_risk_off and "债" not in sector and "货币" not in sector and "红利" not in sector:
                    continue

                query = db.query(FundQuantFeature).filter(
                    (FundQuantFeature.primary_sector.like(f"%{sector}%")) |
                    (FundQuantFeature.fund_name.like(f"%{sector}%")) |
                    (FundQuantFeature.fund_type.like(f"%{sector}%"))
                )

                matched_records = query.all()
                if not matched_records:
                    continue

                # ✅ 1. 将数据转换为 Pandas DataFrame 进行截面向量化运算
                df = pd.DataFrame([{
                    "id": r.id,
                    "code": r.fund_code,
                    "name": r.fund_name,
                    "type": r.fund_type,
                    "sharpe_ratio": float(r.sharpe_ratio or 0.0),
                    "max_drawdown": float(r.max_drawdown or 0.0),  # 通常存为负数，如-0.15。越大(越接近0)越好
                    "momentum_1y": float(r.momentum_1y or 0.0),
                    "raw_obj": r
                } for r in matched_records])

                if df.empty or len(df) < 2:
                    continue

                # ✅ 2. 横截面 Z-Score 标准化 (消除量纲差异)
                for col in ['sharpe_ratio', 'max_drawdown', 'momentum_1y']:
                    col_std = df[col].std()
                    if col_std == 0 or pd.isna(col_std):
                        df[f'{col}_z'] = 0.0
                    else:
                        df[f'{col}_z'] = (df[col] - df[col].mean()) / col_std

                # ✅ 3. A股特色风控：震荡市均值回归惩罚 (反拥挤度)
                if "震荡" in quant_regime or "Oscillating" in quant_regime:
                    # 如果某基金动量超越同类 1.5 个标准差，说明极其拥挤，反而强扣分 (-1.0倍惩罚)
                    df['momentum_1y_z'] = np.where(df['momentum_1y_z'] > 1.5, df['momentum_1y_z'] * -1.0,
                                                   df['momentum_1y_z'])

                # ✅ 4. 动态加权总分
                df['dynamic_score'] = (w_s * df['sharpe_ratio_z']) + (w_d * df['max_drawdown_z']) + (
                            w_m * df['momentum_1y_z'])

                # 按动态得分降序排列
                sorted_df = df.sort_values(by='dynamic_score', ascending=False)

                seen_companies = set()
                top_funds = []

                for _, row_data in sorted_df.iterrows():
                    r = row_data['raw_obj']
                    company_name = r.fund_name[:2]
                    if company_name not in seen_companies:
                        top_funds.append(r)
                        seen_companies.add(company_name)
                    if len(top_funds) >= 2:
                        break

                for row in top_funds:
                    is_bond = "债" in (row.fund_type or "") or "货币" in (row.fund_type or "")
                    candidates.append({
                        "code": row.fund_code,
                        "name": row.fund_name,
                        "sector": sector,
                        "type": row.fund_type,
                        "sharpe_ratio": round(float(row.sharpe_ratio or 0.0), 2),
                        "max_drawdown": f"{round(float(row.max_drawdown or 0.0) * 100, 2)}%",
                        "momentum_1y": f"{round(float(row.momentum_1y or 0.0) * 100, 2)}%",
                        "baseline_holding_months": 6 if is_bond else 18
                    })

            # 🌟 修复候选池不足的问题：放宽到至少 4 只候选
            if len(candidates) < 4:
                fallback_keyword = "债" if is_risk_off else "指数"
                fallback_records = db.query(FundQuantFeature).filter(
                    FundQuantFeature.fund_type.like(f"%{fallback_keyword}%")
                ).all()

                if fallback_records:
                    # 兜底查询简单排序，因为不需要太精细的对冲
                    sorted_fb = sorted(fallback_records,
                                       key=lambda r: (w_s * float(r.sharpe_ratio or 0.0)) + (
                                               w_d * float(r.max_drawdown or 0.0)) + (
                                                             w_m * float(r.momentum_1y or 0.0)),
                                       reverse=True)
                    seen_fb = set()
                    for row in sorted_fb:
                        company_name = row.fund_name[:2]
                        if company_name not in seen_fb:
                            is_bond = "债" in (row.fund_type or "") or "货币" in (row.fund_type or "")
                            candidates.append({
                                "code": row.fund_code, "name": row.fund_name,
                                "sector": "宽基/固收兜底", "type": row.fund_type,
                                "sharpe_ratio": round(float(row.sharpe_ratio or 0.0), 2),
                                "max_drawdown": f"{round(float(row.max_drawdown or 0.0) * 100, 2)}%",
                                "momentum_1y": f"{round(float(row.momentum_1y or 0.0) * 100, 2)}%",
                                "baseline_holding_months": 6 if is_bond else 18
                            })
                            seen_fb.add(company_name)
                        if len(candidates) >= 5:  # 提供5只让它挑3只
                            break

            return candidates[:5]

        except Exception as e:
            logger.error(f"查询离线量化特征库失败: {e}")
            return []
        finally:
            db.close()


# ==========================================
# 4. 投顾引擎主类
# ==========================================
class AdvisorEngine:
    def __init__(self):
        self.screener = FundQuantScreener()
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            max_retries=3
        )
        self.graph = self._build_langgraph()

    def _build_langgraph(self):
        workflow = StateGraph(AdvisorState)

        workflow.add_node("agent1_sector_analyst", self._node_sector_analyst)
        workflow.add_node("bridge_fund_screener", self._node_fund_screener)
        workflow.add_node("agent2_fund_selector", self._node_fund_selector)
        workflow.add_node("agent3_portfolio_manager", self._node_portfolio_manager)

        workflow.add_edge(START, "agent1_sector_analyst")
        workflow.add_edge("agent1_sector_analyst", "bridge_fund_screener")
        workflow.add_edge("bridge_fund_screener", "agent2_fund_selector")
        workflow.add_edge("agent2_fund_selector", "agent3_portfolio_manager")
        workflow.add_edge("agent3_portfolio_manager", END)

        return workflow.compile()

    # 🌟 核心升级：独立抽离确定性因子权重推导函数 (剥离大模型幻觉)
    def _get_dynamic_factor_weights(self, quant_metrics: dict) -> dict:
        """基于系统状态矩阵的确定性数学推导"""
        w_s, w_d, w_m = 0.4, 0.4, 0.2  # 初始基准 (均衡模型)

        regime = quant_metrics.get("quant_regime", "Unknown")
        high_vol_warnings = quant_metrics.get("high_volatility_warnings", 0)

        # 1. 波动率风险溢价转移
        if high_vol_warnings >= 2:
            w_d += 0.3;
            w_s -= 0.15;
            w_m -= 0.15
        elif high_vol_warnings == 1:
            w_d += 0.15;
            w_s -= 0.05;
            w_m -= 0.10

        # 2. 动能溢价
        if "Bullish" in regime and high_vol_warnings < 2:
            w_m += 0.25;
            w_s -= 0.15;
            w_d -= 0.10

        # 3. 震荡防守
        if "Oscillating" in regime or "震荡" in regime or "Bearish" in regime:
            w_m -= 0.15;
            w_s += 0.15

        # 归一化 Softmax 变体 (保证三者之和为 1 且绝不为负)
        w_s = max(0.01, w_s);
        w_d = max(0.01, w_d);
        w_m = max(0.01, w_m)
        total = w_s + w_d + w_m

        weights = {
            "w_sharpe": round(w_s / total, 3),
            "w_drawdown": round(w_d / total, 3),
            "w_momentum": round(w_m / total, 3)
        }
        logger.info(f">>> [Math Engine] 系统根据当前量化面板动态推演出的选基因子权重: {weights}")
        return weights

    # --- Node 1: 宏观分析师 ---
    def _node_sector_analyst(self, state: AdvisorState):
        news = state['market_data'].get('news_analysis', {}).get('top_sector', '无明显热点')
        quant_metrics = state['market_data'].get('quant_metrics', {})

        # ✅ 用纯代码接管权重分配
        factor_weights = self._get_dynamic_factor_weights(quant_metrics)

        # 🌟 核心优化：彻底戒断“散户看新闻炒股”思维，注入机构级“交叉验证”投研框架
        prompt = ChatPromptTemplate.from_template(
            """你是一位拥有20年穿越牛熊经验的顶尖宏观量化策略师。
            你的核心投研哲学是【交叉验证】：绝不盲从单一的新闻政策热点，而是将“宏观叙事(News)”、“真实资金面(Money Flow)”与“量化周期(Regime)”相互印证，推导出真正具备坚实底盘的主攻板块。

            【当前市场量化状态 (Regime)】: {quant_regime}
            【今日真实资金交投热度 (量化打分)】:
            {sector_scores}
            【媒体/新闻驱动的政策热点】: {news}

            请严格按照以下逻辑进行深度推演：
            【去伪存真，寻找共振 (极其重要)】：
                - 🛑 严禁“听风就是雨”：绝不能仅仅因为【媒体热点】提到了某个板块（如贵金属、新能源），就直接将其作为主攻方向！新闻往往带有情绪煽动和滞后性。
                - 🔬 交叉验证机制：你必须将【媒体热点】与【今日真实资金交投热度】进行比对。如果新闻吹捧某赛道，但资金交投榜单上毫无踪影，说明是“散户接盘的虚假繁荣”，请坚决舍弃！
                - 🎯 锁定主攻：只有当一个板块既符合当前的【市场量化状态】(如防守期优先红利/纯债，多头期优选科技/成长)，又在【真实资金交投】中排名前列，或者具备极强的对冲价值时，才可将其选入 `top_sectors`（2-3个）。

            🚨 客户研报表达红线（`sector_logic` 字段输出要求）：
            1. 展现机构级投研深度：向客户汇报时，必须解释这是“剥离了市场短期噪音后，基于底层资金面与宏观周期的共振”得出的结论，体现出极强的客观性。
            2. 语感克制、高级：严禁使用“因为新闻报道了XXX所以我们看好XXX”这种业余话术。请使用“穿透短期情绪博弈，真实资金正加速向XXX靠拢”、“契合当前XXX的宏观象限”等私人银行高级话术。
            3. 严禁暴露内部因子计算公式或“状态为Unknown”等生硬机器语言，字数控制在150字内。
            """
        )
        agent = prompt | self.llm.with_structured_output(SectorAgentOutput)
        result: SectorAgentOutput = agent.invoke({
            "quant_regime": state['quant_regime'],
            "sector_scores": state['sector_quant_scores'],
            "news": news
        })

        # 返回推演出的逻辑，并将代码计算出的权重汇入状态图
        return {
            "top_sectors": result.top_sectors,
            "sector_logic": result.sector_logic,
            "factor_weights": factor_weights
        }

    # --- Node 2: 桥接层 ---
    def _node_fund_screener(self, state: AdvisorState):
        # ✅ 改为传入量化状态(quant_regime)，让筛选器能对震荡市实施拥挤度惩罚
        candidates = self.screener.get_candidates_for_sectors(
            state['top_sectors'],
            state['quant_regime'],
            state['factor_weights']
        )
        return {"candidate_funds": candidates}

    # --- Node 3: 基金研究员 (🌟 晋升为 FOF 组合架构师) ---
    def _node_fund_selector(self, state: AdvisorState):
        if not state['candidate_funds']:
            return {"recommended_funds": []}

        # 🌟 核心修复：增加【强制编队】纪律，必须凑齐 3 只角色互补的基金！
        prompt = ChatPromptTemplate.from_template(
            """你是一位顶尖的量化 FOF 基金经理，正在构建投资组合。
            请基于以下数据，为客户输出高度专业、能创造超额收益(Alpha)的开仓建议。

            【当前市场量化状态 (Regime)】: {quant_regime}
            【宏观配置逻辑】: {sector_logic}
            【客户当前真实持仓】: {portfolio}
            【系统预选的顶尖基金池】: 
            {candidates}

            🚨 投资行为生成铁律：
            1. 【强制编队】：你必须从预选池中挑选出 **3只** 基金构成一个攻守兼备的完整组合！绝对不能只推荐 1 只！
            2. 【角色互补】：这 3 只基金的 `position_role`（组合角色）必须是互补的（例如：包含进攻核心、卫星增强、防守底仓等搭配），不能全部同质化。
            3. 【组合层仓位约束】：审视客户的【真实持仓】，如果客户在某个赛道（如科技、医药）暴露过高，你在推荐这几只基金时，必须明确说明它们是如何【平衡或对冲现有组合风险的】。严防单赛道满仓暴雷！
            4. 【市场状态与买入策略】：`buy_strategy` 必须严格依赖当前【Regime】。如果是高动量基金但在震荡期或顶部特征，严禁建议满仓，必须写明“等趋势确认再买”或“极小仓位定投”。
            5. 【收益来源】：在 `alpha_source` 字段精准指出未来能赚钱的逻辑（行业景气度提升 / 风格切换 / 利率下行 / 政策驱动 / 估值修复）。
            6. 【翻译数据】：把冰冷数据翻译成投资利益，绝不提及“系统为你筛选”等后台词汇。
            """
        )
        agent = prompt | self.llm.with_structured_output(FundAgentOutput)
        result: FundAgentOutput = agent.invoke({
            "quant_regime": state['quant_regime'],
            "sector_logic": state['sector_logic'],
            "portfolio": json.dumps(state['portfolio'], ensure_ascii=False),
            "candidates": json.dumps(state['candidate_funds'], ensure_ascii=False)
        })

        return {"recommended_funds": [f.model_dump() for f in result.recommended_funds]}

    # --- Node 4: 投资组合经理 ---
    def _node_portfolio_manager(self, state: AdvisorState):
        prompt = ChatPromptTemplate.from_template(
            """你是一位顶尖的私人银行财富管家，正在为仅在场外平台交易的高净值客户审视持仓。

            【当前市场量化状态】: {quant_regime}
            【目标资产配置比例模型】: {target_allocation}
            【今日全市场板块交投热度】: {sector_scores}

            🚨 【用户当前真实持仓 (已包含 holding_days 持仓天数)】: 
            {portfolio}

            🚨 【场外资金管理生死红线】：
            1. 识别 C类资产 (波段, 免申购费)：如果持仓名称含"C"且 `holding_days` < 7天：
               - 铁律：无论行情多恶劣，绝对禁止给出 'REDUCE', 'CLEAR' 或 'SELL' 指令！必须给出 'HOLD'，并在 `action_details` 明确指出“静待7天免手续费期满”。
            2. 识别 A类资产 (长线, 有申购费)：如果名称含"A"或不含C。
               - 铁律：长线配置。如果 `holding_days` 较短，尽量建议 'HOLD'。

            【输出要求】
            - action_details 必须极其具体（如“逢高减仓50%”）。
            - holding_advice 给出时间预期（如“建议再持有1-3个月”）。
            - reasoning 必须是一段极具深度的专业分析，必须包含：第一层【基本面与景气度挖掘】，第二层【交易摩擦考量】。严禁只谈费率不谈基本面！
            """
        )
        agent = prompt | self.llm.with_structured_output(PortfolioAgentOutput)
        result: PortfolioAgentOutput = agent.invoke({
            "quant_regime": state['quant_regime'],
            "target_allocation": json.dumps(state['target_allocation'], ensure_ascii=False),
            "sector_scores": state['sector_quant_scores'],
            "portfolio": json.dumps(state['portfolio'], ensure_ascii=False)
        })

        return {
            "overall_strategy": result.overall_strategy,
            "adjustments": [a.model_dump() for a in result.adjustments]
        }

    # ----------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------
    def _get_regime_persistence(self, db: Session) -> int:
        indicators = db.query(MarketIndicator).filter(MarketIndicator.index_code == "sh000300").order_by(
            MarketIndicator.date.desc()).limit(30).all()
        persistence = 1
        if indicators and len(indicators) > 1:
            latest_state = indicators[0].rsi_14 > 50 if indicators[0].rsi_14 else True
            for ind in indicators[1:]:
                if ind.rsi_14 is not None:
                    if (ind.rsi_14 > 50) == latest_state:
                        persistence += 1
                    else:
                        break
        return persistence

    def _get_real_sector_scores(self) -> str:
        try:
            df = ak.stock_board_industry_name_em()
            df = df.sort_values("涨跌幅", ascending=False).head(5)
            score_lines = []
            for _, row in df.iterrows():
                sector = row['板块名称']
                pct_change = row['涨跌幅']
                turnover = row['换手率']
                score_lines.append(f"- 【{sector}】: 动量强度 {pct_change}% (换手率 {turnover}%, 资金交投热度数据)")

            score_lines.append("- 【中长债/纯债】: 避险Beta属性 (宏观防守基准)")
            score_lines.append("- 【红利低波】: 稳定生息资产 (震荡市底仓首选)")
            return "\n".join(score_lines)
        except Exception as e:
            logger.warning(f"获取真实板块数据失败: {e}")
            return "无法获取实时板块数据，请高度依赖新闻与技术指标推断。"

    def generate_advice(self, username: str, market_data: dict) -> dict:
        db: Session = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                return {"status": "error", "message": "未找到该用户，请先录入持仓数据。"}

            portfolio_items = db.query(PortfolioItem).filter(PortfolioItem.user_id == user.id,
                                                             PortfolioItem.shares > 0).all()

            portfolio_data = []
            for p in portfolio_items:
                last_buy = db.query(TransactionLog).filter(
                    TransactionLog.user_id == user.id,
                    TransactionLog.fund_code == p.fund_code,
                    TransactionLog.action == 'BUY'
                ).order_by(TransactionLog.timestamp.desc()).first()

                holding_days = 0
                if last_buy and last_buy.timestamp:
                    try:
                        now = datetime.now(last_buy.timestamp.tzinfo) if last_buy.timestamp.tzinfo else datetime.now()
                        holding_days = (now - last_buy.timestamp).days
                    except:
                        holding_days = (datetime.now() - last_buy.timestamp).days
                elif p.updated_at:
                    try:
                        now = datetime.now(p.updated_at.tzinfo) if p.updated_at.tzinfo else datetime.now()
                        holding_days = (now - p.updated_at).days
                    except:
                        holding_days = (datetime.now() - p.updated_at).days

                portfolio_data.append({
                    "code": p.fund_code,
                    "name": p.fund_name,
                    "shares": p.shares,
                    "holding_days": holding_days
                })

            raw_regime = market_data.get('quant_metrics', {}).get('quant_regime', 'Unknown')
            quant_regime = "震荡筑底(趋势不明朗)" if raw_regime == "Unknown" else raw_regime

            is_risk_off = "Risk-Off" in quant_regime or "Bearish" in quant_regime or "震荡" in quant_regime
            persistence_days = self._get_regime_persistence(db)

            if is_risk_off:
                target_alloc = {"权益类(股票/偏股)": "0% - 20%", "固收类(中长债/纯债)": "60% - 80%",
                                "避险类(黄金/货币)": "10% - 20%"}
            else:
                target_alloc = {"权益类(股票/偏股)": "60% - 80%", "固收类(中长债/纯债)": "10% - 20%", "其他": "10%"}

            sector_scores_real = self._get_real_sector_scores()

            initial_state: AdvisorState = {
                "username": username,
                "portfolio": portfolio_data,
                "market_data": market_data,
                "quant_regime": quant_regime,
                "regime_persistence_days": persistence_days,
                "target_allocation": target_alloc,
                "sector_quant_scores": sector_scores_real,
                "top_sectors": [],
                "sector_logic": "",
                "factor_weights": {},
                "candidate_funds": [],
                "recommended_funds": [],
                "overall_strategy": "",
                "adjustments": [],
                "error": ""
            }

            logger.info(f"========== 启动 Context-Augmented Multi-Agent，服务用户: {username} ==========")
            final_state = self.graph.invoke(initial_state)

            action_plan = {
                "new_recommendations": final_state['recommended_funds'],
                "existing_adjustments": final_state['adjustments']
            }

            record = InvestmentAdvice(
                user_id=user.id,
                market_regime=final_state['quant_regime'],
                recommended_sector=",".join(final_state['top_sectors']),
                recommended_fund=final_state['recommended_funds'][0]['name'] if final_state[
                    'recommended_funds'] else "保持观望",
                reasoning=f"【宏观视野】{final_state['sector_logic']}\n【账户战略】{final_state['overall_strategy']}",
                action_plan=action_plan
            )
            db.add(record)
            db.commit()

            logger.info("========== 多 Agent 投顾引擎执行完毕并入库 ==========")

            return {
                "status": "success",
                "market_context": {
                    "regime": final_state['quant_regime'],
                    "target_sectors": final_state['top_sectors'],
                    "macro_logic": final_state['sector_logic']
                },
                "actions": action_plan,
                "overall_strategy": final_state['overall_strategy']
            }

        except Exception as e:
            db.rollback()
            logger.error(f"!!! [AdvisorEngine] 多智能体协作发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
        finally:
            db.close()