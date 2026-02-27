# app/agents/inv_advisor.py

import os
import json
import logging
import pandas as pd
import numpy as np
import akshare as ak
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from sqlalchemy.orm import Session
from ..models import PortfolioItem, User, InvestmentAdvice, MarketIndicator, FundQuantFeature
from ..database import SessionLocal
from ..config import settings

logger = logging.getLogger("AdvisorEngine_LangGraph_Production")


# ==========================================
# 1. Pydantic 结构化输出模型 (约束各个 Agent 的输出)
# ==========================================

class FactorWeights(BaseModel):
    w_sharpe: float = Field(..., description="夏普比率权重 (0.0 - 1.0)")
    w_drawdown: float = Field(..., description="最大回撤权重 (0.0 - 1.0)")
    w_momentum: float = Field(..., description="动量权重 (0.0 - 1.0)")


class SectorAgentOutput(BaseModel):
    top_sectors: List[str] = Field(..., description="当前最看好的 2-3 个 A股/债券 细分板块名称。")
    # 🌟 优化：禁止 AI 输出数学公式和系统术语
    sector_logic: str = Field(...,
                              description="面向高净值客户的宏观分析。用极具专业感、流畅的投研语言解释逻辑。严禁在文本中暴露权重数字(如0.7)、'量化状态为'、'因子'等生硬术语，字数控制在150字内。")
    factor_weights: FactorWeights = Field(...,
                                          description="根据当前宏观周期动态生成的因子权重矩阵，三个权重相加应为 1.0。该数据不会展示给客户。")


class FundRecommendation(BaseModel):
    code: str = Field(..., description="基金代码")
    name: str = Field(..., description="基金名称")
    trend_prediction: str = Field(...,
                                  description="长线胜率预测：结合真实的夏普比率、最大回撤等量化特征，给出风险收益比评估。")
    holding_period_months: int = Field(...,
                                       description="建议持有的时间（以月为单位，严格依据资产类别调整，如固收短、权益长）。")
    # 🌟 优化：要求使用“人文+金融”语言
    reasoning: str = Field(...,
                           description="推荐理由：把冰冷的量化数据翻译成客户听得懂的投资优势，例如‘该资产在极端行情下抗跌能力极强’。严禁暴露后台系统指令。")


class FundAgentOutput(BaseModel):
    recommended_funds: List[FundRecommendation] = Field(..., description="最终优选出的 3 只左右的基金组合")


class PortfolioAdjustment(BaseModel):
    code: str = Field(..., description="已有持仓的基金代码")
    name: str = Field(..., description="已有持仓的基金名称")
    action: str = Field(..., description="操作指令，仅限 'HOLD'(持有), 'REDUCE'(减仓), 'CLEAR'(清仓), 'ADD'(加仓) 之一")
    # 🌟 优化：内化交易规则，转化为专业投顾建议
    reasoning: str = Field(...,
                           description="调仓理由：必须将A/C类的摩擦成本内化为专业的理财建议（如‘由于您持有的是适合长线的份额，短期赎回存在较高的摩擦成本，建议耐心持有’），绝对禁止出现‘该基金为场外A类’、‘根据平台红线’等机器人般的复读机话语。")


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
    factor_weights: Dict[str, float]

    candidate_funds: List[Dict[str, Any]]
    recommended_funds: List[dict]
    overall_strategy: str
    adjustments: List[dict]
    error: str


# ==========================================
# 3. 本地量化资产筛选器
# ==========================================
class FundQuantScreener:
    def get_candidates_for_sectors(self, sectors: List[str], is_risk_off: bool, weights: Dict[str, float]) -> List[
        Dict]:
        db: Session = SessionLocal()
        candidates = []
        try:
            w_s = weights.get("w_sharpe", 0.4)
            w_d = weights.get("w_drawdown", 0.4)
            w_m = weights.get("w_momentum", 0.2)
            logger.info(f">>> [Quant Engine] 正在应用 LLM 动态因子权重: Sharpe={w_s}, Drawdown={w_d}, Momentum={w_m}")

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

                def calc_dynamic_score(record):
                    sharpe = float(record.sharpe_ratio or 0.0)
                    drawdown = float(record.max_drawdown or 0.0)
                    momentum = float(record.momentum_1y or 0.0)
                    return (w_s * sharpe) + (w_d * drawdown) + (w_m * momentum)

                sorted_records = sorted(matched_records, key=calc_dynamic_score, reverse=True)

                # 🌟 修复同质化推荐：增加一个简单的防重机制，确保同一家基金公司的产品只入选一个
                seen_companies = set()
                top_funds = []
                for r in sorted_records:
                    company_name = r.fund_name[:2]  # 简单提取前两个字作为基金公司特征，如“银华”、“工银”
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

            # 如果候选不足，用宽基兜底
            if len(candidates) < 3:
                fallback_keyword = "债" if is_risk_off else "指数"
                fallback_records = db.query(FundQuantFeature).filter(
                    FundQuantFeature.fund_type.like(f"%{fallback_keyword}%")
                ).all()

                if fallback_records:
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
                        if len(candidates) >= 4:
                            break

            return candidates[:4]

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
            temperature=0.3,  # 稍微调高至 0.3，让行文更有温度和变化
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

    # --- Node 1: 宏观分析师 ---
    def _node_sector_analyst(self, state: AdvisorState):
        news = state['market_data'].get('news_analysis', {}).get('top_sector', '无明显热点')

        prompt = ChatPromptTemplate.from_template(
            """作为顶尖宏观量化策略师，请一步步思考：
            第一步：分析【市场量化状态】，生成最符合当前周期的因子权重矩阵 (w_sharpe, w_drawdown, w_momentum)，三者相加必须为 1.0。
            第二步：寻找基本面与资金面共振的板块。

            【市场量化状态】: {quant_regime}
            【真实板块交投热度】:
            {sector_scores}
            【宏观新闻热点】: {news}

            🚨 客户表达红线：
            1. 在 `sector_logic` 中，你必须用高情商、专业的财富顾问口吻向客户汇报。
            2. 严禁暴露你的内部计算逻辑（绝对不能出现类似“因子权重归一化”、“0.7+0.1=1.0”、“状态为Unknown”这种机器人语言）。
            3. 如果状态不明朗，请用“当前市场正处于震荡整固期”来优雅地替代。
            """
        )
        agent = prompt | self.llm.with_structured_output(SectorAgentOutput)
        result: SectorAgentOutput = agent.invoke({
            "quant_regime": state['quant_regime'],
            "sector_scores": state['sector_quant_scores'],
            "news": news
        })

        weights = {
            "w_sharpe": result.factor_weights.w_sharpe,
            "w_drawdown": result.factor_weights.w_drawdown,
            "w_momentum": result.factor_weights.w_momentum
        }

        return {"top_sectors": result.top_sectors, "sector_logic": result.sector_logic, "factor_weights": weights}

    # --- Node 2: 桥接层 ---
    def _node_fund_screener(self, state: AdvisorState):
        is_risk_off = "Risk-Off" in state['quant_regime'] or "Bearish" in state['quant_regime']
        candidates = self.screener.get_candidates_for_sectors(state['top_sectors'], is_risk_off,
                                                              state['factor_weights'])
        return {"candidate_funds": candidates}

    # --- Node 3: 基金研究员 ---
    def _node_fund_selector(self, state: AdvisorState):
        if not state['candidate_funds']:
            return {"recommended_funds": []}

        prompt = ChatPromptTemplate.from_template(
            """你是一位资深的量化 FOF 基金经理，请基于系统筛选出的标的，为客户输出具有说服力的推荐信。

            【宏观配置逻辑】: {sector_logic}
            【系统预选的顶尖基金池】: 
            {candidates}

            🚨 客户表达红线：
            1. 【翻译数据】：把冰冷的量化指标翻译成客户关心的投资利益。比如最大回撤小，你要说“它在市场大跌时展现出了极强的防御韧性”。
            2. 【展现差异】：如果你推荐了多只基金，请在 reasoning 中强调它们各自不同的定位和特色，不要把两只基金写得一模一样。
            3. 【隐蔽后台】：绝不要提及“系统为你筛选”、“动态因子模型”等后台词汇，表现得像这是你经过深度调研得出的结论。
            """
        )
        agent = prompt | self.llm.with_structured_output(FundAgentOutput)
        result: FundAgentOutput = agent.invoke({
            "sector_logic": state['sector_logic'],
            "candidates": json.dumps(state['candidate_funds'], ensure_ascii=False)
        })

        return {"recommended_funds": [f.model_dump() for f in result.recommended_funds]}

    # --- Node 4: 投资组合经理 ---
    def _node_portfolio_manager(self, state: AdvisorState):
        prompt = ChatPromptTemplate.from_template(
            """你是一位顶尖的私人银行财富管家，正在为高净值客户（仅在支付宝等场外平台交易）审视持仓。

            【当前市场量化状态】: {quant_regime}
            【目标资产配置比例模型】: {target_allocation}
            【建议买入的新资产】: {new_funds}
            【用户当前真实持仓】: {portfolio}

            🚨 【内化规则与表达红线】 (必须强制执行，这是你的专业度所在)：
            你心里清楚A类基金(长线, 1.5%惩罚赎回费)和C类基金(短线波段, 需满7天)的交易底线规则，但**你在对客户说话时，绝对不能像复读机一样背出规则原话！**

            错误表达❌：“该基金为场外A类份额(名称含联接A)，根据平台红线，持有期未满1年不建议卖出，以免产生高额赎回费。”
            正确表达✅：“这只基金目前作为我们底仓的核心资产，考虑到其长线定位，目前继续持有可以避免不必要的短期摩擦成本，静待估值修复。”

            错误表达❌：“识别到C类资产，允许敏捷清仓...”
            正确表达✅：“目前该板块动能有所减弱，得益于该基金灵活的费率结构，建议您可以获利了结，落袋为安。”

            指令约束：
            1. overall_strategy 提供宏观维度的账户调整思路。
            2. reasoning 必须采用上述【正确表达】的高情商、专业的服务口吻。
            """
        )
        agent = prompt | self.llm.with_structured_output(PortfolioAgentOutput)
        result: PortfolioAgentOutput = agent.invoke({
            "quant_regime": state['quant_regime'],
            "target_allocation": json.dumps(state['target_allocation'], ensure_ascii=False),
            "new_funds": json.dumps(state['recommended_funds'], ensure_ascii=False),
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
            portfolio_data = [{"code": p.fund_code, "name": p.fund_name, "shares": p.shares} for p in portfolio_items]

            # 🌟 修复 Unknown 问题：如果量化系统返回 Unknown，用更优雅的词汇替代，防止大模型懵掉
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