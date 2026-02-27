# app/agents/user_recorder.py

import os
import re
import pandas as pd
import akshare as ak
from datetime import datetime
from typing import Optional, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from ..models import PortfolioItem, TransactionLog, User, InvestmentAdvice
from ..database import SessionLocal
from ..config import settings


# ==========================================
# 1. LLM 结构化提取模型 (纯净版)
# ==========================================
class TransactionExtraction(BaseModel):
    # 🌟 核心修改：增加独立的 CLEAR_REPORTS 动作
    action: str = Field(
        description="交易动作，必须且只能输出 'BUY', 'SELL', 'CLEAR'(清空持仓流水), 或 'CLEAR_REPORTS'(清空研报)。")
    fund_identifier: Optional[str] = Field(
        default="",
        description="提取出的基金代码(6位数字)或明确的基金名称。如果是 CLEAR 系列动作可留空。"
    )
    amount: Optional[float] = Field(
        default=0.0,
        description="交易金额（元）。如果是全部卖出，请根据上下文推算出总金额。如果是 CLEAR 系列动作可为0。"
    )
    tx_time: Optional[str] = Field(
        default=None,
        description="交易发生的准确时间，格式为 'YYYY-MM-DD HH:MM:SS'。如果用户未明确提到时间，请输出 null。"
    )


# ==========================================
# 2. 基金本地知识库 (CSV 映射与缓存)
# ==========================================
class FundMatcher:
    _instance = None

    def __new__(cls, csv_path="all_funds_list.csv"):
        if cls._instance is None:
            cls._instance = super(FundMatcher, cls).__new__(cls)
            try:
                app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                exact_path = os.path.join(app_dir, "data", "all_funds_list.csv")

                possible_paths = [exact_path, "app/data/all_funds_list.csv", "data/all_funds_list.csv", csv_path]
                actual_path = None
                for p in possible_paths:
                    if os.path.exists(p):
                        actual_path = p
                        break

                if not actual_path:
                    raise FileNotFoundError(f"找不到数据文件，系统尝试寻找过的绝对路径为: {exact_path}")

                cls._instance.df = pd.read_csv(actual_path, dtype={"code": str}, encoding="utf-8-sig")

                cls._instance.df.columns = cls._instance.df.columns.str.strip()
                if "code" in cls._instance.df.columns:
                    cls._instance.df["code"] = cls._instance.df["code"].astype(str).str.strip()
                if "name" in cls._instance.df.columns:
                    cls._instance.df["name"] = cls._instance.df["name"].astype(str).str.strip()

                print(f">>> [FundMatcher] 成功精准加载本地基金库 ({actual_path})，共 {len(cls._instance.df)} 条记录。")
            except Exception as e:
                print(f"!!! [FundMatcher] 加载基金库失败: {e}")
                cls._instance.df = pd.DataFrame(columns=["code", "name", "type"])

        return cls._instance

    def resolve_fund(self, identifier: str) -> Tuple[str, str]:
        identifier = str(identifier).strip()
        if self.df.empty or not identifier:
            return identifier, "Unknown Fund"

        if identifier.isdigit() and len(identifier) == 6:
            match = self.df[self.df["code"] == identifier]
        else:
            match = self.df[self.df["name"].str.contains(identifier, case=False, na=False)]

        if not match.empty:
            return str(match.iloc[0]["code"]), str(match.iloc[0]["name"])
        return identifier, "Unknown Fund"


# ==========================================
# 3. 真实行情服务
# ==========================================
def get_tx_fund_nav(fund_code: str, tx_date: Optional[datetime] = None) -> float:
    """获取基金在指定交易日的真实净值（用于精确计算历史买卖份额）"""
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        if df is not None and not df.empty:
            df['净值日期'] = pd.to_datetime(df['净值日期'])

            if tx_date:
                target_date = pd.to_datetime(tx_date.date())
                past_df = df[df['净值日期'] <= target_date]
                if not past_df.empty:
                    return float(past_df['单位净值'].iloc[-1])
                else:
                    return float(df['单位净值'].iloc[0])

            return float(df['单位净值'].iloc[-1])
    except Exception as e:
        print(f"!!! [Akshare] 获取基金 {fund_code} 净值失败: {e}")
    raise ValueError(f"无法从市场获取基金 {fund_code} 的实时净值数据，交易中止。")


# ==========================================
# 4. 核心记录器引擎 (全 LLM 接管 + 动态上下文)
# ==========================================
class UserRecorderEngine:
    def __init__(self):
        # 使用 0.0 的 temperature 保证提取的确定性
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            temperature=0.0,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        ).with_structured_output(TransactionExtraction)
        self.matcher = FundMatcher()

    def parse_intent(self, user_input: str, username: str) -> Optional[TransactionExtraction]:
        """第一步：动态注入资产上下文，全量交由 LLM 智能解析"""

        # 🌟 快速正则匹配：专门针对清空研报
        if re.search(r"(清除|清空|删除)(我的)?(所有|全部)?(研报|策略|建议|历史策略)", user_input):
            return TransactionExtraction(action="CLEAR_REPORTS")

        # 🌟 快速正则匹配：专门针对清空持仓流水
        if re.search(r"(清除|清空|删除)(我的)?(所有|全部)?(流水|数据|持仓|交易)", user_input):
            return TransactionExtraction(action="CLEAR")

        db: Session = SessionLocal()
        portfolio_text = "当前无持仓。"

        try:
            user = db.query(User).filter(User.username == username).first()
            if user:
                items = db.query(PortfolioItem).filter(PortfolioItem.user_id == user.id,
                                                       PortfolioItem.shares > 0.01).all()
                if items:
                    lines = [f"- {item.fund_name} (代码:{item.fund_code}): 价值约 {item.shares * item.avg_cost:.2f}元"
                             for item in items]
                    portfolio_text = "\n".join(lines)

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prompt = f"""当前系统绝对时间：【{current_time}】
该用户的【当前真实持仓池】如下：
{portfolio_text}

用户刚刚发送了指令："{user_input}"

作为专业的金融AI交易员，请分析用户指令并提取出结构化JSON：
1. action: 买入/加仓填 'BUY'，卖出/减仓/清仓某只基金填 'SELL'。想清除所有资金交易流水填 'CLEAR'。想单独清除历史策略研报填 'CLEAR_REPORTS'。
2. fund_identifier: 提取基金代码或名称。如果是 CLEAR/CLEAR_REPORTS，留空。
3. amount: 如果用户说“清仓某基金”或“全卖了”，请参考持仓池里的金额填入具体数字。如果是 CLEAR/CLEAR_REPORTS 则为 0。
4. tx_time: 如果用户提到了发生时间（如“昨天”、“2月11日”），请基于系统绝对时间精确推算，格式为 YYYY-MM-DD HH:MM:SS。如果没提时间，不要瞎编，返回 null 即可。
"""
            print(f">>> [Recorder] 正在唤醒 LLM 并注入资产池上下文，分析指令: '{user_input}'...")
            return self.llm.invoke(prompt)

        except Exception as e:
            print(f"!!! [Recorder] LLM 提取失败: {e}")
            return None
        finally:
            db.close()

    def process_transaction(self, username: str, user_input: str) -> str:
        """第二步：执行严谨的金融逻辑并入库"""
        extracted = self.parse_intent(user_input, username)

        if not extracted:
            return "抱歉，我未能准确理解您的指令。建议描述清晰一些，例如：'把有色基金卖掉一半' 或 '清空策略研报'。"

        # 强制兜底映射防弹衣
        if extracted.action in ["买入", "申购", "买", "加仓"]:
            extracted.action = "BUY"
        elif extracted.action in ["卖出", "赎回", "清仓", "卖", "减仓"]:
            extracted.action = "SELL"
        elif extracted.action in ["清除研报", "清空策略", "CLEAR_REPORTS"]:
            extracted.action = "CLEAR_REPORTS"
        elif extracted.action in ["清除", "清空", "删除", "重置", "CLEAR"]:
            extracted.action = "CLEAR"

        if extracted.action not in ["BUY", "SELL", "CLEAR", "CLEAR_REPORTS"]:
            return f"未能识别指令方向({extracted.action})，请明确说明是买、卖、清空持仓还是清空研报。"

        db: Session = SessionLocal()

        # 🌟 [分支 A1]: 执行一键清空【持仓与流水】动作
        if extracted.action == "CLEAR":
            try:
                user = db.query(User).filter(User.username == username).first()
                if user:
                    db.query(PortfolioItem).filter(PortfolioItem.user_id == user.id).delete()
                    db.query(TransactionLog).filter(TransactionLog.user_id == user.id).delete()
                    db.commit()
                return "🧹 遵命！您的【持仓和历史交易流水】已全部清空，您的策略研报仍然保留！"
            except Exception as e:
                db.rollback()
                return f"❌ 清除交易记录失败: {str(e)}"
            finally:
                db.close()

        # 🌟 [分支 A2]: 执行独立清空【历史策略研报】动作
        if extracted.action == "CLEAR_REPORTS":
            try:
                user = db.query(User).filter(User.username == username).first()
                if user:
                    db.query(InvestmentAdvice).filter(InvestmentAdvice.user_id == user.id).delete()
                    db.commit()
                return "📄 遵命！您的【历史策略研报】已全部被销毁，您可以随时生成新的组合建议！"
            except Exception as e:
                db.rollback()
                return f"❌ 清除研报失败: {str(e)}"
            finally:
                db.close()

        # [分支 B]: 执行买卖动作
        if not extracted.fund_identifier or not extracted.amount:
            return "请提供明确的交易金额和基金信息。如果想全部卖出，可以说'清仓某某基金'。"

        fund_code, fund_name = self.matcher.resolve_fund(extracted.fund_identifier)
        if fund_name == "Unknown Fund":
            return f"抱歉，未能在知识库或您的持仓中找到 '{extracted.fund_identifier}'，请检查名称或代码。"

        try:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                user = User(username=username, password_hash="placeholder")
                db.add(user)
                db.commit()

            final_time = datetime.now()
            if extracted.tx_time:
                try:
                    final_time = datetime.strptime(extracted.tx_time, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    pass

            tx_nav = get_tx_fund_nav(fund_code, final_time)
            shares_to_transact = extracted.amount / tx_nav

            log = TransactionLog(
                user_id=user.id,
                action=extracted.action,
                fund_code=fund_code,
                amount=extracted.amount,
                raw_input=user_input,
                timestamp=final_time
            )
            db.add(log)

            portfolio = db.query(PortfolioItem).filter(
                PortfolioItem.user_id == user.id,
                PortfolioItem.fund_code == fund_code
            ).first()

            if extracted.action == "BUY":
                if not portfolio:
                    portfolio = PortfolioItem(
                        user_id=user.id, fund_code=fund_code, fund_name=fund_name,
                        shares=0.0, avg_cost=0.0
                    )
                    db.add(portfolio)

                total_cost_before = portfolio.shares * portfolio.avg_cost
                new_total_shares = portfolio.shares + shares_to_transact

                portfolio.avg_cost = (total_cost_before + extracted.amount) / new_total_shares
                portfolio.shares = new_total_shares

            elif extracted.action == "SELL":
                if not portfolio or portfolio.shares < (shares_to_transact * 0.99):
                    return f"卖出失败：{fund_name} 当前可用份额({portfolio.shares:.2f})不足以卖出金额 {extracted.amount}元。"
                portfolio.shares -= shares_to_transact
                if portfolio.shares < 0.01:
                    portfolio.shares = 0.0
                    portfolio.avg_cost = 0.0

            db.commit()
            return f"✅ 交易已确认：{extracted.action} {extracted.amount}元 {fund_name}({fund_code})。\n⏱️ 确认时间: {final_time.strftime('%Y-%m-%d %H:%M')}\n📊 成交时净值: {tx_nav:.4f}\n🔄 变动份额: {shares_to_transact:.2f}份。"

        except ValueError as ve:
            db.rollback()
            return f"❌ 交易驳回: {str(ve)}"
        except Exception as e:
            db.rollback()
            return f"❌ 系统异常: {str(e)}"
        finally:
            db.close()