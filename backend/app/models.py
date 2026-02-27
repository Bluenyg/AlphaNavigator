# app/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
from sqlalchemy.sql import func
from .database import Base


# ==========================================
# 用户域 (User Domain)
# ==========================================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)  # 🌟 新增：存储加密后的密码


class PortfolioItem(Base):
    __tablename__ = "portfolio_items"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    fund_code = Column(String, index=True)
    fund_name = Column(String)

    # ⚠️ 核心修改：金融系统必须记录【份额】和【平均成本】，绝不能只记金额
    shares = Column(Float, default=0.0)  # 持有份额 (Shares)
    avg_cost = Column(Float, default=0.0)  # 持仓均价/移动加权平均成本 (Average Cost)

    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TransactionLog(Base):
    __tablename__ = "transaction_logs"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    raw_input = Column(String)  # 用户说的原始话 (用于查错和NLP迭代)
    action = Column(String)  # BUY / SELL
    fund_code = Column(String)
    amount = Column(Float)  # 交易发生时的法币金额
    timestamp = Column(DateTime(timezone=True), server_default=func.now())


# ==========================================
# 市场分析域 (Market Domain)
# ==========================================
class MarketNews(Base):
    __tablename__ = "market_news"
    id = Column(Integer, primary_key=True)
    fetch_time = Column(DateTime(timezone=True), server_default=func.now())
    title = Column(String)
    content = Column(Text)
    sentiment_score = Column(Float)  # AI 结合量化打出的情绪分
    related_sector = Column(String)  # 关联的热门板块


class MarketIndicator(Base):
    __tablename__ = "market_indicators"
    id = Column(Integer, primary_key=True)
    date = Column(String, index=True)  # YYYY-MM-DD
    index_code = Column(String, index=True)  # 支持多宽基: sh000300, sz399006 等
    ma20 = Column(Float)
    rsi_14 = Column(Float)
    macd = Column(Float, nullable=True)  # 预留字段
    volatility = Column(Float)  # 20日年化波动率
    # 可直接喂给前端 ECharts 的基础数据表


# ==========================================
# 量化特征域 (Quant Feature Domain) 🌟 核心新增
# ==========================================
class FundQuantFeature(Base):
    __tablename__ = "fund_quant_features"
    id = Column(Integer, primary_key=True)
    fund_code = Column(String, unique=True, index=True)  # 基金代码必须唯一
    fund_name = Column(String)  # 基金名称
    fund_type = Column(String)  # 基金类型 (如 混合型, 债券型)
    primary_sector = Column(String, nullable=True)  # 穿透底层重仓股打上的真实板块标签

    # 核心量化指标
    sharpe_ratio = Column(Float, default=0.0)  # 夏普比率 (越高越好)
    max_drawdown = Column(Float, default=0.0)  # 最大回撤 (负数，绝对值越小越抗跌)
    momentum_1y = Column(Float, default=0.0)  # 近一年动量/涨跌幅
    annual_return = Column(Float, default=0.0)  # 年化收益率

    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ==========================================
# 决策域 (Advisor Domain)
# ==========================================
class InvestmentAdvice(Base):
    __tablename__ = "investment_advices"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    generated_at = Column(DateTime(timezone=True), server_default=func.now())

    # ⚠️ 强烈建议增加的字段：记录当时的量化状态（复盘用）
    market_regime = Column(String, nullable=True)  # 例如：Risk-Off, Bullish

    recommended_sector = Column(String)
    recommended_fund = Column(String)
    reasoning = Column(Text)  # LLM 生成的详细解释
    action_plan = Column(JSON)  # 结构化操作字典 {"new_recommendations": [...], "existing_adjustments": [...]}