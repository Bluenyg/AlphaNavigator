# app/main.py (在文件头部补充引入 asyncio 和 collector)
import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import akshare as ak
from .database import engine, Base, get_db
from .models import MarketIndicator, User, PortfolioItem, TransactionLog, InvestmentAdvice, FundQuantFeature
from .agent.user_recorder import UserRecorderEngine
from .agent.market_analyst import MarketPipeline
from .agent.inv_advisor import AdvisorEngine
from .collectors.news_collector import news_collection_loop  # 引入收集器
import hashlib  # 🌟 新增加密库
import sys
from .services.offline_pipeline import FundDataPipeline
# 🌟 核心修复：解决 Windows 下 FastAPI 底层 asyncio 频繁报 10054 错误的问题
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ==========================================
# 0. 日志与生命周期配置 (Production Standard)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("API_Gateway")


# ==========================================
# 1. 自动量化图表巡检器
# ==========================================
async def market_analysis_loop():
    """异步定时大盘扫描器"""
    logger.info(">>> [AutoAnalyzer] 自动量化分析引擎已启动...")
    await asyncio.sleep(5)

    while True:
        try:
            logger.info(">>> [AutoAnalyzer] 开始执行后台定时量化扫描...")
            # 🌟 核心优化：只调用纯数学计算更新图表，不调用大模型分析
            await asyncio.to_thread(market_pipeline.update_quant_indicators)
        except Exception as e:
            logger.error(f"!!! [AutoAnalyzer] 定时扫描异常: {e}")

        await asyncio.sleep(14400)  # 每4小时跑一次

# ==========================================
# 1.5 自动基金量化特征流水线 (新增)
# ==========================================
async def fund_quant_pipeline_loop():
    """异步定时离线量化特征计算流水线"""
    logger.info(">>> [FundPipeline] 离线基金量化特征计算引擎已启动...")
    await asyncio.sleep(10) # 延迟启动，错峰避开应用刚启动时的资源消耗高峰

    while True:
        try:
            logger.info(">>> [FundPipeline] 开始执行每日全市场基金特征提取与更新...")
            pipeline = FundDataPipeline(csv_filename="all_funds_list.csv", batch_size=50)
            # 🌟 核心：必须使用 to_thread，否则 Pandas 计算和网络 IO 会卡死整个 FastAPI 服务
            await asyncio.to_thread(pipeline.run)
            logger.info(">>> [FundPipeline] 每日基金量化库更新完成。")
        except Exception as e:
            logger.error(f"!!! [FundPipeline] 定时计算异常: {e}")

        # 每天跑一次即可 (24小时 = 86400秒)
        # 生产环境如果追求精准，可以写逻辑判断是否到了晚上 22:00 再执行，这里用 sleep 简化处理
        await asyncio.sleep(86400)

# ==========================================
# 2. 生命周期管理 (挂载所有自动任务)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info(">>> [System] 系统启动：正在校验数据库表结构...")
    Base.metadata.create_all(bind=engine)
    logger.info(">>> [System] 数据库校验完成，AI 投顾引擎上线。")

    # 🌟 启动三大后台自动化引擎 🌟
    # 1. 新闻情报收集（每小时）
    news_task = asyncio.create_task(news_collection_loop())
    # 2. 市场图表指标计算（每4小时）
    market_task = asyncio.create_task(market_analysis_loop())
    # 3. 基金离线量化特征计算（每日） -> 这是你新增的
    fund_task = asyncio.create_task(fund_quant_pipeline_loop())

    yield

    logger.info(">>> [System] 系统关闭：正在清理资源与自动任务...")
    news_task.cancel()
    market_task.cancel()
    fund_task.cancel() # 记得在这里取消任务


# 实例化 FastAPI 应用
app = FastAPI(
    title="Alpha AI Quant Advisor (Production)",
    description="工业级大模型+量化多因子融合投顾系统 API",
    version="3.0.0",  # 升级为 3.0.0 架构
    lifespan=lifespan
)

# 跨域资源共享 (CORS) 配置，确保 Next.js/Vue/React 前端能正常访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请务必将 "*" 替换为前端实际域名 (如 https://your-domain.com)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. 核心引擎全局单例 (避免重复初始化节约内存)
# ==========================================
logger.info(">>> 正在初始化底层 AI 与量化引擎...")
user_recorder = UserRecorderEngine()
market_pipeline = MarketPipeline()
advisor_engine = AdvisorEngine()
logger.info(">>> 底层引擎初始化完毕。")


# ==========================================
# 2. Pydantic 接口请求模型 (API Schemas)
# ==========================================
class ChatRequest(BaseModel):
    username: str = Field(..., description="唯一用户标识")
    message: str = Field(..., description="用户的自然语言交易指令，例如：'买入10000元沪深300ETF'")


class RecommendRequest(BaseModel):
    username: str = Field(..., description="唯一用户标识")
    force_refresh: bool = Field(False, description="是否强制穿透缓存，重新去公网抓取全市场最新数据（耗时约5-10秒）")

class AuthRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")

# ==========================================
# 3. 核心业务接口 (API Routes)
# ==========================================

@app.post("/api/user/auth", tags=["User Actions"], summary="用户登录与自动注册")
def user_auth(req: AuthRequest, db: Session = Depends(get_db)):
    """
    【鉴权中心】：如果用户不存在则自动注册；如果存在则校验密码。
    为了安全，密码使用 SHA-256 进行哈希单向加密存储。
    """
    # 密码加密处理 (千万不要在数据库存明文密码)
    pwd_hash = hashlib.sha256(req.password.encode('utf-8')).hexdigest()

    user = db.query(User).filter(User.username == req.username).first()

    if user:
        # 老用户登录，校验密码
        if user.password_hash != pwd_hash:
            raise HTTPException(status_code=401, detail="密码错误，请重试喵~")
        return {"status": "success", "message": "登录成功"}
    else:
        # 新用户，自动注册
        new_user = User(username=req.username, password_hash=pwd_hash)
        db.add(new_user)
        db.commit()
        return {"status": "success", "message": "新主理人注册成功，欢迎加入！"}

@app.post("/api/user/chat", tags=["User Actions"], summary="处理用户自然语言交易指令")
def user_chat(req: ChatRequest):
    """
    【订单柜台】：接收用户自然语言输入，利用 正则/LLM 提取金融意图，并拉取真实净值计算份额后更新持仓数据库。
    """
    try:
        reply_message = user_recorder.process_transaction(req.username, req.message)
        return {"status": "success", "reply": reply_message}
    except Exception as e:
        logger.error(f"处理用户交易指令失败: {e}")
        raise HTTPException(status_code=500, detail=f"交易处理异常: {str(e)}")


@app.get("/api/user/portfolio", tags=["User Actions"], summary="获取用户当前真实持仓看板")
def get_user_portfolio(username: str, db: Session = Depends(get_db)):
    """
    【资产看板】：前端 Dashboard 必备接口，展示用户当前持仓详情、最新市值与交易流水。
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"status": "success", "portfolio": [], "total_cost_basis": 0, "total_current_value": 0,
                "transactions": []}

    # 1. 获取当前持仓
    items = db.query(PortfolioItem).filter(PortfolioItem.user_id == user.id).all()
    portfolio_data = []
    total_cost = 0.0
    total_current_value = 0.0

    for item in items:
        if item.shares > 0.01:  # 过滤掉已清仓的极小碎片份额
            # 你的原始买入成本 = 持有份额 * 买入均价
            cost = item.shares * item.avg_cost
            total_cost += cost

            # 🌟 核心升级：实时去 Akshare 拉取该基金今天的最新净值
            current_nav = item.avg_cost  # 默认先设为成本价兜底
            try:
                # 为了防止某个基金退市或网络卡顿导致整个接口崩溃，这里必须加 try-except
                df = ak.fund_open_fund_info_em(symbol=item.fund_code, indicator="单位净值走势")
                if df is not None and not df.empty:
                    current_nav = float(df['单位净值'].iloc[-1])
            except Exception as e:
                logger.warning(f"获取基金 {item.fund_code} 实时净值失败，使用成本价兜底。原因: {e}")

            # 你的当前真实市值 = 持有份额 * 今天的最新净值
            current_value = item.shares * current_nav
            total_current_value += current_value

            portfolio_data.append({
                "fund_code": item.fund_code,
                "fund_name": item.fund_name,
                "shares": round(item.shares, 2),
                "avg_cost": round(item.avg_cost, 4),  # 买入时的成本均价
                "cost_basis": round(cost, 2),  # 买入时的总本金 (比如 50)
                "current_nav": round(current_nav, 4),  # 今天的最新净值
                "current_value": round(current_value, 2),  # 🌟 今天的最新市值 (比如 48.5)
                "profit_amount": round(current_value - cost, 2),  # 盈亏绝对额
                "profit_rate": round(((current_value / cost) - 1) * 100, 2) if cost > 0 else 0  # 盈亏百分比
            })

    # 2. 获取最近的交易流水记录
    logs = db.query(TransactionLog).filter(TransactionLog.user_id == user.id).order_by(TransactionLog.id.desc()).limit(
        10).all()
    tx_data = []
    for log in logs:
        # 获取时间戳 (兼容 timestamp 或 created_at 字段)
        t_time = getattr(log, 'timestamp', getattr(log, 'created_at', None))
        time_str = t_time.strftime("%m-%d %H:%M") if t_time else "刚刚"

        tx_data.append({
            "action": "买入" if log.action == "BUY" else "卖出",
            "fund_code": log.fund_code,
            "amount": log.amount,
            "time": time_str
        })

    return {
        "status": "success",
        "portfolio": portfolio_data,
        "total_cost_basis": round(total_cost, 2),  # 你的总投入本金
        "total_current_value": round(total_current_value, 2),  # 🌟 你的当前总市值
        "total_profit": round(total_current_value - total_cost, 2),  # 总盈亏
        "transactions": tx_data
    }


@app.post("/api/market/analyze", tags=["Market Engine"], summary="后台异步触发全市场量化分析")
def trigger_analysis(background_tasks: BackgroundTasks):
    """
    【宏观研究】：通过定时任务（如 crontab）每天收盘后调用此接口一次。
    系统将在后台静默拉取三大指数数据、计算 MACD/RSI/波动率、抓取新闻并做 LLM 情绪打分入库。
    """

    def task():
        logger.info(">>> [Background Task] 开始执行全市场深度量化分析流水线...")
        try:
            result = market_pipeline.run_analysis()
            regime = result.get('quant_metrics', {}).get('quant_regime', '未知')
            logger.info(f">>> [Background Task] 分析完毕. 当前判定状态: {regime}")
        except Exception as e:
            logger.error(f">>> [Background Task] 分析失败: {e}")

    # 将耗时任务交给 FastAPI 背景任务池，立即返回给前端 200 OK
    background_tasks.add_task(task)
    return {"status": "success", "message": "市场分析流水线已在后台启动。"}

@app.get("/api/advisor/history", tags=["Advisor Engine"], summary="获取历史投顾策略")
def get_advisor_history(username: str, db: Session = Depends(get_db)):
    """
    【复盘中心】：获取用户过去生成的历史策略记录，按时间倒序排列。
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return {"status": "success", "history": []}

    # 取出最近的 15 条历史策略
    advices = db.query(InvestmentAdvice).filter(InvestmentAdvice.user_id == user.id) \
        .order_by(InvestmentAdvice.generated_at.desc()).limit(15).all()

    history_data = []
    for adv in advices:
        # 格式化时间为 YYYY-MM-DD HH:MM
        date_str = adv.generated_at.strftime("%Y-%m-%d %H:%M") if adv.generated_at else "未知时间"

        # 解析当时存入的 reasoning，分离出宏观逻辑和整体战略
        reasoning = adv.reasoning or ""
        macro_logic = reasoning
        overall_strategy = ""
        if "【账户战略】" in reasoning:
            parts = reasoning.split("【账户战略】")
            macro_logic = parts[0].replace("【宏观视野】", "").strip()
            overall_strategy = parts[1].strip()

        history_data.append({
            "id": adv.id,
            "date": date_str,
            "status": "success",
            "market_context": {
                "regime": adv.market_regime,
                "target_sectors": adv.recommended_sector.split(",") if adv.recommended_sector else [],
                "macro_logic": macro_logic
            },
            "actions": adv.action_plan if isinstance(adv.action_plan, dict) else {},
            "overall_strategy": overall_strategy
        })

    return {"status": "success", "history": history_data}

@app.get("/api/market/visualization", tags=["Market Engine"], summary="获取市场状态走势数据(用于前端图表)")
def get_visualization_data(
        index_code: str = Query("sh000300",
                                description="指数代码，可选：sh000300 (沪深300), sz399006 (创业板), sh000905 (中证500)"),
        db: Session = Depends(get_db)
):
    """
    【数据中台】：返回用于前端 ECharts/Recharts 渲染的历史技术指标数据。
    支持多指数切换，时间正序返回。
    """
    data = db.query(MarketIndicator) \
        .filter(MarketIndicator.index_code == index_code) \
        .order_by(MarketIndicator.date.desc()) \
        .limit(30) \
        .all()

    if not data:
        return {"status": "success", "index_code": index_code, "xAxis_dates": [], "series": {}}

    # 前端图表要求时间轴从左到右正序排列
    data.reverse()

    return {
        "status": "success",
        "index_code": index_code,
        "xAxis_dates": [d.date for d in data],
        "series": {
            "rsi": [round(d.rsi_14, 2) if d.rsi_14 else 50.0 for d in data],
            "volatility": [round(d.volatility, 4) if d.volatility else 0.0 for d in data],
            "ma20": [round(d.ma20, 2) if d.ma20 else 0.0 for d in data],
            "macd": [round(d.macd, 4) if d.macd else 0.0 for d in data]  # 新增 MACD 渲染支持
        }
    }


@app.post("/api/advisor/recommend", tags=["Advisor Engine"], summary="生成个性化投资建议报告 (LangGraph)")
def get_investment_advice(req: RecommendRequest):
    """
    【投顾大脑】：整合最新的市场量化数据与用户私人持仓。
    基于 LangGraph 多智能体协作架构，通过量化信号筛选目标，并交由 LLM 生成投顾报告。
    """
    try:
        logger.info(f"========== 收到用户 [{req.username}] 的投顾诊断请求 ==========")

        # 1. 数据刷新与降级策略
        if req.force_refresh:
            logger.info(">>> 触发强制刷新：正在重新进行全市场穿透扫描...")
            latest_market_data = market_pipeline.run_analysis()
        else:
            logger.info(">>> 使用普通模式：尝试执行幂等扫描 (复用今日已入库数据以加速响应)...")
            # 因为 market_pipeline 内部已经实现了防重复入库机制，直接调用也很安全且极快
            latest_market_data = market_pipeline.run_analysis()

        if "error" in latest_market_data and "降级" not in latest_market_data.get("news_analysis", {}).get("reasoning",
                                                                                                           ""):
            # 如果是真正的抛错而不是我们写的风控降级
            raise ValueError(f"市场分析模块底层异常: {latest_market_data['error']}")

        # 2. 调用 LangGraph 投顾多智能体协作引擎
        logger.info(">>> 市场数据已就绪，正在唤醒 LangGraph 多智能体网络...")
        advice = advisor_engine.generate_advice(req.username, latest_market_data)

        if advice.get("status") == "error":
            raise HTTPException(status_code=400, detail=advice.get("message"))

        logger.info(f"========== 用户 [{req.username}] 投顾诊断完成 ==========")
        return advice

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.error(f"Advisor API 全局兜底拦截: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="系统生成建议失败，AI 引擎内部错误。")



# ==========================================
# 4. 健康检查探针 (K8s/Docker 必备)
# ==========================================
@app.get("/health", tags=["System"], summary="系统健康探针")
def health_check():
    """用于负载均衡器和容器编排工具监测服务存活状态"""
    return {
        "status": "ok",
        "system": "Alpha AI Quant Advisor",
        "version": app.version,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    # 本地调试启动命令。生产环境建议使用: gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)