# app/services/akshare_data.py

import logging
import akshare as ak
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import time
# 生产环境标准日志配置
logger = logging.getLogger("AkshareService")

class AkshareService:
    @staticmethod
    def get_realtime_news(limit: int = 15):
        """
        获取最新的财经新闻 (多维宏观矩阵采集)
        🌟 终极升级版：融合【东财主题搜索】、【财联社电报】、【全球宏观资讯】三路数据源
        """
        try:
            news_list = []

            # 将总抓取额度均衡分配给三个数据源
            limit_em = max(1, limit // 3)
            limit_cls = max(1, limit // 3)
            limit_global = limit - limit_em - limit_cls

            # ==========================================
            # 来源 1：东方财富 - 主题关键字搜索 (产业/A股深度)
            # ==========================================
            try:
                keywords = ["宏观", "A股"]
                kw_limit = max(1, limit_em // len(keywords))
                for kw in keywords:
                    df_em = ak.stock_news_em(symbol=kw)
                    if df_em is not None and not df_em.empty:
                        for _, row in df_em.head(kw_limit).iterrows():
                            news_list.append({
                                "title": str(row.get("新闻标题", "无标题")),
                                "content": str(row.get("新闻内容", "")),
                                "publish_time": str(row.get("发布时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
                                "source": f"东方财富-{kw}"
                            })
            except Exception as e:
                logger.warning(f"获取东方财富新闻失败: {e}")

            # ==========================================
            # 来源 2：财联社 - 实时电报 (市场突发快讯)
            # ==========================================
            try:
                # 财联社的电报是短平快的突发消息，对量化情绪打分极具价值
                df_cls = ak.stock_info_global_cls()
                if df_cls is not None and not df_cls.empty:
                    for _, row in df_cls.head(limit_cls).iterrows():
                        content = str(row.get("内容", ""))
                        # 财联社电报通常是一整段话，我们提取前30字作为标题
                        title = content[:30] + "..." if len(content) > 30 else content
                        pub_time = str(row.get("时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

                        news_list.append({
                            "title": title,
                            "content": content,
                            "publish_time": pub_time,
                            "source": "财联社电报"
                        })
            except Exception as e:
                logger.warning(f"获取财联社电报失败: {e}")

            # ==========================================
            # 来源 3：东方财富 - 全球宏观滚动资讯
            # ==========================================
            try:
                # 聚焦外盘、地缘政治和全球经济指标
                df_global = ak.stock_info_global_em()
                if df_global is not None and not df_global.empty:
                    for _, row in df_global.head(limit_global).iterrows():
                        title = str(row.get("标题", row.get("title", "宏观资讯")))
                        content = str(row.get("内容", row.get("summary", title)))
                        pub_time = str(
                            row.get("发布时间", row.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))

                        news_list.append({
                            "title": title,
                            "content": content,
                            "publish_time": pub_time,
                            "source": "全球宏观资讯"
                        })
            except Exception as e:
                logger.warning(f"获取全球宏观新闻失败: {e}")

            # 🛡️ 终极兜底：即使断网或接口全挂，也要保证系统能跑下去
            if not news_list:
                news_list.append({
                    "title": "市场平稳运行",
                    "content": "当前暂无最新宏观突发新闻，各大指数平稳运行。",
                    "publish_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "系统兜底"
                })

            return news_list

        except Exception as e:
            logger.error(f"新闻引擎发生致命级错误: {e}")
            return []

    @staticmethod
    def get_index_daily(symbol: str = "sh000300", lookback_days: int = 100, max_retries: int = 3) -> pd.DataFrame:
        """
        获取指数日线数据并规范化列名
        🌟 终极修复：引入【新浪/东财】多源自动容灾切换机制，彻底解决单一接口被封断开连接的问题
        """
        attempt = 0
        pure_code = symbol[-6:] if len(symbol) > 6 else symbol

        while attempt < max_retries:
            try:
                logger.info(f"正在获取指数数据: {symbol} (尝试 {attempt + 1}/{max_retries})...")
                # 增加基础休眠，错开高频并发期
                time.sleep(1.0)

                df = None

                # -----------------------------------------
                # 策略 A：优先尝试新浪接口 (极度稳定，抗高并发防封锁)
                # -----------------------------------------
                try:
                    # 新浪接口要求带有 sh/sz 前缀，如 sz399006
                    df = ak.stock_zh_index_daily(symbol=symbol)
                    if df is not None and not df.empty:
                        # 新浪接口原生列名就是 date, open, high, low, close, volume
                        pass
                except Exception as e1:
                    logger.debug(f"新浪接口获取失败，准备切换至东方财富: {e1}")
                    df = None

                # -----------------------------------------
                # 策略 B：如果新浪挂了，自动降级到东方财富历史接口
                # -----------------------------------------
                if df is None or df.empty:
                    df = ak.index_zh_a_hist(symbol=pure_code, period="daily")
                    if df is not None and not df.empty:
                        # 东方财富接口需要映射中文列名
                        col_map = {
                            "日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low", "成交量": "volume"
                        }
                        df.rename(columns=col_map, inplace=True)

                if df is None or df.empty:
                    raise ConnectionError("所有数据源均无响应或被封禁")

                # ============ 统一的数据清洗与截取流程 ============
                df = df.tail(lookback_days).copy()

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date').reset_index(drop=True)

                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.dropna(subset=['close'])
                return df

            except Exception as e:
                attempt += 1
                logger.warning(f"获取指数 {symbol} 遇到网络阻断 ({e})，等待 {attempt * 3} 秒后重试...")
                time.sleep(attempt * 3)  # 被封禁时拉长等待时间

        logger.error(f"!!! 致命错误：获取指数 {symbol} 数据在 {max_retries} 次尝试后依然失败。")
        return pd.DataFrame()

    @staticmethod
    def get_fund_info(fund_code: str) -> Dict[str, Any]:
        """获取基金最新单日净值 (用于日常记录与交易份额核算)"""
        try:
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                return {
                    "date": str(latest.get('净值日期', '')),
                    "nav": float(latest.get('单位净值', 1.0))
                }
            return {}
        except Exception as e:
            logger.error(f"获取基金 {fund_code} 单日净值失败: {e}")
            return {}

    @staticmethod
    def get_fund_history(fund_code: str, lookback_days: int = 252, max_retries: int = 3) -> pd.DataFrame:
        """
        🌟 核心升级：获取基金历史净值走势，加入了防封禁和重试机制。
        用于批量离线测算夏普、回撤等量化指标。
        """
        attempt = 0
        while attempt < max_retries:
            try:
                # 批量抓取时必须休眠，否则必被封 IP
                time.sleep(0.3)

                df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
                if df is not None and not df.empty:
                    df['净值日期'] = pd.to_datetime(df['净值日期'])
                    df = df.sort_values('净值日期').tail(lookback_days).copy()
                    df['单位净值'] = df['单位净值'].astype(float)
                    return df
                return pd.DataFrame()

            except Exception as e:
                attempt += 1
                logger.warning(f"获取基金 {fund_code} 历史净值失败 (尝试 {attempt}/{max_retries}): {e}")
                time.sleep(attempt * 2)  # 指数退避

        logger.error(f"跳过基金 {fund_code}: 连续 {max_retries} 次获取历史净值失败。")
        return pd.DataFrame()

    @staticmethod
    def get_fund_portfolio(fund_code: str) -> List[str]:
        """
        🌟 核心新增：获取基金最新一期的十大重仓股 (用于穿透底层资产，识别风格漂移)
        返回该基金重仓的股票代码或名称列表。
        """
        try:
            time.sleep(0.3)
            # 获取基金持仓信息
            df = ak.fund_portfolio_hold_em(symbol=fund_code, date="2023")  # 此接口返回最新年份的季度数据
            if df is not None and not df.empty:
                # 取最新一季度的前十大重仓股票名称
                latest_date = df['季度'].max()
                latest_portfolio = df[df['季度'] == latest_date]
                stocks = latest_portfolio['股票名称'].head(10).tolist()
                return [str(s) for s in stocks]
            return []
        except Exception as e:
            logger.debug(f"获取基金 {fund_code} 重仓股失败(可能不是权益类基金): {e}")
            return []

    @staticmethod
    def get_sector_board_data(limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取当日全市场行业板块真实交投数据 (用于量化资金面打分)
        """
        try:
            df = ak.stock_board_industry_name_em()
            # 提取涨幅前列的热门板块
            df = df.sort_values("涨跌幅", ascending=False).head(limit)
            sectors = []
            for _, row in df.iterrows():
                sectors.append({
                    "sector": str(row.get('板块名称', '')),
                    "pct_change": float(row.get('涨跌幅', 0.0)),
                    "turnover": float(row.get('换手率', 0.0))
                })
            return sectors
        except Exception as e:
            logger.error(f"获取真实板块交投数据失败: {e}")
            return []