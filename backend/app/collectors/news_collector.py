# app/collectors/news_collector.py

import asyncio
import logging
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from ..database import SessionLocal
from ..models import MarketNews
from ..services.akshare_data import AkshareService
from ..agent.news_agent import NewsScoringAgent  # 🌟 引入独立打分 Agent

logger = logging.getLogger("NewsCollector")


async def news_collection_loop():
    """后台异步定时新闻收集器 + 独立 Agent 实时打分"""
    logger.info(">>> [NewsCollector] 异步新闻收集流与独立打分 Agent 已启动...")

    # 实例化打分专家
    news_agent = NewsScoringAgent()

    # 启动时等待 10 秒，确保数据库建表完成
    await asyncio.sleep(10)

    while True:
        try:
            news_items = await asyncio.to_thread(AkshareService.get_realtime_news, limit=20)

            if news_items:
                db: Session = SessionLocal()
                try:
                    saved_count = 0
                    time_threshold = datetime.now() - timedelta(hours=24)

                    for item in news_items:
                        title = item.get("title", "")
                        if not title:
                            continue

                        existing = db.query(MarketNews).filter(
                            MarketNews.title == title,
                            MarketNews.fetch_time >= time_threshold
                        ).first()

                        if not existing:
                            # 🌟 核心拦截：新新闻入库前，呼叫小 Agent 进行精确打分
                            score_res = await asyncio.to_thread(
                                news_agent.score_news, title, item.get("content", "")
                            )

                            new_entry = MarketNews(
                                title=title,
                                content=item.get("content", item.get("source", "")),
                                sentiment_score=score_res.sentiment_score,
                                related_sector=score_res.related_sector
                            )
                            db.add(new_entry)
                            saved_count += 1

                    db.commit()
                    if saved_count > 0:
                        logger.info(f">>> [NewsCollector] 小Agent打分完毕，新增 {saved_count} 条精标新闻入库。")
                except Exception as db_err:
                    db.rollback()
                    logger.error(f"!!! [NewsCollector] 数据库写入失败: {db_err}")
                finally:
                    db.close()

        except Exception as e:
            logger.error(f"!!! [NewsCollector] 网络抓取任务异常: {e}")

        # 每小时执行一次
        await asyncio.sleep(3600)