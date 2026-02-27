# app/agents/news_agent.py

import logging
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from ..config import settings

logger = logging.getLogger("NewsAgent")


# 1. 强制规范单条新闻的打分输出
class SingleNewsScore(BaseModel):
    sentiment_score: float = Field(
        ...,
        description="单条新闻的情绪分(0.0-10.0)。0为极度利空(如崩盘/制裁)，5为中性，10为极度利好。必须为正数。"
    )
    related_sector: str = Field(
        ...,
        description="该新闻主要利好的A股或宏观板块（如'半导体'、'出海'、'纯债'），若无明显板块填'宽基/无'。"
    )


# 2. 独立的新闻分析专员
class NewsScoringAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.MODEL_NAME,
            temperature=0.1,  # 极低温度，保证打分客观且稳定
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL
        )
        self.structured_llm = self.llm.with_structured_output(SingleNewsScore)

        self.prompt = ChatPromptTemplate.from_template(
            """你是一个专业的量化金融新闻分析机器。请对以下这单条新闻进行快速的量化特征提取。
            【新闻标题】: {title}
            【新闻内容】: {content}

            请严格按照 JSON 格式输出 sentiment_score 和 related_sector。
            """
        )
        self.chain = self.prompt | self.structured_llm

    def score_news(self, title: str, content: str) -> SingleNewsScore:
        """对单条新闻进行打分拦截"""
        try:
            return self.chain.invoke({"title": title, "content": content})
        except Exception as e:
            logger.warning(f"[NewsAgent] 单条新闻打分失败 ({title[:10]}...): {e}")
            # 失败兜底返回中性分
            return SingleNewsScore(sentiment_score=5.0, related_sector="未知")