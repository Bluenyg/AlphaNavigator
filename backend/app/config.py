# app/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ==========================================
    # 1. 数据库配置 (Local SQLite)
    # ==========================================
    # sqlite:///./ 代表在当前运行目录下生成 invest_db.db 文件
    DATABASE_URL: str = "sqlite:///./invest_db.db"

    # ==========================================
    # 2. LLM 大语言模型配置
    # ==========================================
    # 生产环境建议在根目录建立 .env 文件写入 OPENAI_API_KEY=sk-xxxx
    OPENAI_API_KEY: str = "sk-haYwsKDMzq8X2dIfJolmfWip0miiZR9ynUQvX8AUMiTQM1Gu"
    OPENAI_BASE_URL: str = "https://one-api.bltcy.top/v1"  # 支持 DeepSeek / 硅基流动 等兼容接口
    MODEL_NAME: str = "qwen3-max"

    # ==========================================
    # 3. 量化金融系统参数
    # ==========================================
    RISK_FREE_RATE: float = 0.02  # 无风险收益率 (2%，用于夏普比率等计算)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # 允许在环境变量缺失时不会立刻报错，方便本地调试
        extra = "ignore"

settings = Settings()