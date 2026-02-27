from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

# 【核心修复】：解决 SQLite 与 FastAPI 的多线程并发冲突
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args
    # 移除了针对 PostgreSQL 的 pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 依赖注入项
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()