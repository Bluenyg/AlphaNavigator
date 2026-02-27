# app/services/offline_pipeline.py

import os
import time
import logging
import pandas as pd
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..database import SessionLocal
from ..models import FundQuantFeature
from .akshare_data import AkshareService
from .math_engine import MathEngine

# 配置独立的 Pipeline 日志，方便每天盘后核对运行情况
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OfflineDataPipeline")


class FundDataPipeline:
    def __init__(self, csv_filename: str = "all_funds_list.csv", batch_size: int = 50):
        """
        :param csv_filename: 基金基础列表文件名
        :param batch_size: 数据库每积累多少条记录进行一次 Commit，平衡性能与安全性
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.csv_path = os.path.join(base_dir, "app", "data", csv_filename)
        self.batch_size = batch_size
        self.db = SessionLocal()

    # ==========================================
    # 🌟 新增：幂等性校验 (判断今日是否已更新)
    # ==========================================
    def check_updated_today(self) -> bool:
        """检查数据库中最新的一条量化数据是否是今天更新的"""
        try:
            # 按最后更新时间倒序，取最新的一条记录
            latest_record = self.db.query(FundQuantFeature).order_by(FundQuantFeature.updated_at.desc()).first()
            if not latest_record or not latest_record.updated_at:
                return False

            # 提取记录的日期与今天的日期进行比对
            latest_date = latest_record.updated_at.date()
            today_date = datetime.now().date()

            return latest_date == today_date
        except Exception as e:
            logger.error(f"检查数据库最新日期时发生异常: {e}")
            return False

    def load_fund_list(self) -> pd.DataFrame:
        """从本地 CSV 加载全市场基金列表，并进行深度清洗瘦身"""
        if not os.path.exists(self.csv_path):
            logger.error(f"!!! 致命错误: 未找到基金列表文件 {self.csv_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.csv_path, names=["code", "name", "type"], dtype={"code": str}, header=0)
            original_len = len(df)

            # 1. 过滤基础无效品种 (货币、理财)
            df = df[~df['type'].str.contains("货币|理财", na=False)]

            # 2. 过滤掉冗余的收费份额 (保留主份额或 A类，去掉 C/E/I/H/后端)
            df = df[~df['name'].str.contains(r'[CEIHF](份额)?$|后端', na=False, regex=True)]

            # 3. 过滤掉封闭运作、目前无法买入的基金
            df = df[~df['name'].str.contains("定开|定期开放|持有期|封闭|滚动", na=False)]

            # 4. 支付宝/场外专属防火墙 (彻底屏蔽场内 ETF)
            df = df[~(df['name'].str.contains("ETF", na=False) & ~df['name'].str.contains("联接", na=False))]
            df = df[~df['code'].str.startswith(('51', '15', '56', '58'), na=False)]

            logger.info(
                f"成功加载本地基金列表。经过【支付宝专属】瘦身，过滤了 {original_len - len(df)} 只标的，最终精简为 {len(df)} 只场外核心基金。")
            return df

        except Exception as e:
            logger.error(f"加载基金列表失败: {e}")
            return pd.DataFrame()

    def extract_real_sector(self, fund_code: str) -> str:
        """获取真实底层资产板块 (对抗 A 股风格漂移的终极武器)"""
        portfolio = AkshareService.get_fund_portfolio(fund_code)
        if portfolio:
            top_stocks = ", ".join(portfolio[:5])
            return f"重仓: {top_stocks}"
        return ""

    def process_single_fund(self, code: str, name: str, ftype: str) -> Optional[dict]:
        """拉取单只基金的数据并进行数学计算"""
        logger.debug(f"正在处理: {code} - {name}...")

        df_history = AkshareService.get_fund_history(code, lookback_days=252)
        if df_history.empty:
            logger.warning(f"跳过 {code} ({name}): 无法获取历史净值或数据不足。")
            return None

        metrics = MathEngine.calculate_fund_metrics(df_history, risk_free_rate=0.02)
        if not metrics:
            logger.warning(f"跳过 {code} ({name}): 量化指标计算失败 (可能是次新基不足30天)。")
            return None

        is_bond_or_money = "债" in str(ftype) or "货币" in str(ftype) or "理财" in str(ftype)
        real_sector = ""
        if not is_bond_or_money:
            real_sector = self.extract_real_sector(code)

        if not real_sector:
            real_sector = ftype

        return {
            "fund_code": code.zfill(6),
            "fund_name": name,
            "fund_type": ftype,
            "primary_sector": real_sector,
            "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "momentum_1y": metrics.get("momentum_1y", 0.0),
            "annual_return": metrics.get("annual_return", 0.0)
        }

    # 🌟 修改 run 方法，加入 force 参数，并在最开始做拦截
    def run(self, max_workers: int = 5, force: bool = False):
        """
        执行流水线主循环
        :param max_workers: 并发线程数
        :param force: 是否强制执行 (无视今天是否已经更新过)
        """
        logger.info(f"========== 🚀 启动公募基金离线量化计算 (并发模式, 线程数:{max_workers}) ==========")

        # 🌟 核心拦截逻辑：判断今日是否已拉取
        if not force and self.check_updated_today():
            logger.info("✅ 检测到今日 (T日) 的量化特征已成功写入数据库，跳过本次网络请求，保护接口额度！")
            logger.info("========== 🏁 离线多线程流水线提前结束 ==========")
            self.db.close()
            return

        start_time = time.time()

        df_funds = self.load_fund_list()
        if df_funds.empty:
            return

        total_funds = len(df_funds)
        success_count = 0
        error_count = 0

        tasks = []
        for index, row in df_funds.iterrows():
            tasks.append({
                "code": str(row['code']).zfill(6),
                "name": str(row['name']),
                "ftype": str(row['type'])
            })

        logger.info(f"正在将 {total_funds} 个标的分配进多线程计算池...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_fund = {
                executor.submit(self.process_single_fund, task["code"], task["name"], task["ftype"]): task["code"]
                for task in tasks
            }

            for future in as_completed(future_to_fund):
                code = future_to_fund[future]
                try:
                    data_dict = future.result()

                    if data_dict:
                        existing_record = self.db.query(FundQuantFeature).filter(FundQuantFeature.fund_code == data_dict['fund_code']).first()

                        if existing_record:
                            existing_record.fund_name = data_dict['fund_name']
                            existing_record.fund_type = data_dict['fund_type']
                            existing_record.primary_sector = data_dict['primary_sector']
                            existing_record.sharpe_ratio = data_dict['sharpe_ratio']
                            existing_record.max_drawdown = data_dict['max_drawdown']
                            existing_record.momentum_1y = data_dict['momentum_1y']
                            existing_record.annual_return = data_dict['annual_return']
                        else:
                            new_record = FundQuantFeature(**data_dict)
                            self.db.add(new_record)

                        success_count += 1
                    else:
                        error_count += 1

                    current_total = success_count + error_count
                    if current_total % 20 == 0:
                        logger.info(f"--- 进度: [{current_total}/{total_funds}] --- 成功: {success_count}, 失败: {error_count}")

                    if success_count > 0 and success_count % self.batch_size == 0:
                        self.db.commit()
                        logger.info(f"📦 已将 {self.batch_size} 条记录 Commit 到数据库。")

                except Exception as e:
                    error_count += 1
                    logger.error(f"处理基金 {code} 时发生并发异常: {e}")
                    self.db.rollback()

        try:
            self.db.commit()
            logger.info("📦 最终剩余数据已完成 Commit。")
        except Exception as e:
            logger.error(f"最终提交数据库时发生异常: {e}")
            self.db.rollback()
        finally:
            self.db.close()

        elapsed_time = (time.time() - start_time) / 60
        logger.info("========== 🏁 离线多线程流水线运行完毕 ==========")
        logger.info(f"总耗时: {elapsed_time:.2f} 分钟.")
        logger.info(f"总处理数: {total_funds} | 成功入库: {success_count} | 失败/跳过: {error_count}")


if __name__ == "__main__":
    pipeline = FundDataPipeline(csv_filename="all_funds_list.csv", batch_size=50)
    # 如果手动运行该脚本，可以传 force=True 强制覆盖今天的数据
    pipeline.run(max_workers=5, force=False)