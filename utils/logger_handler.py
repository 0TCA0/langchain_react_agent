from datetime import datetime
import logging
import os

from utils.path_tool import get_abs_path
LOG_ROOT = get_abs_path('logs')
os.makedirs(LOG_ROOT, exist_ok=True)
# 日志格式配置：时间-日志器名称-日志级别-文件名:行号-日志内容
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,  # 控制台输出级别
        file_level: int = logging.DEBUG,    # 文件输出级别
        log_file: str = None,               # 日志文件路径
) -> logging.Logger:
    """创建全局 logger，同时输出到控制台和日志文件。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 避免重复添加处理器，否则在 Streamlit 重跑时会重复打印日志。
    # logger.handlers：这是一个列表，存储了当前 logger 已经绑定的所有处理器（比如控制台输出器、文件写入器）
    if logger.handlers:
        return logger

    # 控制台日志：控制台用于实时观察应用行为。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)
    # 文件日志：更适合排查线上/回溯类问题。
    if not log_file:   # 日志文件的存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)
    return logger

# 将logger定义为全局属性，方便其他模块直接使用
logger = get_logger()
if __name__ == '__main__':
    logger.info("信息日志")
    logger.error("错误日志")
    logger.debug("调试日志")


