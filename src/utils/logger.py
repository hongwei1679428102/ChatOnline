import logging
import colorlog

def setup_logger():
    """配置彩色日志"""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        fmt='%(asctime)s - %(log_color)s%(levelname)-8s%(reset)s - %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING': 'yellow',
            'ERROR':   'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    ))
    
    logger = colorlog.getLogger(__name__)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    # 移除可能存在的默认处理器
    for handler in logger.handlers[:-1]:
        logger.removeHandler(handler)
    
    return logger

logger = setup_logger() 