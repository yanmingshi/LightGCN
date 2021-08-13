#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : logger.py
# @Author: yanms
# @Date  : 2021/8/3 14:47
# @Desc  :
import datetime
import logging
import os
from logging import handlers
import colorlog

LOG_ROOT = './log/'


class Logger(object):
    log_colors_config = {
        'DEBUG': 'cyan',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', back_count=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        s_datefmt = "%d %b %H:%M"
        s_formatter = colorlog.ColoredFormatter(fmt, s_datefmt, log_colors=self.log_colors_config)

        log_filename = '{}-{}.log'.format(filename, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        if not os.path.exists(LOG_ROOT):  #
            os.makedirs(LOG_ROOT)
        log_filepath = os.path.join(LOG_ROOT, log_filename)
        file_fmt = "%(asctime)-15s %(levelname)s  %(message)s"
        file_date_fmt = "%a %d %b %Y %H:%M:%S"
        file_formatter = logging.Formatter(file_fmt, file_date_fmt)

        self.logger = logging.getLogger(filename)
        # format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(s_formatter)  # 设置屏幕上显示的格式

        fh = logging.FileHandler(log_filepath, encoding='utf-8')
        fh.setFormatter(file_formatter)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(fh)


if __name__ == '__main__':
    log = Logger('all', level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    Logger('error', level='error').logger.error('error')
