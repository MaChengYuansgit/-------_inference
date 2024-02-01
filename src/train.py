import os
import random
import logging  
from fire import initF, Model, Runner, Data
from config import cfg

# 配置logging
logging.basicConfig(level=logging.INFO,  # 设置日志级别
                    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
                    datefmt='%Y-%m-%d %H:%M:%S',  # 日期格式
                    filename='logs/training.log',  # 日志文件名
                    filemode='w')  # 写入模式

def main(cfg):
    logging.info("Starting training process")  # 使用logging记录信息
    initF(cfg)

    model = Model(cfg)
    data = Data(cfg)

    if cfg['show_data']:
        data.showTrainData()
        logging.info("Showing training data")  # 记录操作
    else:  
        train_loader, val_loader = data.getTrainValDataloader()
        runner = Runner(cfg, model)
        runner.train(train_loader, val_loader)
        logging.info("Training completed")  # 训练完成的记录

if __name__ == '__main__':
    main(cfg)
