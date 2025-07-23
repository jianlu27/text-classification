import os
import transformers
from transformers.utils import logging
import torch
import time
import requests

def download_bert_model(model_name='bert-base-uncased', save_path='./bert_model', num_labels=2, max_retries=3, timeout=300):
    """
    下载BERT模型到本地，使用镜像源并设置超时时间
    :param model_name: 模型名称
    :param save_path: 保存路径
    :param num_labels: 标签数量
    :param max_retries: 最大重试次数
    :param timeout: 超时时间（秒）
    """
    # 设置镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 设置日志级别
    logging.set_verbosity_info()
    
    # 设置requests超时时间
    transformers.utils.hub.REQUESTS_KWARGS = {'timeout': timeout}
    
    if os.path.exists(save_path):
        print(f"模型已存在于{save_path}")
        return
    
    print(f"开始下载{model_name}模型到{save_path}...")
    
    for attempt in range(max_retries):
        try:
            # 下载tokenizer
            print("下载分词器...")
            tokenizer = transformers.BertTokenizer.from_pretrained(
                model_name,
                mirror='tuna',
                local_files_only=False,
                resume_download=True
            )
            
            # 下载模型
            print("下载模型...")
            model = transformers.BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                mirror='tuna',
                local_files_only=False,
                resume_download=True
            )
            
            # 保存到本地
            print("保存模型到本地...")
            os.makedirs(save_path, exist_ok=True)
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
            
            print("模型下载完成！")
            return
            
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10
                print(f"下载超时或连接错误: {str(e)}")
                print(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print("达到最大重试次数，下载失败。")
                raise e
        except Exception as e:
            print(f"下载失败: {str(e)}")
            raise e

if __name__ == "__main__":
    # 设置下载参数
    MODEL_NAME = 'bert-base-uncased'
    SAVE_PATH = './bert_model'
    NUM_LABELS = 2
    MAX_RETRIES = 3
    TIMEOUT = 300  # 5分钟超时
    
    try:
        download_bert_model(
            model_name=MODEL_NAME,
            save_path=SAVE_PATH,
            num_labels=NUM_LABELS,
            max_retries=MAX_RETRIES,
            timeout=TIMEOUT
        )
    except Exception as e:
        print(f"程序执行出错: {str(e)}") 