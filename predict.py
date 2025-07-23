import os
import torch
import transformers
import pandas as pd
import numpy as np
import joblib
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, classification_report

class BERTLogisticEvaluator:
    """
    BERT+逻辑回归模型评估器
    支持CoLA（语言可接受性）和SST-2（情感分析）两个任务
    可以对验证集进行评估，也可以对无标签数据进行预测
    """
    def __init__(self, task='cola', model_path='./bert_model'):
        """
        初始化评估器
        Args:
            task: 任务名称，可选'cola'或'sst2'
            model_path: BERT预训练模型路径
        """
        self.task = task.lower()
        self.model_path = model_path
        # 根据任务名称确定保存的逻辑回归模型路径
        self.saved_model_path = f"{self.task}_lr_model/model_params.joblib"
        # 设置计算设备（GPU/CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化Spark会话
        self.spark = SparkSession.builder \
            .appName(f"{self.task.upper()}_Evaluate") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()
        # 设置Spark日志级别
        self.spark.sparkContext.setLogLevel("ERROR")

        # 检查并加载BERT模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BERT模型未找到: {model_path}")
        print(f"加载 BERT 模型与 Tokenizer from {model_path}...")
        self.tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
        self.bert = transformers.BertModel.from_pretrained(model_path, output_hidden_states=True).to(self.device)
        self.bert.eval()  # 设置为评估模式

        # 检查并加载训练好的逻辑回归模型参数
        if not os.path.exists(self.saved_model_path):
            raise FileNotFoundError(f"未找到保存的Logistic回归模型参数: {self.saved_model_path}")
        self.lr_params = joblib.load(self.saved_model_path)
        print(f"任务: {self.task.upper()} | 使用设备: {self.device}")

    def extract_features(self, sentences, batch_size=32):
        """
        使用BERT模型提取文本特征
        Args:
            sentences: 待处理的句子列表
            batch_size: 批处理大小，默认32
        Returns:
            final_feats: 包含BERT特征和句子长度的numpy数组
        """
        features, lengths = [], []
        with torch.no_grad():  # 不计算梯度，节省内存
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                # 对批次数据进行分词和编码
                tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                # 获取BERT输出
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                # 提取[CLS]标记的输出作为句子表示
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.append(cls_embeddings)
                # 计算每个句子的词数
                lengths.extend([len(s.split()) for s in batch])
        # 将BERT特征和句子长度特征拼接
        final_feats = np.hstack([np.vstack(features), np.array(lengths).reshape(-1, 1)])
        return final_feats

    def load_labeled_dataset(self, path):
        """
        加载带标签的数据集（用于评估）
        Args:
            path: 数据集文件路径（TSV格式）
        Returns:
            DataFrame: 包含句子和标签的Pandas DataFrame
        """
        schema = StructType([
            StructField("sentence", StringType(), True),
            StructField("label", IntegerType(), True)
        ])
        return self.spark.read.csv(path, schema=schema, sep='\t', header=True).toPandas()

    def load_unlabeled_dataset(self, path):
        """
        加载无标签的数据集（用于预测）
        Args:
            path: 数据集文件路径（TSV格式）
        Returns:
            DataFrame: 只包含句子的Pandas DataFrame
        """
        schema = StructType([
            StructField("sentence", StringType(), True)
        ])
        return self.spark.read.csv(path, schema=schema, sep='\t', header=True).toPandas()

    def predict(self, sentences):
        """
        对输入句子进行预测
        Args:
            sentences: 待预测的句子列表
        Returns:
            preds: 预测标签数组
        """
        # 提取特征
        X = self.extract_features(sentences)
        # 使用保存的参数重建逻辑回归模型
        clf = LogisticRegression()
        clf.coef_ = self.lr_params['coefficients'].reshape(1, -1)
        clf.intercept_ = np.array([self.lr_params['intercept']])
        clf.classes_ = np.array([0, 1])
        # 进行预测
        preds = clf.predict(X)
        return preds

    def evaluate(self, val_path):
        """
        在验证集上评估模型性能
        Args:
            val_path: 验证集文件路径
        """
        print(f"加载验证集: {val_path}")
        df = self.load_labeled_dataset(val_path)
        y_true = df['label'].values
        sentences = df['sentence'].tolist()
        y_pred = self.predict(sentences)

        # 计算并输出多个评估指标
        print("\n📊 验证集评估指标:")
        print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
        print(f"精确率: {precision_score(y_true, y_pred):.4f}")
        print(f"召回率: {recall_score(y_true, y_pred):.4f}")
        print(f"F1 分数: {f1_score(y_true, y_pred):.4f}")
        print(f"MCC: {matthews_corrcoef(y_true, y_pred):.4f}")
        print("\n分类报告:")
        print(classification_report(y_true, y_pred, digits=4))

    def predict_and_save(self, text_path, output_path):
        """
        对无标签数据进行预测并保存结果
        Args:
            text_path: 输入文本文件路径
            output_path: 预测结果保存路径
        """
        print(f"加载待预测文本: {text_path}")
        df = self.load_unlabeled_dataset(text_path)
        sentences = df['sentence'].tolist()
        # 进行预测
        preds = self.predict(sentences)
        # 将预测结果添加到DataFrame并保存
        df['prediction'] = preds
        df.to_csv(output_path, sep='\t', index=False)
        print(f"预测结果保存到: {output_path}")

def main():
    """
    主函数，设置任务参数并运行评估和预测
    支持CoLA和SST-2两个任务，可通过修改task参数切换
    """
    task = 'sst2'  # 可改为 'sst2'
    model_path = './bert_model'  # BERT模型路径
    val_path = f'processed_data/{task}_dev.tsv'  # 验证集路径（带标签）
    text_path = f'processed_data/{task}_test.tsv'  # 测试集路径（无标签）
    pred_output = f'processed_data/{task}_test_pred.tsv'  # 预测结果输出路径

    # 创建评估器实例
    evaluator = BERTLogisticEvaluator(task=task, model_path=model_path)
    evaluator.evaluate(val_path)  # 在验证集上评估模型性能
    evaluator.predict_and_save(text_path, pred_output)  # 对测试集进行预测并保存结果
    # 停止Spark会话，避免连接错误
    evaluator.spark.stop()

if __name__ == "__main__":
    main()
