import os
import torch
import transformers
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from sklearn.metrics import matthews_corrcoef, accuracy_score
import joblib

class TextClassifier:
    """
    文本分类器类，用于CoLA（语言可接受性）任务
    使用BERT模型提取特征，然后使用逻辑回归进行分类
    """
    def __init__(self, model_path='./bert_model'):
        """
        初始化分类器
        Args:
            model_path: BERT预训练模型的路径
        """
        # 初始化 SparkSession，设置本地多线程、内存和仓库目录
        self.spark = SparkSession.builder \
            .appName("GlueTextClassification") \
            .master("local[*]") \
            .config("spark.driver.memory", "6g") \
            .config("spark.sql.warehouse.dir", os.path.abspath("./spark-warehouse")) \
            .getOrCreate()
        # 设置Spark日志级别为ERROR，减少不必要的输出
        self.spark.sparkContext.setLogLevel("ERROR")

        # 检查BERT模型是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BERT模型未找到，请先运行download_bert.py下载模型到{model_path}")

        print(f"从{model_path}加载模型...")
        # 加载BERT分词器和模型
        self.tokenizer = transformers.BertTokenizer.from_pretrained(model_path)
        self.model = transformers.BertModel.from_pretrained(model_path, output_hidden_states=True)

        # 设置设备（GPU如果可用，否则使用CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        print(f"使用设备: {self.device}")

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
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # 提取[CLS]标记的输出作为句子表示
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.append(cls_embeddings)
                # 计算每个句子的词数
                lengths.extend([len(s.split()) for s in batch])
        # 将BERT特征和句子长度特征拼接
        final_feats = np.hstack([np.vstack(features), np.array(lengths).reshape(-1, 1)])
        return final_feats

    def load_dataset(self, path):
        """
        加载数据集
        Args:
            path: 数据集文件路径（TSV格式）
        Returns:
            DataFrame: Spark DataFrame对象
        """
        schema = StructType([
            StructField("sentence", StringType(), True),
            StructField("label", IntegerType(), True)
        ])
        return self.spark.read.csv(path, schema=schema, sep='\t', header=True)

    def prepare_dataset(self, input_path):
        """
        准备数据集，包括加载数据和特征提取
        Args:
            input_path: 输入数据文件路径
        Returns:
            tuple: (Spark DataFrame, 特征列名列表)
        """
        # 加载数据并转换为Pandas DataFrame
        df = self.load_dataset(input_path).toPandas()
        # 提取BERT特征
        feats = self.extract_features(df['sentence'].tolist())
        # 生成特征列名
        feature_names = [f"feature_{i}" for i in range(feats.shape[1] - 1)] + ["length"]
        # 创建特征DataFrame
        df_feats = pd.DataFrame(feats, columns=feature_names)
        # 合并原始数据和特征
        df_final = pd.concat([df[['sentence', 'label']], df_feats], axis=1)
        return self.to_spark_df(df_final, feature_names)

    def to_spark_df(self, pandas_df, feature_cols):
        """
        将Pandas DataFrame转换为Spark DataFrame
        Args:
            pandas_df: Pandas DataFrame
            feature_cols: 特征列名列表
        Returns:
            tuple: (Spark DataFrame, 特征列名列表)
        """
        # 定义Spark DataFrame的schema
        schema = StructType([
            StructField("sentence", StringType(), True),
            StructField("label", IntegerType(), True),
            *[StructField(col, FloatType(), True) for col in feature_cols]
        ])
        return self.spark.createDataFrame(pandas_df, schema=schema), feature_cols

    def train_and_evaluate(self, train_path, val_path):
        """
        训练和评估模型
        Args:
            train_path: 训练集文件路径
            val_path: 验证集文件路径
        """
        print("准备训练集...")
        train_df, train_cols = self.prepare_dataset(train_path)
        print("准备验证集...")
        val_df, _ = self.prepare_dataset(val_path)

        # 将特征列组装成向量
        assembler = VectorAssembler(inputCols=train_cols, outputCol="features")
        train_vec = assembler.transform(train_df).select("sentence", "label", "features")
        val_vec = assembler.transform(val_df).select("sentence", "label", "features")

        # 初始化逻辑回归模型
        lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=50)
        
        # 定义参数网格进行交叉验证
        # 包括：L2正则化参数、ElasticNet混合参数、最大迭代次数
        paramGrid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
            .addGrid(lr.maxIter, [50, 100]) \
            .build()

        # 初始化二分类评估器
        evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        
        # 设置交叉验证
        crossval = CrossValidator(estimator=lr,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=3)

        print("开始训练模型...")
        cv_model = crossval.fit(train_vec)
        
        # 保存最佳模型的参数
        model_save_path = "cola_lr_model"
        os.makedirs(model_save_path, exist_ok=True)
        
        # 获取最佳模型参数
        best_model = cv_model.bestModel
        model_params = {
            'coefficients': best_model.coefficients.toArray(),
            'intercept': best_model.intercept,
            'numFeatures': best_model.numFeatures,
            'numClasses': best_model.numClasses
        }
        
        # 使用joblib保存模型参数
        joblib.dump(model_params, os.path.join(model_save_path, 'model_params.joblib'))
        print(f"模型参数已保存到 '{model_save_path}'")

        # 在验证集上评估模型
        val_preds = cv_model.transform(val_vec)
        pred_pd = val_preds.select("label", "prediction").toPandas()
        # 计算Matthews相关系数和准确率
        mcc = matthews_corrcoef(pred_pd["label"], pred_pd["prediction"])
        acc = accuracy_score(pred_pd["label"], pred_pd["prediction"])
        print(f"验证集MCC: {mcc:.4f}, 准确率: {acc:.4f}")

def main():
    """
    主函数，设置路径并运行训练流程
    """
    model_path = "./bert_model"  # BERT模型路径
    train_path = "processed_data/cola_train.tsv"  # 训练集路径
    val_path = "processed_data/cola_dev.tsv"  # 验证集路径

    # 创建分类器实例并运行训练评估
    clf = TextClassifier(model_path=model_path)
    clf.train_and_evaluate(train_path, val_path)
    # 停止Spark会话，避免连接错误
    clf.spark.stop()

if __name__ == "__main__":
    main()


