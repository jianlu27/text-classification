import pandas as pd
import os
from sklearn.model_selection import train_test_split

def process_cola_data(data_dir):
    """
    处理CoLA数据集，统一格式
    :param data_dir: CoLA数据集目录
    :return: 处理后的训练集、验证集和测试集
    """
    print("处理CoLA数据集...")
    
    # 读取训练集
    train_file = os.path.join(data_dir, "train.tsv")
    train_data = pd.read_csv(train_file, sep='\t', header=None,
                            names=['source', 'label', 'note', 'sentence'])
    
    # 读取验证集
    dev_file = os.path.join(data_dir, "dev.tsv")
    dev_data = pd.read_csv(dev_file, sep='\t', header=None,
                          names=['source', 'label', 'note', 'sentence'])
    
    # 读取测试集
    test_file = os.path.join(data_dir, "test.tsv")
    test_data = pd.read_csv(test_file, sep='\t')
    
    # 统一格式
    train_data = train_data[['sentence', 'label']]
    dev_data = dev_data[['sentence', 'label']]
    test_data = test_data[['sentence']]
    
    # 打印数据集信息
    print(f"CoLA训练集大小: {len(train_data)}")
    print(f"CoLA验证集大小: {len(dev_data)}")
    print(f"CoLA测试集大小: {len(test_data)}")
    print(f"CoLA训练集标签分布:\n{train_data['label'].value_counts()}")
    
    return train_data, dev_data, test_data

def process_sst2_data(data_dir):
    """
    处理SST-2数据集，统一格式
    :param data_dir: SST-2数据集目录
    :return: 处理后的训练集、验证集和测试集
    """
    print("\n处理SST-2数据集...")
    
    # 读取训练集
    train_file = os.path.join(data_dir, "train.tsv")
    train_data = pd.read_csv(train_file, sep='\t')
    
    # 读取验证集
    dev_file = os.path.join(data_dir, "dev.tsv")
    dev_data = pd.read_csv(dev_file, sep='\t')
    
    # 读取测试集
    test_file = os.path.join(data_dir, "test.tsv")
    test_data = pd.read_csv(test_file, sep='\t')
    
    # 统一格式
    train_data = train_data[['sentence', 'label']]
    dev_data = dev_data[['sentence', 'label']]
    test_data = test_data[['sentence']]
    
    # 打印数据集信息
    print(f"SST-2训练集大小: {len(train_data)}")
    print(f"SST-2验证集大小: {len(dev_data)}")
    print(f"SST-2测试集大小: {len(test_data)}")
    print(f"SST-2训练集标签分布:\n{train_data['label'].value_counts()}")
    
    return train_data, dev_data, test_data

def save_processed_data(data, output_file):
    """
    保存处理后的数据
    :param data: DataFrame格式的数据
    :param output_file: 输出文件路径
    """
    data.to_csv(output_file, sep='\t', index=False)
    print(f"已保存到: {output_file}")

def preprocess_all_data(base_dir=".", output_dir="processed_data"):
    """
    处理所有数据集并保存
    :param base_dir: 数据集根目录
    :param output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理CoLA数据集
    cola_dir = os.path.join(base_dir, "CoLA")
    cola_train, cola_dev, cola_test = process_cola_data(cola_dir)
    
    # 保存CoLA处理后的数据
    save_processed_data(cola_train, os.path.join(output_dir, "cola_train.tsv"))
    save_processed_data(cola_dev, os.path.join(output_dir, "cola_dev.tsv"))
    save_processed_data(cola_test, os.path.join(output_dir, "cola_test.tsv"))
    
    # 处理SST-2数据集
    sst2_dir = os.path.join(base_dir, "SST-2")
    sst2_train, sst2_dev, sst2_test = process_sst2_data(sst2_dir)
    
    # 保存SST-2处理后的数据
    save_processed_data(sst2_train, os.path.join(output_dir, "sst2_train.tsv"))
    save_processed_data(sst2_dev, os.path.join(output_dir, "sst2_dev.tsv"))
    save_processed_data(sst2_test, os.path.join(output_dir, "sst2_test.tsv"))
    
    print("\n数据预处理完成！")
    return {
        'cola': {
            'train': cola_train,
            'dev': cola_dev,
            'test': cola_test
        },
        'sst2': {
            'train': sst2_train,
            'dev': sst2_dev,
            'test': sst2_test
        }
    }

if __name__ == "__main__":
    # 数据预处理
    processed_data = preprocess_all_data() 