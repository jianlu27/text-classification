## 📁 项目结构

> 🔧 本项目通过 `download_bert.py` 下载模型到本地（模型文件较大，未上传至仓库），  
> 💾 使用 `preprocess_data.py` 对原始数据进行处理，输出保存至 `processed_data/` 目录。

```bash
py/
├── CoLA/                      # CoLA数据集（原始）
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── SST-2/                     # SST-2数据集（原始）
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
├── processed_data/            # 预处理后的数据（由preprocess_data.py生成）
│   ├── cola_dev.tsv
│   ├── cola_test.tsv
│   ├── cola_train.tsv
│   ├── sst2_dev.tsv
│   ├── sst2_test.tsv
│   └── sst2_train.tsv
├── cola_lr_model/             # CoLA任务的逻辑回归模型参数
│   └── model_params.joblib
├── sst2_lr_model/             # SST-2任务的逻辑回归模型参数
│   └── model_params.joblib
├── cola_test_pred.tsv         # CoLA测试集预测结果
├── sst2_test_pred.tsv         # SST-2测试集预测结果
├── download_bert.py           # 下载BERT模型的脚本（模型未上传，请运行此脚本获取）
├── preprocess_data.py         # 数据预处理脚本（生成processed_data目录）
├── train_cola.py              # 训练CoLA任务模型的脚本
├── train_sst2.py              # 训练SST-2任务模型的脚本
├── predict.py                 # 预测脚本，加载模型进行推理
└── requirements.txt           # Python依赖包列表
