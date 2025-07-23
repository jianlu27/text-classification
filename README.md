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

## 🔁 复现说明（注意项目包含已运行代码完成的文件夹）

1. 克隆本项目后，请先下载 BERT 模型（因模型文件较大未纳入仓库）：

    ```bash
    python download_bert.py
    ```

2. 运行预处理脚本，提取 BERT 特征并生成训练/验证/测试数据：

    ```bash
    python preprocess_data.py
    ```

3. 训练模型（将自动保存模型参数至 `cola_lr_model/` 和 `sst2_lr_model/`）：

    ```bash
    python train_cola.py
    python train_sst2.py
    ```

    > ⚠️ 若模型文件已存在，训练脚本会覆盖原模型参数，如无需重新训练请先备份。

4. 对测试集进行预测：

    ```bash
    python predict.py
    ```

    > 默认输出文件为：  
    > - `cola_test_pred.tsv`  
    > - `sst2_test_pred.tsv`
