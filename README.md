# TextClassfication

# **TextClassfication**

---

中文文本分类；bert；Pytorch

# **介绍**

---

模型： bert-chinese-base

机器：window；3070Ti

# 环境

---

python3.8

torch==1.13.1+cu116

transformers==4.31.0

# **数据集**

---

今日头条文本分类数据集

地址：https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset

# 使用说明

---

1. 准备数据
    
    下载数据：https://github.com/aceimnorstuvwxz/toutiao-text-classfication-dataset
    
    处理数据：运行`data\toutiaonews38w\data_processor.py` 生成 `train.csv`，`dev.csv`
    
2. 配置参数：
    
    请在`config.py`中配置参数
    
3. 训练：
    
    ```python
    python run.py
    ```
    
4. 推理：
    
    ```python
    python predict.py
    ```
    

<aside>
💡 注意，为了快速验证代码是否可以运行，我在`utils/data_utils.py`中加载数据时，只截取了2000条数据

</aside>

# 未完待续

---

todo：添加更多模型

# 参考

---

https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch