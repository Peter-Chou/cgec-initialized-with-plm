# 中文语法纠错

## 环境准备

### 模型训练/测试环境准备

Python版本：3.8

``` bash
# 在项目根目录下运行
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

其中 torch==1.7.1+cu101, transformers==4.3.3, pypinyin==0.40.0, jieba==0.42.1, sentencepiece==0.1.95

### 模型评估环境准备

下载评估模型所需的docker镜像

``` bash
docker pull zhou00009/m2scorer-chinese:1.1.0
```

或者可以手工构建该镜像，两者选其一

``` bash
# 在项目根目录下运行
cd ./docker
docker build -t zhou00009/m2scorer-chinese:1.1.0 -f m2scorer-chinese.dockerfile .
```

## 下载数据

``` bash
# 在项目根目录下运行
python download_data.py
```

## 数据预处理

``` bash
# 在项目根目录下运行
python preprocess_data.py
```

## 训练模型

``` bash
# 模型训练(初筛阶段/正式训练阶段)
model_name=model_name_you_select  # e.g. roberta_roberta
stage=stage_of_your_model  # stage 可以是 preselect / formal
# 在项目根目录下运行
./scripts/train_model.sh --mode=${stage} --model_name=${model_name}
```

## 模型评估

### 获得模型在测试集上的预测结果

``` bash
model_name=model_name_you_select
stage=stage_of_your_model
epoch_num=best_ckpt_folder_name  # e.g. epoch-6, epoch-10
# 在项目根目录下运行
./scripts/eval_model.sh --model_name=${model_name} --mode=${stage} --ckpt=${epoch_num}
```

模型在测试集上的预测会在`./evaluation/${model_name}/${stage}/${model_name}_${stage}_${epoch_num}.txt`文件里。

### 评估模型

``` bash
model_name=model_name_you_select
stage=stage_of_your_model
epoch_num=best_ckpt_folder_name
# 在项目根目录下运行
docker run --rm --user $(id -u):$(id -g) \
	--mount type=bind,source=`pwd`/evaluation,target=/evaluation \
	-t zhou00009/m2scorer-chinese:1.1.0 \
	evaluation/${model_name}/${stage}/${model_name}_${stage}_${epoch_num}.txt
```

模型在测试集上表现的评估结果会在`./evaluation/${model_name}/${stage}/${model_name}_${stage}_${epoch_num}_score_result`文件里。


## 实验结果

### 初次筛选阶段

#### 消融实验

| model name             |   P   |   R   | <img src="https://latex.codecogs.com/svg.image?\inline&space;\mathbf{F_{0.5}}" title="\inline \mathbf{F_{0.5}}" height=13 /> |
| ---------------------- | :---: | :---: | :--------------------------------------------------------------------------------------------------------------------------: |
| RoBERTa_RoBERTa        | 32.86 | 24.62 |                                                            30.80                                                             |
| RoBERTa_RoBERTa_share  | 32.80 | 24.54 |                                                            30.73                                                             |
| RoBERTa_RoBERTa_nomask | 29.76 | 20.07 |                                                            27.14                                                             |



#### 使用decoder异构分别初始化Transformer方法的模型表现

| model name            |   P   |   R   | <img src="https://latex.codecogs.com/svg.image?\inline&space;\mathbf{F_{0.5}}" title="\inline \mathbf{F_{0.5}}" height=13 /> |
| --------------------- | :---: | :---: | :----------------------------------------------------------: |
| **ERNIE_ERNIE_share** | 35.42 | 24.78 |                          **32.62**                           |
| ERNIE_ERNIE           | 34.71 | 25.92 |                            32.50                             |
| RoBERTa_RoBERTa       | 32.86 | 24.62 |                            30.80                             |
| MacBERT_MacBERT       | 32.57 | 23.85 |                            30.35                             |
| WoBERT_WoBERT         | 30.62 | 26.75 |                            29.76                             |
| UniLM_UniLM           | 31.03 | 23.46 |                            29.15                             |

#### 使用decoder同构分别初始化Transformer方法的模型表现

| model name       |   P   |   R   | <img src="https://latex.codecogs.com/svg.image?\inline&space;\mathbf{F_{0.5}}" title="\inline \mathbf{F_{0.5}}" height=13 /> |
| ---------------- | :---: | :---: | :----------------------------------------------------------: |
| **MacBERT_GPT2** | 29.89 | 27.96 |                          **29.48**                           |
| RoBERTa_GPT2     | 29.73 | 27.80 |                            29.32                             |
| ERNIE_GPT2       | 29.01 | 28.13 |                            28.83                             |
| WoBERT_GPT2      | 28.16 | 30.14 |                            28.53                             |
| UniLM_GPT2       | 28.74 | 25.89 |                            28.12                             |

#### 使用直接初始化Transformer方法到模型表现

| model name |   P   |   R   | <img src="https://latex.codecogs.com/svg.image?\inline&space;\mathbf{F_{0.5}}" title="\inline \mathbf{F_{0.5}}" height=13 /> |
| ---------- | :---: | :---: | :----------------------------------------------------------: |
| **T5**     | 39.19 | 25.26 |                          **35.30**                           |
| BART       | 37.18 | 24.43 |                            33.66                             |


### 正式训练阶段

| model name        |   P   |   R   | <img src="https://latex.codecogs.com/svg.image?\inline&space;\mathbf{F_{0.5}}" title="\inline \mathbf{F_{0.5}}" height=13 /> |
| ----------------- | :---: | :---: | :----------------------------------------------------------: |
| **ERNIE_ERNIE**   | 47.75 | 26.35 |                          **41.08**                           |
| ERNIE_ERNIE_share | 48.10 | 25.19 |                            40.70                             |
| T5                | 44.11 | 28.04 |                            39.58                             |
| MacBERT_GPT2      | 41.59 | 27.68 |                            37.79                             |
