本项目使用TensorFlow实现了一个简易的聊天机器人

## 项目结构

* **corpus**：存放语料数据
* **data**：存放经过预处理的训练数据
* **doc**：存放资料文档
* **hparams**：存放预定义的超参数json文件
* **models**：
  * **basic_model.py**：定义了seq2seq model基础架构，包含Base和BasicModel两个类，BasicModel继承Base实现了build_encoder和_create_decoder_cell两个抽象方法
  * **attention_model.py**：继承basic_model，重写了_create_decoder_cell方法，加入AttentionMechanism。
  * **model_helper.py**：model创建所需的基础方法封装。
* **utils**：
  * **eval_utils.py**：模型评估计算方法封装
  * **iterator.py**：数据迭代器封装，作为model的参数传入。
  * **misc_utils.py**：各种各样的杂项操作封装
  * **param_utils.py**：Python参数解析操作封装
  * **preprocess_util.py**：数据预处理封装
  * **train_utils.py**：训练所需的辅助方法封装
  * **vocabulary.py**：词汇表封装，作为model的参数传入
* **chatbot.py**：封装了模型训练、推断、对话的总体流程

## 使用

corpus和data文件夹已经预置了一些语料数据，shell进入项目顶级目录，运行命令：

* python chatbot.py --mode train即可开始训练
* python chatbot.py --mode infer开始推断，结果存放在outputs/infer_output.txt文件中
* python chatbot.py --mode chat可以开始对话模式

如果想要训练其他数据集，需要按照corpus文件夹下的语料数据格式存放，使用utils/preprocess.py进行语聊数据预处理，然后进行训练

## 文档

详细的资料整理在doc目录下的文档中.
