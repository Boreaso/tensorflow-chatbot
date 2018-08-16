<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

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

训练

    python chatbot.py --mode train

推断，结果存放在outputs/infer_output.txt文件中

    python chatbot.py --mode infer

聊天交互

    python chatbot.py --mode chat  可以开始对话模式

如果想要训练其他数据集，需要按照corpus文件夹下的语料数据格式存放，使用utils/preprocess.py进行语聊数据预处理，然后进行训练

## 网络架构
Seq2seq网络架构如下：

![Seq2seq网络架构](https://github.com/Boreaso/tensorflow-chatbot/raw/master/images/seq2seq_architecture.png)

如图所示，模型接受一个序列输入“ABC”，编码-解码操作产生一个序列输出“WXYZ”。<EOS\>（End Of Sequence）用作模型预测的定界符，是用户指定的特殊字符，不包含在所要训练的数据词汇表中。当模型解码遇到<EOS\>就不再继续进行预测。编码器输入是未经Padding的原始串‘ABC’。在训练阶段，target input被Padding为“<EOS>WXYZ”作为每个时间步Decoder的输入, target output被Padding为“WXYZ<EOS>”作为优化目标输出（Label）。在用已有的模型进行推断时，<EOS\>是做为整个解码操作的初始输入，加上Encoder的final_state一同作为Decoder Cell的初始输入进行解码操作，所以如果把Encoder的finalstate记为Decoder的state_0，第一次解码输出记做output_0，第一次解码状态记做state_1，以此类推，那么整个解码流程的输入序列是(<EOS\>，state_0)->(output0, state_1)->(output_1，state_2)->……直到output_n为<EOS\>。

Seq2seq的工作流程分为编码阶段和解码阶段。在编码阶段，处于编码结构的LSTM通过计算得到一个固定维度大小的特征表示v（LSTM的最终状态或由注意力机制引入的所有状态的加权平均,维数为指定的RNN隐层单元个数）。在解码阶段，处于解码器结构中的LSTM以v作为初始状态，对下一时刻的序列元素进行预测，每个时刻可能出现的概率最大的元素将被选择(此处也可以引入BeamSearch)。

## 详细文档

详细的资料整理在doc目录下的文档中.
