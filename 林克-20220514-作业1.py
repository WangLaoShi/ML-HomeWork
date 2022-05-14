
#### 神经网络语言模型

顾名思义，神经网络语言模型肯定利用了神经网络模型。在神经网络语言模型中，开始出现词向量的概念，不同于统计语言模型利用词频计算句子的概率，神经网络语言模型利用上下文（有可能只用上文或只用下文，看模型的设计），来预测目标位置的词，通过反向梯度下降，让其尽可能的向目标词靠拢。  

> 词向量（Word embedding），又叫 Word 嵌入式自然语言处理（NLP）中的一组语言建模和特征学习技术的统称，其中来自词汇表的单词或短语被映射到实数的向量。 从概念上讲，它涉及从每个单词一维的空间到具有更低维度的连续向量空间的数学嵌入。

##### 词的表示

在自然语言处理任务中，首先需要考虑词如何在计算机中表示。通常，有两种表示方式：one-hot representation和distribution representation。

* 离散表示（one-hot representation）



例如：
苹果 [0，0，0，1，0，0，0，0，0，……]



* 分布式表示（distribution representation）

word embedding指的是将词转化成一种分布式表示，又称词向量。分布式表示将词表示成一个定长的连续的稠密向量。

分布式表示优点:
(1)词之间存在相似关系：
是词之间存在“距离”概念，这对很多自然语言处理的任务非常有帮助。
(2)包含更多信息：
词向量能够包含更多信息，并且每一维都有特定的含义。在采用one-hot特征时，可以对特征向量进行删减，词向量则不能。

##### 如何生成词向量










* 基于语言模型(language model)

语言模型生成词向量是通过训练神经网络语言模型NNLM（neural network language model），词向量做为语言模型的附带产出。NNLM背后的基本思想是对出现在上下文环境里的词进行预测，这种对上下文环境的预测本质上也是一种对共现统计特征的学习。
较著名的采用neural network language model生成词向量的方法有：Skip-gram、CBOW、LBL、NNLM、C&W、GloVe等。接下来，以目前使用最广泛CBOW模型为例，来介绍如何采用语言模型生成词向量。


比较经典的是 Word2vec 中的 CBOW 以及 skip-gram；当然还有最近比较火的 Bert 模型（利用Masked Language Model）、Xlnet（Permutation Language Model）等基于transformer的语言模型。







### 信息论



### NLP 的词向量表示

词向量表示模型比较基础的有 word of bag、word2vec 等，现在比较流行的还有Bert、Xlnet 等，先主要讲一下有名的 skip-gram 模型和 CBOW 模型，这俩都属于 word2vec 模型，同时总结一下比较有名的优化方法 hierarchical softmax 和 negative sampling。

Word2Vec 是从大量文本语料中以无监督的方式学习语义知识的一种模型，它被大量地用在自然语言处理（NLP）中。那么它是如何帮助我们做自然语言处理呢？Word2Vec 其实就是通过学习文本来用词向量的方式表征词的语义信息，即通过一个嵌入空间使得语义上相似的单词在该空间内距离很近。Embedding 其实就是一个映射，将单词从原先所属的空间映射到新的多维空间中，也就是把原先词所在空间嵌入到一个新的空间中去。

我们从直观角度上来理解一下，cat这个单词和kitten属于语义上很相近的词，而dog和kitten则不是那么相近，iphone这个单词和kitten的语义就差的更远了。通过对词汇表中单词进行这种数值表示方式的学习（也就是将单词转换为词向量），能够让我们基于这样的数值进行向量化的操作从而得到一些有趣的结论。比如说，如果我们对词向量kitten、cat以及dog执行这样的操作：kitten - cat + dog，那么最终得到的嵌入向量（embedded vector）将与puppy这个词向量十分相近。

Word2Vec模型中，主要有 Skip-Gram 和 CBOW 两种模型，从直观上理解，Skip-Gram 是给定 input word 来预测上下文。而 CBOW 是给定上下文，来预测input word。

![sXkeyn](https://upiclw.oss-cn-beijing.aliyuncs.com/uPic/sXkeyn.jpg)

#### Skip-Gram 

[参考地址](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)




#####
##### BP 算法

上面的例子是在已经学习好了中心词向量矩阵和上下文向量矩阵完成的，那么在这之前就要训练模型学习这两个矩阵，训练的过程就是一个BP算法。在学习的过程是一个监督学习，对于一个中心词我们是有明确的上下文的，我们期望窗口内的词的输出概率最大。在此之前我们学习另一种目标函数的表示方法。对于一个句子：S=（w1 w2 w3 w4 w5 w6）.有六个单词分别问 w1~w6，我们用P(D=1|wi,wj;θ)表示wi,wj作为上下文的概率，其中θ为模型参数，后文会接着解释模型参数。如果wi,wj作为上下文，我们就希望下面这个概率公式越大越好，其中概率公式表示是条件概率的逻辑回归表示，其中u和v分别代表中心词向量和上下文向量：

![HEUGrQ](https://upiclw.oss-cn-beijing.aliyuncs.com/uPic/HEUGrQ.jpg)

那么目标函数就是要最大化下面这个函数，其中 w 是文章 T 中的词，c 是中心词 w 对应的所有上下文单词：

![rXobtP](https://upiclw.oss-cn-beijing.aliyuncs.com/uPic/rXobtP.png)

对于刚刚说的要学习中心词向量矩阵W和上下文向量矩阵W’就是要更新目标函数的



#### CBOW

[[fdsahkjfhdsa]]

### FastText

FastText 是 Facebook 在 2016 年提出的一个文本分类算法，是一个有监督模型，其简单高效，速度快，在工业界被广泛的使用。在学术界，可以作为 baseline 的一个文本分类模型。

