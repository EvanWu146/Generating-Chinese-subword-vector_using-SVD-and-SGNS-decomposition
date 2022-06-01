# 基于 SVD 分解以及基于 SGNS 两种⽅法构建汉语⼦词向量并进⾏评测

## 算法说明

1. 调⽤库：

```python
import numpy as np 
import pandas as pd 
import pickle 
from tqdm import tqdm 
from sklearn.decomposition import TruncatedSVD 
import multiprocessing 
from gensim.models import Word2Vec 
from gensim.models.word2vec import LineSentence 
import logging
```



2. 设计WordEmbedding类，包含数据的初始化处理以及SVD和SGNS两种⽅法对汉语⼦ 词向量评测的功能：

```python
class WordEmbedding
```

3. 类的初始化：

    ```python
     def get_vocab(self) # 读取词表，词表来源于上⼀次作业所构建的词表，以列表-字符串的形式存储，⽤pickle模型⽂ 件保存并读取，⽂件名为：vocab0。
      def load_lines(self) # 读取语料⾏，语料内容来⾃于上⼀次作业的test_BPE.txt的分词结果，读⼊的时候将内容⾏⽤ split()⽅法分词。
    
    def get_word_vector(self) # 获取词向量，从⽂件中获得需要评测的词向量，每⾏⽤split⽅法分隔开两个词。
    
    def get_subword_vector(self, window_size) # 获取⼦词向量，从处理好的语料⾏中以给定的窗⼝⼤⼩（默认为5）获取组合的词对。对于任意 ⼦词向量对中⻓度⼤于4的词，将会被裁剪。最后需要添加所有⼦词向量的逆向量
    ```

4. 主要处理函数：

```python
def SVG_process(self) # SVG⽅法 
"""

1. 调⽤get_subword_vector函数获取⼦词向量

2. 构建词表⻓度的numpy全0⽅形矩阵，并转换为dataframe形式

3. 利⽤dataframe的字符串索引功能，使⽤⼦词向量进⾏计数，记录⼦词向量在词表中的出现频 率

4. 将dataframe转为numpy矩阵形式，使⽤

TruncatedSVD(n_components=3).fit_transform()进⾏SVD分解，其中选取的奇异值个数 为3

5. 对待评测的词向量求相似度：先判断词向量是否在词表⾥，若在则计算其余弦相似度（两个词 的特征向量的点积除以它们的L2范式积）

6. 将相似度值列表保存到模型⽂件中。 """ 

def SGNS_process(self) # SGNS⽅法 
"""
 1. 调⽤Word2Vec(LineSentence('dataset.txt'), vector_size=100, window=2,

sg=1, hs=0, min_count=1,

workers=multiprocessing.cpu_count())，获取初始词向量。 所⽤初始词向量来源：同SVG，来⾃于上⼀次作业的test_BPE.txt的分词结果。 词向量维数：100（默认） 总共进⾏了5轮训练，每次训练的原始词数（批次⼤⼩）为1334584词，学习效率分别为（1到5 轮）：2033311 effective words/s，2075792 effective words/s，2028630 effective words/s，2018426 effective words/s，2093805 effective words/s

2. 对待评测的词向量，调⽤vec_sgns.wv.similarity(word[0], word[1])，求取相似度

3. 将相似度值列表保存到模型⽂件中。
"""
```