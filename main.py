import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging


class WordEmbedding:
    def __init__(self):
        self.vocab = []
        self.load_lines()
        self.get_vocab()
        self.get_word_vector()
        pass

    def get_vocab(self):  # 读取词表
        with open('vocab0', 'rb') as f:
            self.vocab = pickle.load(f)
            self.vocab = list(set(self.vocab))
            f.close()
        print("The number of the vocabulary: ", len(self.vocab))

    def load_lines(self):  # 读取语料行
        f = open('dataset.txt')

        self.raw_text = f.readlines()
        self.text = []
        f.close()

        for i in self.raw_text:  # 将语料行分词
            self.text.append(i.split())

        print("The length of the text lines: ", len(self.text))

    def get_word_vector(self):
        # 获取词向量，从文件中获得需要评测的词向量，每行用split方法分隔开两个词。
        with open('pku_sim_test.txt', 'r') as f:
            temp = f.readlines()
            self.word_vector = []
            for t in temp:
                self.word_vector.append(t.split())
            print("The length of the word vector: ", len(self.word_vector))
            f.close()

    def get_subword_vector(self, window_size=5):
        # 获取子词向量，从处理好的语料行中以给定的窗口大小（默认为5）获取组合的词对
        self.subword_vector = []
        print("Start to initial the subword vector...")
        tbar = tqdm(total=len(self.text))
        for i in self.text:
            for k in range(len(i) - window_size + 1):
                for l in range(k + 1, k + window_size + 1):
                    if l <= len(i) - 1:
                        former = i[k]
                        latter = i[l]
                        # 对于任意子词向量对中长度大于4的词，将会被裁剪
                        if len(former) > 4:
                            former = former[-4:]
                        if len(latter) > 4:
                            latter = latter[:4]
                        self.subword_vector.append([former, latter])
            tbar.update(1)
        tbar.close()
        # 最后需要添加所有子词向量的逆向量
        temp = [x[::-1] for x in self.subword_vector]
        self.subword_vector.extend(temp)

    def SVG_process(self):  # SVG方法
        self.get_subword_vector()
        M = np.zeros((len(self.vocab), len(self.vocab)))
        df = pd.DataFrame(M, index=self.vocab, columns=self.vocab)
        print("Calculating the subword vector...")
        # 利用dataframe的字符串索引功能，使用子词向量进行计数，记录子词向量在词表中的出现频率
        tbar = tqdm(total=len(self.subword_vector))
        for i in self.subword_vector:
            try:
                df.at[i[0], i[1]] += 1
            except:
                pass
            tbar.update(1)
        tbar.close()

        M = np.array(df)
        print(np.max(M))

        svd = TruncatedSVD(n_components=3)
        self.result = svd.fit_transform(M)
        print(self.result.shape)

        self.sim_svd = []
        print("Calculating the sim_svd...")
        tbar = tqdm(total=len(self.word_vector))
        for word in self.word_vector:
            if word[0] in self.vocab and word[1] in self.vocab:
                idx0 = self.vocab.index(word[0])
                idx1 = self.vocab.index(word[1])
                vector0 = self.result[idx0]
                vector1 = self.result[idx1]
                norm0 = np.linalg.norm(vector0)
                norm1 = np.linalg.norm(vector1)
                frac0 = np.dot(vector0, vector1)
                frac1 = norm0 * norm1
                if frac1 != 0:
                    sim = np.around(frac0 / frac1, 1)
                else:
                    sim = 0
                self.sim_svd.append(sim)
            else:
                self.sim_svd.append(0)
            tbar.update(1)
        tbar.close()
        # print(self.sim_svd)
        f = open('sim_svd', 'wb')
        pickle.dump(self.sim_svd, f)
        f.close()

    def SGNS_process(self):
        print("Calculating the sim_sgns...")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.sim_sgns = []
        vec_sgns = Word2Vec(LineSentence('dataset.txt'), vector_size=100, window=2, sg=1, hs=0, min_count=1,
                         workers=multiprocessing.cpu_count())
        tbar = tqdm(total=len(self.word_vector))
        for word in self.word_vector:
            try:
                self.sim_sgns.append(vec_sgns.wv.similarity(word[0], word[1]))
            except:
                self.sim_sgns.append(0)
            tbar.update(1)
        tbar.close()
        # print(self.sim_sgns)
        f = open('sim_sgns', 'wb')
        pickle.dump(self.sim_sgns, f)
        f.close()

    def output(self):
        f = open('2019211333.txt', 'w', encoding='utf-8')
        with open('sim_svd', 'rb') as f1:
            self.sim_svd = pickle.load(f1)
            f1.close()

        with open('sim_sgns', 'rb') as f1:
            self.sim_sgns = pickle.load(f1)
            f1.close()

        for i in range(500):
            f.write(str(' '.join(self.word_vector[i])) + '    ' + str(self.sim_svd[i])
                    + '    ' + str(self.sim_sgns[i]) + '\n')

        f.close()




example = WordEmbedding()
example.SVG_process()
# example.SGNS_process()
# example.output()