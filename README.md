# nlp journey

> Your Journey to NLP Starts Here ! 

[![Star](https://img.shields.io/github/stars/msgi/nlp-journey?color=success)](https://github.com/msgi/nlp-journey/)
[![Fork](https://img.shields.io/github/forks/msgi/nlp-journey)](https://github.com/msgi/nlp-journey/fork)
[![GitHub Issues](https://img.shields.io/github/issues/msgi/nlp-journey?color=success)](https://github.com/msgi/nlp-journey/issues)
[![License](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/msgi/nlp-journey)

***全面拥抱tensorflow 2.0，代码全部修改为tensorflow 2.0版本。***

**安装方式：**

```shell script
pip install funnlp
```

## 一. 基础知识

* [基础知识](docs/basic.md)
* [工具教程](tutorials/)
* [实践笔记](docs/notes.md)
* [常见问题](docs/fq.md)
* [实现代码](funnlp/)

## 二. 经典书目([`百度云`](https://pan.baidu.com/s/14z5SnM28guarUZfZihdTPw) 提取码：txqx)

1. 概率图入门. [`原书地址`](https://stat.ethz.ch/~maathuis/papers/Handbook.pdf)
2. Deep Learning.深度学习必读. [`原书地址`](https://www.deeplearningbook.org/)
3. Neural Networks and Deep Learning. 入门必读. [`原书地址`](http://neuralnetworksanddeeplearning.com/)
4. 斯坦福大学《语音与语言处理》第三版：NLP必读. [`原书地址`](http://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)

## 三. 必读论文

### 01) 必读NLP论文

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. [GPT: Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
3. [GPT-2: Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
4. [Transformer-XL: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
5. [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
6. [XLM: Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau.
7. [RoBERTa: Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
8. [DistilBERT: a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) by Victor Sanh, Lysandre Debut and Thomas Wolf. 
9. [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
10. [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) by Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
11. [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
12. [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.
13. [XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) by Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov.
14. [MMBT: Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/pdf/1909.02950.pdf) by Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Davide Testuggine.
15. [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.


### 02) 模型及优化

1. LSTM(Long Short-term Memory). [`地址`](http://www.bioinf.jku.at/publications/older/2604.pdf)
2. Sequence to Sequence Learning with Neural Networks. [`地址`](https://arxiv.org/pdf/1409.3215.pdf)
3. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. [`地址`](https://arxiv.org/pdf/1406.1078.pdf)
4. Residual Network(Deep Residual Learning for Image Recognition). [`地址`](https://arxiv.org/pdf/1512.03385.pdf)
5. Dropout(Improving neural networks by preventing co-adaptation of feature detectors). [`地址`](https://arxiv.org/pdf/1207.0580.pdf)
6. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. [`地址`](https://arxiv.org/pdf/1502.03167.pdf)

### 03) 综述论文

1. An overview of gradient descent optimization algorithms. [`地址`](https://arxiv.org/pdf/1609.04747.pdf)
2. Analysis Methods in Neural Language Processing: A Survey. [`地址`](https://arxiv.org/pdf/1812.08951.pdf)
3. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. [`地址`](https://arxiv.org/pdf/1910.10683.pdf)
4. A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications. [`地址`](https://arxiv.org/pdf/2001.06937.pdf)
5. A Gentle Introduction to Deep Learning for Graphs. [`地址`](https://arxiv.org/pdf/1912.12693.pdf)

### 04) 文本预训练

1. A Neural Probabilistic Language Model. [`地址`](https://www.researchgate.net/publication/221618573_A_Neural_Probabilistic_Language_Model)
2. word2vec Parameter Learning Explained. [`地址`](https://arxiv.org/pdf/1411.2738.pdf)
3. Language Models are Unsupervised Multitask Learners. [`地址`](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
4. An Empirical Study of Smoothing Techniques for Language Modeling. [`地址`](https://dash.harvard.edu/bitstream/handle/1/25104739/tr-10-98.pdf?sequence=1)
5. Efficient Estimation of Word Representations in Vector Space. [`地址`](https://arxiv.org/pdf/1301.3781.pdf)
6. Distributed Representations of Sentences and Documents. [`地址`](https://arxiv.org/pdf/1405.4053.pdf)
7. Enriching Word Vectors with Subword Information(FastText). [`地址`](https://arxiv.org/pdf/1607.04606.pdf). [`解读`](https://www.sohu.com/a/114464910_465975)
8. GloVe: Global Vectors for Word Representation. [`官网`](https://nlp.stanford.edu/projects/glove/)
9. ELMo (Deep contextualized word representations). [`地址`](https://arxiv.org/pdf/1802.05365.pdf)
10. Pre-Training with Whole Word Masking for Chinese BERT. [`地址`](https://arxiv.org/pdf/1906.08101.pdf)

### 05) 文本分类

1. Bag of Tricks for Efficient Text Classification (FastText). [`地址`](https://arxiv.org/pdf/1607.01759.pdf)
2. Convolutional Neural Networks for Sentence Classification. [`地址`](https://arxiv.org/pdf/1408.5882.pdf)
3. Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification. [`地址`](http://www.aclweb.org/anthology/P16-2034)

### 06) 文本生成

1. A Deep Ensemble Model with Slot Alignment for Sequence-to-Sequence Natural Language Generation. [`地址`](https://arxiv.org/pdf/1805.06553.pdf)
2. SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient. [`地址`](https://arxiv.org/pdf/1609.05473.pdf)

### 07) 文本相似性

1. Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks. [`地址`](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.6492&rep=rep1&type=pdf)
2. Learning Text Similarity with Siamese Recurrent Networks. [`地址`](https://www.aclweb.org/anthology/W16-1617)
3. A Deep Architecture for Matching Short Texts. [`地址`](http://papers.nips.cc/paper/5019-a-deep-architecture-for-matching-short-texts.pdf)

### 08) 自动问答

1. A Question-Focused Multi-Factor Attention Network for Question Answering. [`地址`](https://arxiv.org/pdf/1801.08290.pdf)
2. The Design and Implementation of XiaoIce, an Empathetic Social Chatbot. [`地址`](https://arxiv.org/pdf/1812.08989.pdf)
3. A Knowledge-Grounded Neural Conversation Model. [`地址`](https://arxiv.org/pdf/1702.01932.pdf)
4. Neural Generative Question Answering. [`地址`](https://arxiv.org/pdf/1512.01337v1.pdf)
5. Sequential Matching Network A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots．[`地址`](https://arxiv.org/abs/1612.01627)
6. Modeling Multi-turn Conversation with Deep Utterance Aggregation．[`地址`](https://arxiv.org/pdf/1806.09102.pdf)
7. Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network．[`地址`](https://www.aclweb.org/anthology/P18-1103)
8. Deep Reinforcement Learning For Modeling Chit-Chat Dialog With Discrete Attributes. [`地址`](https://arxiv.org/pdf/1907.02848.pdf)

### 09) 机器翻译

1. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. [`地址`](https://arxiv.org/pdf/1406.1078v3.pdf)
2. Neural Machine Translation by Jointly Learning to Align and Translate. [`地址`](https://arxiv.org/pdf/1409.0473.pdf)
3. Transformer (Attention Is All You Need). [`地址`](https://arxiv.org/pdf/1706.03762.pdf)
4. Transformer-XL:Attentive Language Models Beyond a Fixed-Length Context. [`地址`](https://arxiv.org/pdf/1901.02860.pdf)

### 10) 自动摘要

1. Get To The Point: Summarization with Pointer-Generator Networks. [`地址`](https://arxiv.org/pdf/1704.04368.pdf)
2. Deep Recurrent Generative Decoder for Abstractive Text Summarization. [`地址`](https://aclweb.org/anthology/D17-1222)

### 11) 关系抽取

1. Distant Supervision for Relation Extraction via Piecewise Convolutional Neural Networks. [`地址`](https://www.aclweb.org/anthology/D15-1203)
2. Neural Relation Extraction with Multi-lingual Attention. [`地址`](https://www.aclweb.org/anthology/P17-1004)
3. FewRel: A Large-Scale Supervised Few-Shot Relation Classification Dataset with State-of-the-Art Evaluation. [`地址`](https://aclweb.org/anthology/D18-1514)
4. End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures. [`地址`](https://www.aclweb.org/anthology/P16-1105)

## 四. 必读博文

1. 应聘机器学习工程师？这是你需要知道的12个基础面试问题. [`地址`](https://www.jiqizhixin.com/articles/2020-01-06-9)
2. 如何学习自然语言处理（综合版）. [`地址`](https://mp.weixin.qq.com/s/lJYp4hUZVsp-Uj-5NqoaYQ)
3. The Illustrated Transformer.[`地址`](https://jalammar.github.io/illustrated-transformer/)
4. Attention-based-model. [`地址`](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)
5. Modern Deep Learning Techniques Applied to Natural Language Processing. [`地址`](https://nlpoverview.com/)
6. Bert解读. [`地址`](https://zhuanlan.zhihu.com/p/49271699)
7. 难以置信！LSTM和GRU的解析从未如此清晰（动图+视频）。[`地址`](https://blog.csdn.net/dqcfkyqdxym3f8rb0/article/details/82922386)
8. 深度学习中优化方法. [`地址`](https://blog.csdn.net/u012328159/article/details/80311892)
9. 从语言模型到Seq2Seq：Transformer如戏，全靠Mask. [`地址`](https://spaces.ac.cn/archives/6933)
10. Applying word2vec to Recommenders and Advertising. [`地址`](http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/)
11. 2019 NLP大全：论文、博客、教程、工程进展全梳理. [`地址`](https://zhuanlan.zhihu.com/p/108442724)

## 五. 相关优秀github项目

* transformers. [`地址`](https://github.com/huggingface/transformers)

> [`一份教程`](https://rsilveira79.github.io/fermenting_gradients/machine_learning/nlp/pytorch/pytorch-transformer-squad/)

* HanLP. [`地址`](https://github.com/hankcs/HanLP)

## 六. 相关优秀博客

* [52nlp](http://www.52nlp.cn/)
* [科学空间/信息时代](https://kexue.fm/category/Big-Data)
* [刘建平Pinard](https://www.cnblogs.com/pinard/)
* [零基础入门深度学习](https://www.zybuluo.com/hanbingtao/note/433855)
