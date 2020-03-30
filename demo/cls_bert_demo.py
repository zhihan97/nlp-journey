from smartnlp.classfication.dl.bert_classifier import BertTextClassifier

if __name__ == '__main__':
    bert_classifier = BertTextClassifier('../tutorials/03.poem_generation/data/bert/chinese/bert_config.json',
                                         '../tutorials/03.poem_generation/data/bert/chinese/bert_model.ckpt',
                                         '../tutorials/03.poem_generation/data/bert/chinese/vocab.txt',
                                         True,
                                         '../tutorials/03.poem_generation/data/bert/sentiment.txt')
