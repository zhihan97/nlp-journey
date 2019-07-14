from nlp.classfication.dl.bert_classifier import BertTextClassifier

if __name__ == '__main__':
    bert_classifier = BertTextClassifier('./data/bert/chinese/bert_config.json',
                                         './data/bert/chinese/bert_model.ckpt',
                                         './data/bert/chinese/vocab.txt',
                                         True,
                                         './data/bert/sentiment.txt')
