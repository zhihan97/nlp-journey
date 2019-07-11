from gensim.models import LdaModel


class LdaTopicModel:

    def __init__(self, model_path,
                 train=False,
                 file_path=None):
        self.model_path = model_path
        if not train:
            self.model = self.load_model()
        else:
            self.file_path = file_path
            self.model = self.train()

    def train(self):

        model = LdaModel(self, num_topics=10)
        return model

    def save_model(self):
        self.model.save(self.model_path)
        pass

    def load_model(self):
        try:
            model = LdaModel.load(self.model_path)
        except FileNotFoundError:
            model = None
        return model

    def preprocess(self):
        pass
