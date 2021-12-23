import gensim
import re

sanitize_re = re.compile('[^\sa-zа-я]+', re.UNICODE | re.IGNORECASE)


class SentencesReader:
    def __init__(self, doc_path):
        self.doc_path = doc_path

    def __iter__(self):
        for sentence in open(self.doc_path, 'r').readlines():
            preprocessed = gensim.utils.simple_preprocess(sentence)
            if len(preprocessed) > 0:
                yield preprocessed

    @staticmethod
    def sanitize(string):
        return sanitize_re.sub('', string)


def train(doc_path, model_path):
    corpus = SentencesReader(doc_path)
    model = gensim.models.Word2Vec(sentences=corpus)
    model.save(model_path)


def load(model_path):
    return gensim.models.Word2Vec.load(model_path)


if __name__ == '__main__':
    train('data/source_full.txt', 'models/lurk_full.gensim.model.bin')
