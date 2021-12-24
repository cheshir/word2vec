import gensim
import random


class Model:
    def __init__(self):
        self.model = gensim.models.Word2Vec.load('models/lurk_full.gensim.model.bin')

    def predict_next_word(self, context: str, topn=10):
        context = [self.normalize(word) for word in context.split()]
        return self.model.predict_output_word(context, topn=topn)

    def sentence_from_context(self, context: str, length=7):
        sentence = []
        for i in range(length):
            next_words = self.predict_next_word(context, topn=5)
            word = random.choice(next_words)[0] if len(next_words) > 0 else 'непонятно'
            sentence.append(word)
            context += " " + word

        return ' '.join(sentence)

    def most_similar(self, query: str):
        positive, negative = self.parse_most_similar_query(query)
        return self.model.wv.most_similar(positive=positive, negative=negative)

    def less_similar(self, words: str):
        return self.model.wv.doesnt_match(words.split())

    @staticmethod
    def parse_most_similar_query(query: str):
        tokens = query.lower().split()
        positive = []
        negative = []
        is_next_positive = True

        for t in tokens:
            if t == "+":
                is_next_positive = True
            elif t == "-":
                is_next_positive = False
            else:
                bucket = positive if is_next_positive else negative
                bucket.append(t)

        return positive, negative

    @staticmethod
    def normalize(word: str):
        return word.lower()
