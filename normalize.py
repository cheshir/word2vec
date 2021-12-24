import nltk
from nltk.corpus import stopwords


def word(w: str, skip_stop_words=False):
    w = w.lower()
    if not skip_stop_words:
        return w
    return word if word not in stopwords.words('russian') and word not in stopwords.words('english') else None


def word_list(wl: list, skip_stop_words=False):
    normalized = []
    for w in wl:
        norm_word = word(w, skip_stop_words)
        if norm_word is not None:
            normalized.append(norm_word)

    return normalized


def sentence(s: str, skip_stop_words=False):
    normalized = []
    for w in nltk.word_tokenize(s, language='russian'):
        norm_word = word(w, skip_stop_words)
        if norm_word is not None:
            normalized.append(norm_word)

    return ' '.join(normalized)
