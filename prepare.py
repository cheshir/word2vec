import cli
import nltk
import re
import tqdm
import os
import sys


sentence_sanitize_re = re.compile(r'^[^a-zа-я]*', re.UNICODE | re.IGNORECASE)


class SentencesReader:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        t = tqdm.tqdm(total=os.path.getsize(self.filename))

        for line in open(self.filename, 'r').readlines():
            t.update(sys.getsizeof(line))
            line = self.sanitize_sentence(line)

            try:
                sentences = nltk.sent_tokenize(line, language='russian')
            except IndexError:
                print(f"Tokenize failed on the line '{line}'")
                continue

            if len(sentences) == 0:
                continue

            for s in sentences[:-1]:
                s = self.sanitize_sentence(s)
                if self.is_valid_sentence(s):
                    yield s.strip()

            last_sentence = sentences[-1].strip()

            if self.is_valid_sentence(last_sentence):
                yield last_sentence

        t.close()

    @staticmethod
    def sanitize_sentence(line):
        return sentence_sanitize_re.sub('', line)

    @staticmethod
    def is_valid_sentence(sentence):
        return sentence.count(' ') > 1

    @staticmethod
    def is_sentence_completed(sentence):
        return sentence.endswith(('.', '!', '?'))

    @staticmethod
    # @link https://www.kite.com/python/answers/how-to-check-if-a-string-contains-letters-in-python
    def is_string_contains_letters(s):
        return s.lower().islower()


if __name__ == '__main__':
    killer = cli.GracefulKiller()
    reader = SentencesReader('./data/source_full.txt')

    with open('data/sentences_full.txt', 'w', encoding='utf-8') as f:
        for sentence in reader:
            f.write(sentence + '\n')
            if killer.kill_now:
                break
