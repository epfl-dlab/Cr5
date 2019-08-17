import os
import io
import numpy as np
import gzip

from collections import Counter

class Cr5_Model:
    def __init__(self, data_folder_path=None, model_name=None):
        if data_folder_path is None:
            self.data_folder_path = ''
        else:
            self.data_folder_path = data_folder_path

        if model_name is None:
            self.model_name = ''
        else:
            self.model_name = model_name

        self.embs_per_lang = {}
        self.lang_codes = []

    def read_embs(self, lang_code):
        lang_data_path = self.get_lang_data_path(lang_code)
        word_2_emb = {}

        with gzip.open(lang_data_path, 'rt', encoding='utf-8') as file:
            for line in file:
                parts = line.split(" ")
                word = ' '.join(parts[:-300])
                vec = parts[-300:]
                emb = np.array(vec, dtype=np.float64)
                word_2_emb[word] = emb

        return word_2_emb

    def get_lang_data_path(self, lang_code):
        lang_data_path = os.path.join(self.data_folder_path, '{}_{}.txt.gz'.format(self.model_name, lang_code))

        return lang_data_path


    def load_langs(self, lang_codes):
        for lang_code in lang_codes:
            if not os.path.isfile(self.get_lang_data_path(lang_code)):
                raise Exception("The model for language code `{}` is not available in the folder at `{}`.".format(lang_code, self.get_lang_data_path(lang_code)))
        self.lang_codes = lang_codes

        for lang_code in lang_codes:
            self.embs_per_lang[lang_code] = self.read_embs(lang_code)

    def get_document_embedding(self, tokens, lang_code):
        if lang_code not in self.lang_codes:
            raise Exception("Model for language code `{}` has not been loaded.".format(lang_code))

        tf_tokens = dict(Counter(tokens))

        words_in_vocab = [word for word in tf_tokens.keys() if word in self.embs_per_lang[lang_code]]

        if len(words_in_vocab) == 0:
            raise Exception("No matching tokens with the vocabulary were found.")

        tfs = np.array([tf_tokens[word] for word in words_in_vocab])
        embs = np.array([self.embs_per_lang[lang_code][word] for word in words_in_vocab])

        normalized_tfs = tfs / np.linalg.norm(tfs)

        for i in range(len(tfs)):
            embs[i] = embs[i] * tfs[i]

        doc_emb = embs.sum(axis=0)
        doc_emb = doc_emb / np.linalg.norm(tfs)

        return doc_emb
    
    def tokenize(self, text):
        import nltk
        try:
            return nltk.tokenize.word_tokenize(text)
        except LookupError as e:
            nltk.download('punkt')
            return nltk.tokenize.word_tokenize(text) 