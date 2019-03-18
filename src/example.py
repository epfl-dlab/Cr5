import numpy as np
from cr5 import Cr5_Model
from scipy.spatial import distance
from collections import Counter

model = Cr5_Model('./model_28_txt/','joint_28') # path_to_pretrained_model, model_prefix
model.load_langs(['en', 'it']) # list_of_languages

tokens = ['beautiful', 'landscape'] # list_of_tokens_contained_in_en_document
tokens_it = ['bel', 'paesaggio'] # list_of_tokens_contained_in_it_document

en_doc_emb = model.get_document_embedding(tokens, 'en')
it_doc_emb = model.get_document_embedding(tokens_it, 'it')