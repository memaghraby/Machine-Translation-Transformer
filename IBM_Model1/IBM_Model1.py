
import string
import pathlib
import json
from tensorflow import keras
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import math
# nltk.download('punkt')

class IBM:
  MAX_NUM_OF_ITERATIONS = 2
  EPSILON = 1e-20
  def __init__(self, isTrain = False) -> None:
    if isTrain:
      text_file = keras.utils.get_file(
    fname="spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
    )  
      self.text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"
    self.t_spa_en_mat = None

  
  def train(self):
    print('Henaaa')
    en_sentences, spa_sentences  = self.preprocess()
    en_tokens, en_dictionary = self.tokenize_sentences(en_sentences)
    spa_tokens, spa_dictionary = self.tokenize_sentences(spa_sentences)
    t_en_spa_mat = self.expectation_maximization(spa_dictionary,en_dictionary,spa_tokens,en_tokens)
    
    # Saving the output
    np.save('t_en_spa_mat', t_en_spa_mat)
    
    #Writng language map
    out_file = open("spanish_index_map.json", "w")
    json.dump(spa_dictionary, out_file, indent = 6)
    out_file.close()
    
    out_file = open("english_index_map.json", "w")
    json.dump(en_dictionary, out_file, indent = 6)
    out_file.close()

  
    

  def is_converged(self, new,old,num_of_iterations):
    # print('Checking convergence')
    if num_of_iterations > self.MAX_NUM_OF_ITERATIONS :
        return True

    for i in range(len(new)):
        for j in range(len(new[0])):
            if math.fabs(new[i][j]- old[i][j]) > self.EPSILON:
                return False
    
    print('Coverged')
    return True

  def preprocess(self):
    with open(self.text_file, encoding="utf8") as f:
      lines = f.read().split("\n")[:-1]
    text_pairs = []
    en_sentences = []
    spa_sentences = []
    for line in lines:
        eng, spa = line.split("\t")
        text_pairs.append((spa, eng))
        if len(eng.split()) < 11 and len(spa.split()) < 11:
            en_sentences.append(eng)
            spa_sentences.append(spa)
    
    return en_sentences, spa_sentences

  def tokenize_sentences(self, sentence_list):
    tokenized_sentences = []
    translate_table = dict((ord(char), None) for char in string.punctuation)
    word_dictionary = {}
    lang_order = 0
    for sentence in sentence_list:
        sentence = sentence.translate(translate_table)
        tokens = word_tokenize(sentence.lower())
        produced_sentence = ""
        for token in tokens:
            if token not in word_dictionary:
                word_dictionary[token] = lang_order
                lang_order += 1
            produced_sentence = produced_sentence + str(token) + " "
        produced_sentence = produced_sentence[:(
            len(produced_sentence) - 1)]  # remove last empty
        tokenized_sentences.append(produced_sentence)
    return tokenized_sentences, word_dictionary


  def expectation_maximization(self, spa_word_dict, en_word_dict, spanish_sentences, english_sentences):

    # Initialize the translation probability matrix
    t_spa_en_mat = np.full((len(en_word_dict), len(
        spa_word_dict)), 1/len(spa_word_dict), dtype=float)
    # print(np.where(t_spa_en_mat == 0))
    t_spa_en_mat_prev = np.full(
        (len(en_word_dict), len(spa_word_dict)), 1, dtype=float)

    num_of_iter = 0
    while not self.is_converged(t_spa_en_mat, t_spa_en_mat_prev, num_of_iter):
        print('Iteration : ', num_of_iter)
        num_of_iter += 1

        # Initialization of variables
        t_spa_en_mat_prev = t_spa_en_mat.copy()
        count_en_spa = np.full(
            (len(en_word_dict), len(spa_word_dict)), 0, dtype=float)
        total_spa = np.full(len(spa_word_dict), 0, dtype=float)
        e_total = np.full((len(en_word_dict)), 0, dtype=float)
        for idx_en, en_sent in enumerate(english_sentences):
            for en_word in en_sent.split():
                en_index = en_word_dict[en_word]
                spa_sen_words = spanish_sentences[idx_en].split(" ")
                for spa_word in spa_sen_words:
                    spa_index = spa_word_dict[spa_word]
                    e_total[en_index] += t_spa_en_mat[en_index][spa_index]
            
            
            # collect counts
            for en_word in en_sent.split():
                en_index = en_word_dict[en_word]
                spa_sen_words = spanish_sentences[idx_en].split(" ")
                for spa_word in spa_sen_words:
                    count_en_spa[en_index][spa_index] += t_spa_en_mat[en_index][spa_index] / e_total[en_index]
                    
                    total_spa[spa_index] += t_spa_en_mat[en_index][spa_index] / e_total[en_index]

        print('Finished first loop')

        # estimate probabilities
        for spa_word in spa_word_dict.keys():
            for eng_word in en_word_dict.keys():
                spa_index = spa_word_dict[spa_word]
                eng_index = en_word_dict[eng_word]
                if count_en_spa[eng_index][spa_index] != 0:
                    t_spa_en_mat[eng_index][spa_index] = count_en_spa[en_index][spa_index] / total_spa[spa_index]

        print("finish ")
    # end while

    # print(t_spa_en_mat)

    return t_spa_en_mat

  def load(self):

    # Load the language maps
    with open("spanish_index_map.json", "r") as f:
      self.spa_dictionary = json.load(f)
    with open("english_index_map.json", "r") as f:
      self.en_dictionary = json.load(f)

    print('Loaded Language Maps')

    # loading npy array
    self.t_spa_en_mat = np.load("t_en_spa_mat.npy")
    print('Loaded Spanish to English Matrix')
  
  def getEnglishDict(self):
    return self.en_dictionary

  def getSpanishDict(self):
    return self.spa_dictionary

  def getTranslationTable(self):
    return self.t_spa_en_mat

  def translate(self, en_word, spa_word):
    idx_spa = self.spa_dictionary[spa_word]
    idx_en = self.en_dictionary[en_word]
    return self.t_spa_en_mat[idx_en][idx_spa]

  def test(self):
    print((self.t_spa_en_mat[self.t_spa_en_mat == np.min(self.t_spa_en_mat)]).shape, self.t_spa_en_mat.shape)

IBM().getTranslationTable()