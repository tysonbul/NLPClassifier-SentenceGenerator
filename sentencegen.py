# your code goes here
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import pickle
import pprint
import random

from timeit import default_timer as timer

text_file = './data/t8.shakespeare.txt'
pa_save_to_cache = True
pa_load_from_cache = False
pa_unigram_cache_file = './pa_unigramsobj.pkl'
pa_bigram_cache_file = './pa_bigramsobj.pkl'

class NGramCollection():
    def __init__(self):
        pass

    @staticmethod
    def get_top_ngrams(ngram_dict, top=15):
        top_vals = []
        for val in range(top):
            item = max(ngram_dict.items(), key=lambda a: a[1])
            top_vals.append(item)
            ngram_dict.pop(item[0])
        return top_vals

class UnigramCollection():

    def __init__(self):
        self.unigrams = defaultdict(int)
        pass

class BigramCollection():

    def __init__(self):
        self.bigrams = defaultdict(int)
        self.bigrams_probs = defaultdict(float)

    def next_token(self, token):
        """ Adds the next token to the bigram collection """
        try:
            new_bigram = (self.previous_word, token)
        except:
            self.previous_word = token
        else:
            self.bigrams[new_bigram] += 1
            self.previous_word = token

    def determine_bigram_probabilities(self, unigram_dict):
        """ Determine the bigram probabilities give the corresponding unigram counts"""
        for key, value in self.bigrams.items():
            self.bigrams_probs[key] = float(value)/float(unigram_dict[key[0]])

    def find_ngram_with_token(self, token):
        """ Retrieve the bigrams with matching prefix """
        matching = []
        for key, value in self.bigrams_probs.items():
            if key[0] == token:
                matching.append((key, value))
        return matching

    def generate_sentence(self):
        """ Generate a sentence from teh bigram collection """

        def normalize_probabilities(probas):
            sum_probs = sum([prob[1] for prob in probas])
            probas = [(val[0], float(val[1])/sum_probs) for val in probas]
            return probas

        def create_prob_ranges(norm_probs):
            """ Create ranges in which probabilities add up to 1 """
            accum_val = 0
            ranges = []
            for pair in norm_probs:
                accum_val += pair[1]
                ranges.append(accum_val)
            return ranges

        sentence = []

        prev_token = START_SENTENCE
        tokens = self.find_ngram_with_token(prev_token)
        # Generate tokens for sentence until end sentence token found
        while prev_token != END_SENTENCE:
            # Sort tokens based on probabilities
            tokens = sorted(tokens, key=lambda tok: tok[1])
            # Normalize token probabilities
            normalized_tokens = normalize_probabilities(tokens)
            # Create 0 - 1 ranges
            prob_ranges = create_prob_ranges(normalized_tokens)

            r = random.random()
            selected_index = len(prob_ranges) - 1
            # Select token that r fell into
            for index, val in enumerate(prob_ranges):
                if r < val:
                    selected_index = index
                    break

            prev_token = tokens[selected_index][0][1]
            sentence.append(prev_token)
            tokens = self.find_ngram_with_token(prev_token)

        # Return sentence items joined by ' ' and without final end sentence tag
        return ' '.join(sentence[:(len(sentence)-1)])


def count_ngrams():
    """
        Code for Part A: reads in text file and makes
        unigram and bigram counts of the words parsed
    """

    unigrams = UnigramCollection()
    bigrams = BigramCollection()

    # Generate unigram and bigram collections from file
    if pa_load_from_cache:
        # Load collections from saved file
        with open(pa_unigram_cache_file, 'rb') as un_f:
            unigrams = pickle.load(un_f)

        with open(pa_bigram_cache_file, 'rb') as bi_f:
            bigrams = pickle.load(bi_f)
    else:
        # Generate collections from raw text file
        tokenizer = RegexpTokenizer(r'\w+')
        with open(text_file, 'r') as f:
            for line in f:
                if len(line) > 0:
                    # Tokenize the line
                    tokens = tokenizer.tokenize(line.lower())
                    # Add tokens to collections
                    for token in tokens:
                        unigrams.unigrams[token] += 1
                        bigrams.next_token(token)
        # Save collections to external file, for future quick load
        if pa_save_to_cache:
            with open(pa_unigram_cache_file, 'wb') as un_f:
                pickle.dump(unigrams, un_f)
            with open(pa_bigram_cache_file, 'wb') as bi_f:
                pickle.dump(bigrams, bi_f)

    print("")
    print("Top 15 unigrams:")
    pprint.pprint(NGramCollection.get_top_ngrams(unigrams.unigrams))
    print("")
    print("Top 15 bigrams:")
    pprint.pprint(NGramCollection.get_top_ngrams(bigrams.bigrams))

########  Code for Part B  #######
from nltk.tokenize import sent_tokenize

pb_load_from_cache = True
pb_save_to_cache = False
pb_unigram_cache_file = './pb_unigramsobj.pkl'
pb_bigram_cache_file = './pb_bigramsobj.pkl'

START_SENTENCE = "<S>"
END_SENTENCE = "</S>"

num_sentences_to_generate = 5

random.seed()

def generate_sentences():

    def parse_grams():
        """
            Populate ngram collections either from raw
            text file or cached files
        """

        # Make lists fo inner functions can assign
        unigrams = [UnigramCollection()]
        bigrams = [BigramCollection()]

        # Generate unigram and bigram collections from file
        if pb_load_from_cache:
            with open(pb_unigram_cache_file, 'rb') as un_f:
                unigrams[0] = pickle.load(un_f)

            with open(pb_bigram_cache_file, 'rb') as bi_f:
                bigrams[0] = pickle.load(bi_f)
        else:
            with open(text_file, 'r') as f:
                # Read the file in sections, the \n\n is a good boundry
                for _, section in enumerate(f.read().split('\n\n')):
                    if len(section) > 0:
                        # Break up the sections into sentences
                        for sentence in sent_tokenize(section.lower()):
                            tokens = word_tokenize(sentence)
                            # Insert start and end sentence tokens
                            tokens.insert(0, START_SENTENCE)
                            tokens.append(END_SENTENCE)
                            for token in tokens:
                                unigrams[0].unigrams[token] += 1
                                bigrams[0].next_token(token)
            # Save collections to cache if desired
            if pb_save_to_cache:
                with open(pb_unigram_cache_file, 'wb') as un_f:
                    pickle.dump(unigrams[0], un_f)
                with open(pb_bigram_cache_file, 'wb') as bi_f:
                    pickle.dump(bigrams[0], bi_f)

        return (unigrams[0], bigrams[0])

    # Get gram collections
    unigrmas_obj, bigrams_obj = parse_grams()

    # Determine bigram probabilities
    bigrams_obj.determine_bigram_probabilities(unigrmas_obj.unigrams)

    # Generate sentences from bigram collection
    for count in range(num_sentences_to_generate):
        print(bigrams_obj.generate_sentence())


if __name__ == '__main__':
    program_time = timer()
    # For part A
    #count_ngrams()

    # For part B
    generate_sentences()

    print("Exectution time: {0}".format(timer()-program_time))

