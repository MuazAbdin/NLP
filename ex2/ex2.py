import random
import re
import pandas as pd
from copy import deepcopy
from nltk.corpus import brown
from typing import Callable, List, Tuple
from collections import defaultdict, Counter


def print_table(dictionary: dict, sentence):
    c_dict = deepcopy(dictionary)
    # pprint.pprint(json.loads(json.dumps(c_dict)))
    # x_headers = list(range(len(c_dict)))
    x_headers = [sentence[k - 1][0] for k in c_dict]
    y_headers = list(c_dict[next(iter(c_dict))].keys())
    # print(y_headers)
    data = []
    for x in range(len(x_headers)):
        row = []
        for y in y_headers:
            row.append(c_dict[x + 1][y])
        data.append(row)

    table = pd.DataFrame(data, x_headers, y_headers)
    print(table)


def represent_confusion_mat(conf_mat, tagger_name: str):
    # print(conf_mat)
    real_tags = list(conf_mat.keys())
    # print(real_tags)
    predicted_tags = set()
    for k, v in conf_mat.items():
        predicted_tags |= set(v.keys())
        # predicted_tags = predicted_tags.union(set(v.keys()))
    # print(predicted_tags)
    data = []
    for r_t in real_tags:
        row = []
        for p_t in predicted_tags:
            row.append(conf_mat[r_t][p_t])
        data.append(row)
    table = pd.DataFrame(data, real_tags, predicted_tags)
    pd.DataFrame(table).to_csv(f'{tagger_name}.csv')
    # print(f'=={tagger_name}==')
    # print(table)
    # print('======')


class HMMTagger:
    TS = TaggedSentence = List[Tuple[str, str]]
    Tagger = Callable[[TaggedSentence], dict]

    def __init__(self):
        self.train_set, self.test_set = self.partition_corpus()
        self.words_counter = Counter()
        self.tags_counter = Counter()
        self.MLT_words = self.MLT_baseline()

        self.trans_stats = self.transition_stats(self.train_set)
        self.emit_stats = self.emission_stats(self.train_set)

        self.pseudowords = self.pseudowords_regex()
        self.p_train_set, self.p_test_set = self.update_corpus_words()
        self.p_words_counter = Counter()
        self.p_tags_counter = Counter()
        self.set_ps_counters()
        self.p_trans_stats = self.transition_stats(self.p_train_set)
        self.p_emit_stats = self.emission_stats(self.p_train_set)

    ########################
    ##  3.(a) Setup data  ##
    ########################
    @staticmethod
    def partition_corpus():
        tagged_sents = list(brown.tagged_sents(categories='news'))
        random.shuffle(tagged_sents)
        size = int(len(tagged_sents) * 0.1)
        train_set, test_set = tagged_sents[size:], tagged_sents[:size]
        return train_set, test_set

    ######################################
    ##  3.(b) most likely tag baseline  ##
    ######################################
    @staticmethod
    def simplify_tag(tag: str):
        if '+' in tag:
            tag = tag.split('+')[0]
        if '-' in tag:
            tag = tag.split('-')[0]
        return tag

    def MLT_baseline(self):
        words_tags = defaultdict(Counter)
        for sent in self.train_set:
            self.words_counter.update(w[0] for w in sent)
            self.tags_counter.update(self.simplify_tag(w[1]) for w in sent)
            for w, t in sent:
                words_tags[w][self.simplify_tag(t)] += 1

        words_best_tag = defaultdict(lambda: 'NN')
        # words_best_tag = dict()
        for w in words_tags.keys():
            words_best_tag[w] = max(words_tags[w], key=words_tags[w].get)

        return words_best_tag

    def test_MLT_baseline(self, sentence: TaggedSentence):
        """
        Returns the corresponding MLT for the sentence
        """
        prediction = deepcopy(self.MLT_words)
        return {w: prediction[w] for w, t in sentence}

    def MLT_error_rate(self):
        return self.calculate_error_rate(self.test_set, self.test_MLT_baseline)

    #################################
    ##  3.(c) a bigram HMM tagger  ##
    #################################
    def transition_stats(self, train_set):
        trans_counts = defaultdict(Counter)
        for sent in train_set:
            prev_t = '*'  # special padding symbol
            for w, t in sent:
                t = self.simplify_tag(t)
                trans_counts[prev_t][t] += 1
                prev_t = t
            trans_counts[prev_t]['STOP'] += 1

        return dict(trans_counts)

    def trans_prob(self, t, t_given):
        return self.trans_stats[t_given][t] / self.tags_counter[t_given]

    def emission_stats(self, train_set):
        emit_counts = defaultdict(Counter)
        for sent in train_set:
            for w, t in sent:
                emit_counts[self.simplify_tag(t)][w] += 1
        # print(emit_counts)
        return dict(emit_counts)

    def Viterbi(self, sentence: TaggedSentence, q, e):
        tags = set(self.trans_stats.keys()) - {'*'}
        dp_table = defaultdict(lambda: defaultdict(int))
        back_ptr = defaultdict(lambda: defaultdict(lambda: 'NN'))
        w = sentence[0][0]
        for t in tags:
            dp_table[1][t] = q(t, '*') * e(w, t)
            back_ptr[1][t] = '*'
        for k in range(2, len(sentence) + 1):
            w = sentence[k - 1][0]
            for t in tags:
                dp_table[k][t], back_ptr[k][t] = 0, 'NN'
                for s in tags:
                    prop = dp_table[k - 1][s] * q(t, s) * e(w, t)
                    if prop > dp_table[k][t]:
                        back_ptr[k][t], dp_table[k][t] = s, prop

        # print_table(dp_table, sentence)
        # print_table(back_ptr, sentence)

        last_arg, last_val = 'NN', 0
        for u in tags:
            prop = dp_table[len(sentence)][u] * q('STOP', u)
            if prop > last_val:
                last_arg, last_val = u, prop
        result = [last_arg]
        for k in range(len(sentence) - 1, 0, -1):
            result = [back_ptr[k + 1][result[0]]] + result

        # print(result)
        return result

    def emit_prob(self, w, t):
        # t_count = sum(self.emit_stats[t].values())
        # print(f't_cout = {t_count} <=> {self.tags_counter[t]}')
        # return self.emit_stats[t][w] / t_count
        return self.emit_stats[t][w] / self.tags_counter[t]

    def test_raw_HMM_tagger(self, sentence: TaggedSentence):
        prediction = self.Viterbi(sentence, self.trans_prob, self.emit_prob)
        return {sentence[k][0]: prediction[k] for k in range(len(sentence))}

    def raw_bigram_HMM_tagger_error_rate(self):
        return self.calculate_error_rate(self.test_set, self.test_raw_HMM_tagger)

    #####################################
    ##  3.(d) Using Add-one smoothing  ##
    #####################################
    def smooth_emit_prob(self, w, t):
        # t_count = sum(self.emit_stats[t].values())
        return (1 + self.emit_stats[t][w]) / (self.tags_counter[t] + len(self.words_counter))

    def test_smoothed_HMM_tagger(self, sentence: TaggedSentence):
        prediction = self.Viterbi(sentence, self.trans_prob, self.smooth_emit_prob)
        return {sentence[k][0]: prediction[k] for k in range(len(sentence))}

    def smoothed_bigram_HMM_tagger_error_rate(self):
        return self.calculate_error_rate(self.test_set, self.test_smoothed_HMM_tagger)

    ###############################
    #  3.(e) Using pseudo-words  ##
    ###############################
    @staticmethod
    def pseudowords_regex():
        patterns = {
            'PsWDate': "(0?[1-9]|[12][0-9]|3[01])[-/.](0?[1-9]|1[012])[-/.](19|20)\d{2}",
            'PsWTimeStamp': "([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]",
            'PsWNumber': "\d+([,.]\d+)?",
            'PsWMoney': ".*[\$|\€].*",
            'PsWPercentage': "\d+(\.\d+)?%",
            'PsWAbbreviation': "[A-Z]+",
            'PsWPersonName': "[A-Z][a-z]+",
            'PsWTy': "[a-zA-Z]+ty",
            'PsWIng': "[a-zA-Z]+ing",
            'PsWNess': "[a-zA-Z]+ness",
            'PsWMent': "[a-zA-Z]+ment",
            'PsWEr': "[a-zA-Z]+er",
            'PsWEd': "[a-zA-Z]+ed",
        }
        return {k: re.compile(v) for k, v in patterns.items()}

    def replace_word(self, word: str) -> str:
        for ps_w, patt in self.pseudowords.items():
            if re.match(patt, word):
                return ps_w
        return 'PsWDef'

    def update_set_words(self, learning_set):
        p_learning_set = list()
        for sent in learning_set:
            ps_sent = list()
            for w, t in sent:
                if self.words_counter[w] < 5:
                    ps_sent.append((self.replace_word(w), t))
                else:
                    ps_sent.append((w, t))
            p_learning_set.append(ps_sent)
        return p_learning_set

    def update_corpus_words(self):
        return self.update_set_words(self.train_set), \
               self.update_set_words(self.test_set)

    def set_ps_counters(self):
        for sent in self.p_train_set:
            self.p_words_counter.update(w[0] for w in sent)
            self.p_tags_counter.update(self.simplify_tag(w[1]) for w in sent)

    def p_trans_prob(self, t, t_given):
        return self.p_trans_stats[t_given][t] / self.p_tags_counter[t_given]

    def p_emit_prob(self, w, t):
        return self.p_emit_stats[t][w] / self.p_tags_counter[t]

    def smooth_p_emit_prob(self, w, t):
        return (1 + self.p_emit_stats[t][w]) / (self.p_tags_counter[t] + len(self.p_words_counter))

    def test_ps_HMM_tagger(self, sentence: TaggedSentence):
        prediction = self.Viterbi(sentence, self.p_trans_prob, self.p_emit_prob)
        return {sentence[k][0]: prediction[k] for k in range(len(sentence))}

    def ps_bigram_HMM_tagger_error_rate(self):
        return self.calculate_error_rate(self.p_test_set, self.test_ps_HMM_tagger)

    def test_smoothed_ps_HMM_tagger(self, sentence: TaggedSentence):
        prediction = self.Viterbi(sentence, self.p_trans_prob, self.smooth_p_emit_prob)
        return {sentence[k][0]: prediction[k] for k in range(len(sentence))}

    def smoothed_ps_bigram_HMM_tagger_error_rate(self):
        return self.calculate_error_rate(self.p_test_set, self.test_smoothed_ps_HMM_tagger)

    ###########################
    ##  ERROR RATE FUNCTION  ##
    ###########################
    def calculate_error_rate(self, test_set, tagger: Tagger):
        known, correct_known, unknown, correct_unknown = 0, 0, 0, 0
        confusion_matrix = defaultdict(Counter)
        for sent in test_set:
            predicted_tags = tagger(sent)
            for w, t in sent:
                t = self.simplify_tag(t)
                confusion_matrix[t][predicted_tags[w]] += 1
                if w in self.MLT_words:
                    known += 1
                    correct_known += (t == predicted_tags[w])
                else:
                    unknown += 1
                    correct_unknown += (t == predicted_tags[w])

        represent_confusion_mat(confusion_matrix, tagger.__name__)
        # the error rate (i.e., 1−accuracy) for known words and for unknown words,
        # as well as the total error rate
        known_error_rate = 1 - (correct_known / known)
        unknown_error_rate = 0.0
        if unknown:
            unknown_error_rate = 1 - (correct_unknown / unknown)
        total_error_rate = 1 - (correct_known + correct_unknown) / (known + unknown)

        return known_error_rate, unknown_error_rate, total_error_rate
