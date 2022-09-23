
import nltk

nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank

from collections import defaultdict, Counter, namedtuple
from itertools import product, chain
from Chu_Liu_Edmonds_algorithm import min_spanning_arborescence_nx
from tqdm import trange, tqdm
from random import shuffle

Arc = namedtuple('Arc', 'head tail weight')


def partition_corpus(parsed_sentences):
    # the test set is formed by the last 10% of the sentences.
    size = -round(len(parsed_sentences) / 10)
    return parsed_sentences[:size], parsed_sentences[size:]


def raw_feature_vector(parsed_sentences):
    vocabulary, POS_tags = set(), set()

    for sent in parsed_sentences:
        sent_words, sent_tags = set(), set()
        for node in sent.nodes.values():
            sent_words.add(node['word'])
            sent_tags.add(node['tag'])

        vocabulary |= set(product(sent_words, repeat=2))
        POS_tags |= set(product(sent_tags, repeat=2))

    return defaultdict(int, {v: k for k, v in enumerate(vocabulary | POS_tags)})


class MSTParser:
    def __init__(self, parsed_sentences):
        self._train_set, self._test_set = partition_corpus(parsed_sentences)
        self._features = raw_feature_vector(parsed_sentences)
        self._weights = Counter()

    def _get_features_idx(self, node1, node2):
        return [self._features[node1['word'], node2['word']],
                self._features[node1['tag'], node2['tag']]]

    def _max_score_tree(self, sentence):
        tree_arcs = []
        for x, y in product(sentence.nodes, repeat=2):
            i, j = self._get_features_idx(sentence.nodes[x], sentence.nodes[y])
            weight = -(self._weights[i] + self._weights[j])
            tree_arcs.append(Arc(x, y, weight))
        return min_spanning_arborescence_nx(tree_arcs, None)

    def _sentence_features_sum(self, sentence):
        stack = [0]
        score = Counter()
        while stack:
            parent = sentence.nodes[stack.pop()]
            dependents = list(chain.from_iterable(parent['deps'].values()))
            stack.extend(dependents)
            for child in dependents:
                score.update(self._get_features_idx(parent, sentence.nodes[child]))
        return score

    def _tree_features_sum(self, tree, sentence):
        score = Counter()
        for arc in tree.values():
            score.update(self._get_features_idx(sentence.nodes[arc.head],
                                                sentence.nodes[arc.tail]))
        return score

    def train(self, num_iter=2):
        print("... TRAINING THE MODEL ...")
        avg_weights = Counter()
        shuffle(self._train_set)

        for iteration in trange(num_iter, desc='Iterations', leave=True):
            for sent in tqdm(self._train_set, desc='Sentences', leave=False):
                max_tree = self._max_score_tree(sent)
                features_sum = self._sentence_features_sum(sent)
                features_sum.subtract(self._tree_features_sum(max_tree, sent))
                self._weights.update(features_sum)
                avg_weights.update(self._weights)

        for key in avg_weights.keys():
            avg_weights[key] /= num_iter * len(self._train_set)

        self._weights = avg_weights

    def evaluate(self):
        print("... EVALUATING THE MODEL ...")

        total_accuracy = 0
        for sent in tqdm(self._test_set, desc='Sentences', leave=False):
            predicted_edges = {(edge.head, edge.tail)
                               for edge in self._max_score_tree(sent).values()}
            gold_standard_edges = {(node['head'], node['address'])
                                   for node in sent.nodes.values()}
            shared_edges = predicted_edges & gold_standard_edges
            total_accuracy += len(shared_edges) / (len(sent.nodes) - 1)

        return round(total_accuracy / len(self._test_set), ndigits=3)


if __name__ == '__main__':
    data = dependency_treebank.parsed_sents()
    mst = MSTParser(data)
    mst.train()
    print("==================")
    accuracy = mst.evaluate()
    print(f'Accuracy = {accuracy * 100} %')
