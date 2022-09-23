from itertools import combinations

import spacy
import wikipedia
import random


class Extractor:
    def __init__(self, page_name: str):
        # nlp_model = spacy.load('en')
        nlp_model = spacy.load('en_core_web_sm')
        page = wikipedia.page(page_name, auto_suggest=False).content
        analyzed_page = nlp_model(page)
        self._analyzed_page = analyzed_page
        # print(analyzed_page.text)

    def extract(self):
        raise NotImplementedError("Please Implement this method")


class POSExtractor(Extractor):
    """
    An extractor based only on the POS tags of the tokens in the document
    """

    def _find_all_PNs(self):
        p_nouns = []

        idx = 0
        while idx < len(self._analyzed_page):
            word = self._analyzed_page[idx]
            if word.pos_ != 'PROPN':
                idx += 1
                continue
            noun = {'words': [word], 'start': idx, 'end': idx}

            while True:
                idx += 1
                if idx < len(self._analyzed_page):
                    word = self._analyzed_page[idx]
                    if word.pos_ == 'PROPN':
                        noun['words'].append(self._analyzed_page[idx])
                        noun['end'] = idx
                    else:
                        idx += 1
                        break
                else:
                    break

            p_nouns.append(noun)

        return p_nouns

    def extract(self):
        p_nouns = self._find_all_PNs()
        triplets = []

        for i in range(len(p_nouns) - 1):
            has_verb, no_punct = False, True
            relation = []
            noun_1, noun_2 = p_nouns[i], p_nouns[i + 1]
            for j in range(noun_1['end'] + 1, noun_2['start']):
                word = self._analyzed_page[j]
                if word.pos_ == 'PUNCT':
                    no_punct = False
                    break
                has_verb = True if word.pos_ == 'VERB' else False
                if word.pos_ == 'VERB' or word.pos_ == 'ADP':
                    relation.append(word)
            if has_verb and no_punct:
                triplets.append((noun_1['words'], relation, noun_2['words']))

        return triplets


class DepTreeExtractor(Extractor):
    """
    An extractor based on the dependency trees of the sentences in the document
    """

    def _find_all_PN_heads_sets(self):
        PN_heads = [word for word in self._analyzed_page
                    if word.pos_ == 'PROPN' and word.dep_ != 'compound']
        return {head: [child for child in head.children if child.dep_ == 'compound'] + [head]
                for head in PN_heads}

    def print_heads(self):
        print(self._find_all_PN_heads_sets())

    def extract(self):
        triplets = []
        PN_heads_sets = self._find_all_PN_heads_sets()

        heads = list(PN_heads_sets.keys())
        for h1, h2 in combinations(heads, r=2):
            subj = PN_heads_sets[h1]
            obj = PN_heads_sets[h2]
            # 1st condition:
            if h1.head == h2.head and h1.dep_ == 'nsubj' and h2.dep_ == 'dobj':
                relation = [h1.head]
                triplet = (subj, relation, obj)
                triplets.append(triplet)
            # 2nd condition:
            elif h1.head == h2.head.head and h1.dep_ == 'nsubj' and h2.head.dep_ == 'prep' \
                    and h2.dep_ == 'pobj':
                relation = [h1.head, h2.head]
                triplet = (subj, relation, obj)
                triplets.append(triplet)

        return triplets


def evaluate_on_page(page_name):
    print(f'EVALUATE ON "{page_name}" PAGE')
    pos_res = POSExtractor(page_name).extract()
    print(f'The total number of triplets outputted by POSExtractor is: {len(pos_res)}')
    tree_res = DepTreeExtractor(page_name).extract()
    print(f'The total number of triplets outputted by DepTreeExtractor is: {len(tree_res)}\n')
    return pos_res, tree_res


def get_sents(triplet):
    sentence = ""
    for i in range(3):
        for word in triplet[i]:
            sentence += word.text + " "
    return sentence[:-1]


if __name__ == '__main__':
    # for page_name in ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]:
    for page_name in ["Donald Trump"]:
        pos_res, tree_res = evaluate_on_page(page_name)
        # print(f'pos = {pos_res[:5]}\ntree= {tree_res[:5]}')
        # for trp1, trp2 in zip(pos_res[:5], tree_res[:5]):
        #     print(f'pos: {trp1} ==> {get_sents(trp1)}\ntree: {trp2} ==> {get_sents(trp2)}')

        # print("********* POS triplets from " + page_name + " wikipedia page *********")
        # for triplet in pos_res:
        # # for triplet in random.sample(pos_res, 5):
        #     print(f'pos: {triplet} ==> {get_sents(triplet)}')
        #     # print(get_sents(triplet))
        # print("********* tree triplets from " + page_name + " wikipedia page *********")
        # for triplet in tree_res:
        # # for triplet in random.sample(tree_res, 5):
        #     print(f'tree: {triplet} ==> {get_sents(triplet)}')
        #     # print(get_sents(triplet))
