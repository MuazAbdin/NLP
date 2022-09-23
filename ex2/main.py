from ex2 import *
import pandas as pd
from collections import defaultdict, Counter
import json
import pprint

if __name__ == '__main__':
    # sent = [('the', 'A'), ('the', 'A'), ('the', 'B'), ('boy', 'C'),
    #         ('boy', 'D'), ('the', 'A'), ('boy', 'C'), ('eat', 'V')]
    # word_tags = defaultdict(lambda: defaultdict(int))
    # for w, t in sent:
    #     word_tags[w][t] += 1
    #
    # pprint.pprint(json.loads(json.dumps(word_tags)))
    #
    # words_best_tag = dict()
    # for w in word_tags.keys():
    #     words_best_tag[w] = max(word_tags[w], key=word_tags[w].get)
    #
    # print(words_best_tag)

    # train_set, test_set = partition_corpus()
    # print(len(train_set), len(test_set))
    # print(train_set[0])
    # print(test_set[0])
    # train_set = [[('the', 'B'), ('boy', 'C'), ('eat', 'V'), ('the', 'B')],
    #              [('boy', 'D'), ('an', 'X'), ('an', 'X'), ('the', 'A')],
    #              [('eat', 'V'), ('the', 'B'), ('boy', 'D'), ('an', 'Y')]]
    # best = MLT_baseline(train_set)
    # print(best)
    # error_rate = baseline_error_rate(test_set, best)
    # print(error_rate)
    # emit_counts = Counter()
    # for sent in train_set:
    #     emit_counts.update(sent)
    # print(emit_counts)
    # print(emit_counts.keys())
    tagger = HMMTagger()
    # w_headers = list(tagger.words_counter.keys())
    # t_headers = list(tagger.tags_counter.keys())
    # data = []
    # s_data = []
    # # print(w_headers)
    # # print(t_headers)
    # emit_dist = deepcopy(tagger.emit_dist)
    # sm_emit_dist = deepcopy(tagger.smoothed_emit_dist)
    # # w_headers = list(emit_dist.keys())
    # # print(w_headers)
    # # t_headers = list(emit_dist[w_headers[0]].keys())
    # # print(t_headers)
    # # for w, ts in emit_dist.items():
    # for w in w_headers:
    #     row, s_row = [], []
    #     # print(f'{w} : ', end='')
    #     for t in t_headers:
    #         row.append(emit_dist[w][t])
    #         s_row.append(sm_emit_dist[w][t])
    #         # print(f'({t},{tagger.emit_dist[w][t]})', end='')
    #     # print()
    #     data.append(row)
    #     s_data.append(s_row)
    #
    # em = pd.DataFrame(data, w_headers, t_headers)
    # # pd.DataFrame(em).to_csv("em.csv")
    # s_em = pd.DataFrame(s_data, w_headers, t_headers)
    # # pd.DataFrame(s_em).to_csv("s_em.csv")
    # print(em)
    # print(s_em)
    #
    # trans_dist = deepcopy(tagger.trans_dist)
    # t2_headers = ['*'] + t_headers + ['STOP']
    # t_data = []
    # for t1 in t2_headers:
    #     t_row = []
    #     for t2 in t2_headers:
    #         t_row.append(trans_dist[t1][t2])
    #     t_data.append(t_row)
    #
    # tr = pd.DataFrame(t_data, t2_headers, t2_headers)
    # print(tr)

    # print(len(tagger.MLT_words))
    # print(len(tagger.words_counter))
    # print(len(tagger.tags_counter))
    # pprint.pprint(json.loads(json.dumps(tagger.trans_dist)))
    # pprint.pprint(json.loads(json.dumps(tagger.emit_dist)))
    # pprint.pprint(json.loads(json.dumps(tagger.smoothed_emit_dist)))

    print(tagger.MLT_error_rate())
    print(tagger.raw_bigram_HMM_tagger_error_rate())
    print(tagger.smoothed_bigram_HMM_tagger_error_rate())
    print(tagger.ps_bigram_HMM_tagger_error_rate())
    print(tagger.smoothed_ps_bigram_HMM_tagger_error_rate())

    #   known_error_rate   unknown_error_rate   total_error_rate
    # (0.06496396778295888, 0.7605140186915887, 0.12281383598911777)
    # (0.6181644764730818, 0.7605140186915887, 0.6300038865137971)
    # (0.13586265366680794, 0.6787383177570093, 0.1810143801010493)
    # (0.12441198316414959, 0.530713640469738, 0.21181500194325686)
    # (0.08405545927209701, 0.507226738934056, 0.1750874465604353)
