import wikipedia, spacy


def nouns_extractor(analyzed_page):
    i = 0
    nouns = []
    while i < len(analyzed_page):
        word = analyzed_page[i]
        if word.pos_ == 'PROPN':
            noun = dict()
            noun['words'] = []
            noun['words'].append(word)
            noun['start'] = i
            noun['end'] = i
            while True:
                i += 1
                if i < len(analyzed_page):
                    word = analyzed_page[i]
                    if word.pos_ == 'PROPN':
                        noun['words'].append(word)
                        noun['end'] = i
                    else:
                        i += 1
                        break
                else:
                    break
            nouns.append(noun)
        else:
            i += 1
    return nouns


def extractor(page):
    nlp = spacy.load("en")
    analyzed_page = nlp(page)

    nouns = nouns_extractor(analyzed_page)
    triplets = []
    for i in range(len(nouns) - 1):
        legal = True
        verb = False
        relation = []
        noun_1 = nouns[i]
        noun_2 = nouns[i + 1]
        for j in range(noun_1['end'] + 1, noun_2['start']):
            word = analyzed_page[j]
            if word.pos_ == 'PUNCT':
                legal = False
                break
            elif word.pos_ == 'VERB':
                verb = True
            if word.pos_ == 'VERB' or word.pos_ == 'ADP':
                relation.append(word)
        if legal and verb:
            triplets.append((noun_1['words'], relation, noun_2['words']))
    return triplets

def trees_extractor(page):
    nlp = spacy.load("en")
    sentences = list(nlp(page).sents)
    heads = []
    for sentence in sentences:
        for word in sentence:
            if word.pos_ == 'PROPN' and word.dep_ != 'compound':
                heads.append(word)

