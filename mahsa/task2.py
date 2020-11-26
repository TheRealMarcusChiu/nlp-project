import re
import spacy
from nltk.corpus import wordnet as wn
from spacy import displacy

nlp = spacy.load("en_core_web_sm")


def Task2(input):
    e1_search = re.search(r'<e1>[\W\S.,`\':$]+</e1>', input)
    e1 = e1_search.group()
    e11 = re.sub(r'</e1>|<e1>', "", e1)
    e11 = e11.replace(" ", "")
    e2_search = re.search(r'<e2>[\W\S.,`\':$]+</e2>', input)
    e2 = e2_search.group()
    e22 = re.sub(r'</e2>|<e2>', "", e2)
    e22 = e22.replace(" ", "")
    input1 = re.sub(r'</e2>|<e2>|</e1>|<e1>', "", input)
    input1 = input1.replace("  ", " ")
    statement = nlp(input1.strip())
    print("\n\n")
    for token in statement:
        print('Token :{0:15}lemma:{1:15}POS:{2:15}Dependency Parsing:{3:15}Head text:{4:15}Head POS:{5:15}'.format(
            token.text, token.lemma_, token.pos_, token.dep_, token.head.text, token.head.pos_, ))
    print("\n")
    entity = nlp(e11 + ',' + e22)
    for ent in entity.ents:
        print('entity:{0:10}tag:{1:10}'.format(ent.text, ent.label_))
    if len(entity.ents) == 0:
        print('entity:{0:10}tag:{1:10}'.format(e22, "unknown"))
        print('entity:{0:10}tag:{1:10}'.format(e11, "unknown"))
    elif len(entity.ents) == 1:
        if ent.text == e11:
            print('entity:{0:10}tag:{1:10}'.format(e22, "unknown"))
        else:
            print('entity:{0:10}tag:{1:10}'.format(e11, "unknown"))
    print('\n')
    for token in statement:
        synsets = wn.synsets(token.text)
        hypernym_set = set()
        hyponyms_set = set()
        meronyms_set = set()
        holonyms_set = set()
        for synset in synsets:
            hypernym_set = hypernym_set.union(set(synset.hypernyms()))
            hyponyms_set.union(set(synset.hyponyms()))
            hyponyms_set = meronyms_set.union(set(synset.part_meronyms()))
            meronyms_set = meronyms_set.union(set(synset.substance_meronyms()))
            holonyms_set = holonyms_set.union(set(synset.part_holonyms()))
            holonyms_set = holonyms_set.union(set(synset.substance_holonyms()))
        print(token.text, ' hypernyms: ', hypernym_set)
        print(token.text, ' hyponyms: ', hyponyms_set)
        print(token.text, ' meronyms: ', meronyms_set)
        print(token.text, ' holonyms: ', holonyms_set)
    displacy.serve(statement, style="dep")


# input = " With <e1> NAZA </e1> FC he won the <e2> Malaysia </e2> Premier League 2007-08 championship . "
print("\n\nPlease enter the test sentence")
input_test = input()
