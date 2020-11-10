import re
import numpy as np
import spacy
from spacy import displacy
from collections import Counter


def possible_relations(data):
    relation_type = []
    relation_type_direction = []
    for i in range(len(data) - 1):
        if data[i].split("\n")[1] not in relation_type_direction:
            relation_type_direction.append(data[i].split("\n")[1])
            relation_type.append(data[i].split("\n")[1].split("(")[0])
    return relation_type


def entities_type(data):
    entity_types = {'PERSON': 0, 'NORP': 1, 'FAC': 2, 'ORG': 3, 'GPE': 4, 'LOC': 5, 'PRODUCT': 6, 'EVENT': 7,
                    'WORK_OF_ART': 8, 'LAW': 9, 'LANGUAGE': 10, 'DATE': 11, 'TIME': 12, 'PERCENT': 13, 'MONEY': 14,
                    'QUANTITY': 15, 'ORDINAL': 16, 'CARDINAL': 17, 'UNKNOWN': 18}
    e1S_type = np.zeros((len(data) - 1, len(entity_types)))
    e2S_type = np.zeros((len(data) - 1, len(entity_types)))
    nlp = spacy.load("en_core_web_sm")
    for i in range(len(data) - 1):
        e1 = re.search(r'<e1>[\W\S.,`\':$]+</e1>', data[i]).group()
        e11 = re.sub(r'\s</e1>|<e1>\s', "", e1)
        e2 = re.search(r'<e2>[\W\S.,`\':$]+</e2>', data[i]).group()
        e22 = re.sub(r'\s</e2>|<e2>\s', "", e2)
        doc = nlp(e11 + ',' + e22)
        for ent in doc.ents:
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            if ent.text == e11:
                # e1_start_char = ent.start_char
                # e1_end_char = ent.end_char
                e1_type = ent.label_
                e1S_type[i, entity_types[e1_type]] = 1
            if ent.text == e22:
                # e2_start_char = ent.start_char
                # e2_end_char = ent.end_char
                e2_type = ent.label_
                e2S_type[i, entity_types[e2_type]] = 1
        if sum(e1S_type[i]) == 0:
            e1S_type[i, entity_types['UNKNOWN']] = 1
        if sum(e2S_type[i]) == 0:
            e2S_type[i, entity_types['UNKNOWN']] = 1
    return e1S_type, e2S_type


file_name = "data/train.txt"
file_ = open(file_name)
corpus = file_.read()
file_.close()
splited_data = corpus.split("\n\n\n")

relations_list = possible_relations(splited_data)
# print(relations_list)

e1Type, e2Type = entities_type(splited_data)
file_NoSm = open('out.txt', 'w')
file_NoSm.write(str(e2Type))
file_NoSm.close()
print(e2Type)
