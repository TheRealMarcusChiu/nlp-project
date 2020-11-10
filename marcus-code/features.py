import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

# doc = nlp("Autonomous cars shift insurance liability toward manufacturers")
# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
doc = nlp("Ada Lovelace was born in London")

for token in doc:
    print('\n', token.text,
          '\n\ttoken.has_vector: ', token.has_vector,
          '\n\ttoken.vector: ', [token.vector[0], token.vector[len(token.vector)-1]],
          '\n\ttoken.vector_norm: ', token.vector_norm,
          '\n\ttoken.is_out_of_vocab: ', token.is_oov,
          '\n\ttoken.lemma: ', token.lemma_,
          '\n\ttoken.pos: ', token.pos_,
          '\n\ttoken.tag: ', token.tag_,
          '\n\ttoken.shape: ', token.shape_,
          '\n\ttoken.is_alpha: ', token.is_alpha,
          '\n\ttoken.is_stop: ', token.is_stop,
          '\n\ttoken.dep: ', token.dep_,
          '\n\ttoken.head.text: ', token.head.text,
          '\n\ttoken.head.pos: ', token.head.pos_,
          '\n\ttoken.children: ', [child for child in token.children],)

for ent in doc.ents:
    print(ent.text,
          ent.start_char,
          ent.end_char,
          ent.label_,)

# displacy.serve(doc, style="dep")
# displacy.serve(doc, style="ent")
