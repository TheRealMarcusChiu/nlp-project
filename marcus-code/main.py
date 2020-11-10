import nltk

sentence = 'High levels of expression of ENTITY1 were found together with low levels of ENTITY2'
tokens = nltk.word_tokenize(sentence)

WINDOW_SIZE = 3
# make sure that we don't overflow but using the min and max methods
FIRST_INDEX = max(tokens.index("ENTITY1") - WINDOW_SIZE, 0)
SECOND_INDEX = min(sentence.index("ENTITY2") + WINDOW_SIZE, len(tokens))
trimmed_tokens = tokens[FIRST_INDEX: SECOND_INDEX]

normalized_tokens = []
porter = nltk.PorterStemmer()

for t in trimmed_tokens:
    normalized = t.lower()
    if (normalized in nltk.corpus.stopwords.words('english')
        or normalized.isdigit() or len(normalized) < 2):
        continue
    stemmed = porter.stem(t)
    normalized_tokens.append(stemmed)

# file_ = open("data/train.txt")
# corpus = file_.read()
# file_.close()
# splited_data = corpus.split("\n\n\n")