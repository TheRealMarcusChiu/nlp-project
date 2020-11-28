import pandas as pd


def load_data(file_path):
    file = open(file_path)
    corpus = file.read()
    file.close()
    split_data = corpus.split("\n\n\n")

    sentences = []
    labels = []

    for sd in split_data:
        tmp = sd.split('\n')
        sentences.append(tmp[0].partition('"')[2].rsplit('"', 1)[0].strip())
        labels.append(tmp[1])
    labels = pd.factorize(labels)[0]

    data = {'sentences': sentences,
            'labels': labels}
    return pd.DataFrame(data)


df = load_data("../data/semeval_train.txt")

print(df['sentences'][0])
print(df['labels'][0])
print('unique labels: ', df['labels'].unique())

