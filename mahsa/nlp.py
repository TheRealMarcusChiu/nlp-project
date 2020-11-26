import re
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.utils.data
import time

nlp = spacy.load("en_core_web_sm")


# defining the neural network model: 3 layers fully connected
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# Data loader will call CustomDataset to get the data in batches of size 64 or 32 or 16
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train):
        self.X = x_train
        self.Y = y_train

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


# possible_relations will extract all the distinct relations that exist in data set
def possible_relations(data):
    relation_types = {'Other'}  # set data structure
    for i in range(len(data) - 1):
        relation_types.add(data[i].split("\n")[1].strip())
    relation_types.remove('Other')
    relation_types = sorted(relation_types)  # returns list data structure
    relation_types.append('Other')  # add 'Other' at end of list
    relation_types = dict(zip(relation_types, range(len(relation_types))))
    return relation_types


# data_relation_extract will return the data's relation
def data_relation_extract(data, relation_types):
    y = np.zeros((len(data) - 1, 1))
    for j in range(len(data) - 1):
        dict_key = data[j].split("\n")[1].strip()
        y[j] = relation_types[dict_key]
    return y


# et_ed_sl_em will extract different features such as : e1 type, e2 type, the count of letters exists between e1 and e2,
# sentence length, e1 word embedding, e2 word embedding.
def et_ed_sl_em(data):
    entity_types = {'PERSON': 0, 'NORP': 1, 'FAC': 2, 'ORG': 3, 'GPE': 4, 'LOC': 5, 'PRODUCT': 6, 'EVENT': 7,
                    'WORK_OF_ART': 8, 'LAW': 9,
                    'LANGUAGE': 10, 'DATE': 11, 'TIME': 12, 'PERCENT': 13, 'MONEY': 14, 'QUANTITY': 15, 'ORDINAL': 16,
                    'CARDINAL': 17, 'UNKNOWN': 18}
    e1S_type = np.zeros((len(data) - 1, len(entity_types)))
    e2S_type = np.zeros((len(data) - 1, len(entity_types)))
    e1_e2_distance = np.zeros((len(data) - 1, 1))
    sentence_length = np.zeros((len(data) - 1, 1))
    e1_embeding = []
    e2_embeding = []
    for i in range(len(data) - 1):
        sentence_length[i] = len(data[i])
        e1_search = re.search(r'<e1>[\W\S.,`\':$]+</e1>', data[i])
        e1 = e1_search.group()
        e1_location = e1_search.span()[1]
        e11 = re.sub(r'</e1>|<e1>', "", e1).strip()
        e2_search = re.search(r'<e2>[\W\S.,`\':$]+</e2>', data[i])
        e2 = e2_search.group()
        e2_location = e2_search.span()[0]
        e22 = re.sub(r'</e2>|<e2>', "", e2).strip()

        e1_e2_distance[i] = e2_location - e1_location
        doc = nlp(e11 + ',' + e22)
        for ent in doc.ents:
            if ent.text == e11:
                e1_type = ent.label_
                e1S_type[i, entity_types[e1_type]] = 1
            if ent.text == e22:
                e2_type = ent.label_
                e2S_type[i, entity_types[e2_type]] = 1
        if sum(e1S_type[i]) == 0:
            e1S_type[i, entity_types['UNKNOWN']] = 1
        if sum(e2S_type[i]) == 0:
            e2S_type[i, entity_types['UNKNOWN']] = 1

        word_emb = np.zeros((96,))
        tokens = nlp(e11)
        for token in tokens:
            word_emb = word_emb + token.vector
        e1_embeding.append(word_emb)

        word_emb_e2 = np.zeros((96,))
        tokense2 = nlp(e22)
        for tokene2 in tokense2:
            word_emb_e2 = word_emb_e2 + tokene2.vector
        e2_embeding.append(word_emb_e2)

    return e1S_type, e2S_type, e1_e2_distance, sentence_length, np.array(e1_embeding), np.array(e2_embeding)


# will extract the propesitions between e1 and e2 and will extract the tokens between
#  e1,e2 entities and represent them using word embeding
def prep_emb(data):
    prep = {"with": 0, "at": 1, "from": 2, "into": 3, "during": 4, "including": 5, "until": 6, "against": 7, "among": 8,
            "throughout": 9,
            "of": 10, "to": 11, "in": 12, "for": 13, "on": 14, "by": 15, "despite": 16, "towards": 17, "upon": 18,
            "about": 19, "like": 20,
            "through": 21, "over": 22, "before": 23, "between": 24, "after": 25, "since": 26, "without": 27,
            "under": 28, "around": 29, "near": 30}
    POS = {"PUNCT": 0, "ADJ": 1, "CCONJ": 2, "NUM": 3, "DET": 4, "PRON": 5, "ADP": 6, "VERB": 7, "NOUN": 8, "PROPN": 9,
           "ADV": 10, "AUX": 11, "other": 12}
    POS_beetween_e1e2 = np.zeros((len(data) - 1, len(POS)))
    sentence_prep = np.zeros((len(data) - 1, len(prep)))
    e1_e2_embeding = []
    for i in range(len(data) - 1):
        data_between_entities = re.search('</e1>(.*)<e2>', data[i]).group(1).strip()
        doc = nlp(data_between_entities)
        word_emb_e1_e2 = np.zeros((96,))
        for token in doc:
            if token.tag_ == "IN":
                if prep.get(token.text) is not None:
                    sentence_prep[i, prep[token.text]] = 1

            if POS.get(token.pos_) is None:
                POS_beetween_e1e2[i, POS["other"]] = 1
            else:
                POS_beetween_e1e2[i, POS[token.pos_]] = 1

            word_emb_e1_e2 = word_emb_e1_e2 + token.vector
        e1_e2_embeding.append(word_emb_e1_e2)

    return sentence_prep, np.array(e1_e2_embeding), POS_beetween_e1e2


def get_relation_list(file_name):
    file_ = open(file_name)
    corpus = file_.read()
    file_.close()
    splited_data = corpus.split("\n\n\n")
    return possible_relations(splited_data)


def get_features(file_name, relations_list):
    file_ = open(file_name)
    corpus = file_.read()
    file_.close()
    splited_data = corpus.split("\n\n\n")

    ys = data_relation_extract(splited_data, relations_list)
    prepositions, word_emb_e1_e2, POS_beetween_e1e2 = prep_emb(splited_data)
    e1Type, e2Type, e1e2distance, sentence_length, e1_emb, e2_emb = et_ed_sl_em(splited_data)
    return np.concatenate(
        (e1Type, e2Type, e1e2distance, prepositions, e1_emb, e2_emb, word_emb_e1_e2, POS_beetween_e1e2),
        axis=1), ys


def build_and_train_model(X_features_train, Y_train, num_classes):
    input_size = X_features_train.shape[1]

    # Hyper parameters
    hidden_size = 500
    hidden_size2 = 300
    num_epochs = 80
    batch_size = 64
    learning_rate = 0.001

    train_loader = torch.utils.data.DataLoader(  # Reading the training set features in batches
        dataset=CustomDataset(X_features_train, Y_train),
        batch_size=batch_size,
        shuffle=True)

    # Creating the neural network model
    model = NeuralNet(input_size, hidden_size, hidden_size2, num_classes)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (X, Y) in enumerate(train_loader):
            X = X.to(torch.float32)
            Y = Y.to(torch.long)
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, Y[:, 0])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    return model


def compute_accuracy(cm):
    diagonal = np.array([cm[i][i] for i in range(len(cm[0]))])
    return 100 * sum(diagonal) / cm.sum()


def compute_macro_precision_recall_f_scores(cm):
    # https://medium.com/data-science-in-your-pocket/calculating-precision-recall-for-multi-class-classification-9055931ee229
    diagonal = np.array([cm[i][i] for i in range(len(cm[0]))])
    macro_recall = np.mean(np.divide(diagonal, cm.sum(axis=0), out=np.zeros_like(diagonal), where=cm.sum(axis=0) != 0))
    macro_precision = np.mean(
        np.divide(diagonal, cm.sum(axis=1), out=np.zeros_like(diagonal), where=cm.sum(axis=1) != 0))
    # https://tomaxent.com/2018/04/27/Micro-and-Macro-average-of-Precision-Recall-and-F-Score/
    macro_f = 2 * macro_recall * macro_precision / (macro_recall + macro_precision)
    return macro_recall, macro_precision, macro_f


def compute_cm_without_direction(cm):
    rows_even = cm[[i for i in range(0, len(cm), 2)], :]
    rows_odd = cm[[i for i in range(1, len(cm), 2)], :]
    rows_odd = np.vstack((rows_odd, np.zeros(len(rows_odd[0]))))
    cmm = rows_even + rows_odd

    cols_even = cmm[:, [i for i in range(0, len(cmm[0]), 2)]]
    cols_odd = cmm[:, [i for i in range(1, len(cmm[0]), 2)]]
    cols_odd = np.hstack((cols_odd, np.zeros(shape=(len(cols_even[0]), 1))))
    return cols_even + cols_odd


def predict_and_compute_confusion_matrix(model, X_features_test, Y_test, relations_list):
    # compute confusion matrix
    cm = np.zeros((len(relations_list), len(relations_list)))
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(  # Reading the test set features in batches
            dataset=CustomDataset(X_features_test, Y_test),
            batch_size=32,
            shuffle=False)
        for xs, ys in test_loader:
            xs = xs.to(torch.float32)
            ys = ys.to(torch.long)[:, 0]
            _, predicted = torch.max(model(xs).data, 1)
            for i in range(len(ys)):
                y_pred = predicted[i].item()
                y_gold = ys[i]
                cm[y_pred, y_gold] = cm[y_pred, y_gold] + 1
    return cm


def compute_scores(cm):
    # use confusion matrix to compute macro-(precision/recall/f)-scores
    print('Accuracy: {}%'.format(compute_accuracy(cm)))

    macro_recall, macro_precision, macro_f = compute_macro_precision_recall_f_scores(cm)
    print('Macro Recall: {} (with direction)'.format(macro_recall))
    print('Macro Precision: {} (with direction)'.format(macro_precision))
    print('Macro F Score: {} (with direction)'.format(macro_f))

    # compute confusion matrix (no direction distinction)
    cmm = compute_cm_without_direction(cm)
    macro_recall, macro_precision, macro_f = compute_macro_precision_recall_f_scores(cmm)
    print('Macro Recall: {} (without direction)'.format(macro_recall))
    print('Macro Precision: {} (without direction)'.format(macro_precision))
    print('Macro F Score: {} (without direction)'.format(macro_f))


relations_list = get_relation_list("semeval_train.txt")

# train or load model
load_model = True
if load_model:
    model = torch.load('trained-model.pt')
else:
    X_features_train, Y_train = get_features("semeval_train.txt", relations_list)
    model = build_and_train_model(X_features_train, Y_train, num_classes=len(relations_list))
    torch.save(model, 'trained-model.pt')

# test model
start = time.time()
X_features_test, Y_test = get_features("semeval_test.txt", relations_list)
cm = predict_and_compute_confusion_matrix(model, X_features_test, Y_test, relations_list)
end = time.time()
print('prediction time for {} samples : {}'.format(len(Y_test), end - start))
print('prediction time for 1 sample : {}'.format((end - start) / len(Y_test)))
