import re
import numpy as np
import spacy
from spacy import displacy
from collections import Counter
import torch
# import torchvision
import torch.nn as nn
import torch.utils.data
# import torchvision.transforms as transforms

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
    relation_types = {}
    index = -1
    for i in range(len(data)-1):
        dict_key = data[i].split("\n")[1].strip()
        if relation_types.get(dict_key) is None:
            index += 1
            relation_types[dict_key] = index
    return relation_types

# data_relation_extract will return the data's relation
def data_relation_extract(data, relation_types):
    y = np.zeros((len(data)-1, 1))
    for j in range(len(data)-1):
        dict_key = data[j].split("\n")[1].strip()
        y[j] = relation_types[dict_key]
    return y

# et_ed_sl_em will extract different features such as : e1 type, e2 type, the count of letters exists between e1 and e2,
# sentence length, e1 word embedding, e2 word embedding.
def et_ed_sl_em(data):
    entity_types = {'PERSON': 0, 'NORP': 1, 'FAC': 2, 'ORG': 3, 'GPE': 4, 'LOC': 5, 'PRODUCT': 6, 'EVENT': 7, 'WORK_OF_ART': 8, 'LAW': 9,
                    'LANGUAGE': 10, 'DATE': 11, 'TIME': 12, 'PERCENT': 13, 'MONEY': 14, 'QUANTITY': 15, 'ORDINAL': 16, 'CARDINAL': 17, 'UNKNOWN': 18}
    e1S_type = np.zeros((len(data)-1, len(entity_types)))
    e2S_type = np.zeros((len(data)-1, len(entity_types)))
    e1_e2_distance = np.zeros((len(data)-1, 1))
    sentence_length = np.zeros((len(data)-1, 1))
    e1_embeding = []
    e2_embeding = []
    for i in range(len(data)-1):
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
        doc = nlp(e11+','+e22)
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
            word_emb = word_emb+token.vector
        e1_embeding.append(word_emb)

        word_emb_e2 = np.zeros((96,))
        tokense2 = nlp(e22)
        for tokene2 in tokense2:
            word_emb_e2 = word_emb_e2+tokene2.vector
        e2_embeding.append(word_emb_e2)

    return e1S_type, e2S_type, e1_e2_distance, sentence_length, np.array(e1_embeding), np.array(e2_embeding)

# will extract the propesitions between e1 and e2 and will extract the tokens between
#  e1,e2 entities and represent them using word embeding
def prep_emb(data):
    prep = {"with": 0, "at": 1, "from": 2, "into": 3, "during": 4, "including": 5, "until": 6, "against": 7, "among": 8, "throughout": 9,
            "of": 10, "to": 11, "in": 12, "for": 13, "on": 14, "by": 15, "despite": 16, "towards": 17, "upon": 18, "about": 19, "like": 20,
            "through": 21, "over": 22, "before": 23, "between": 24, "after": 25, "since": 26, "without": 27, "under": 28, "around": 29, "near": 30}
    POS = {"PUNCT":0,"ADJ":1,"CCONJ":2,"NUM":3,"DET":4,"PRON":5,"ADP":6,"VERB":7,"NOUN":8,"PROPN":9,"ADV":10,"AUX":11,"other":12}
    POS_beetween_e1e2 = np.zeros((len(data)-1, len(POS)))
    sentence_prep = np.zeros((len(data)-1, len(prep)))
    e1_e2_embeding = []
    for i in range(len(data)-1):
        data_between_entities = re.search('</e1>(.*)<e2>', data[i]).group(1).strip()
        doc = nlp(data_between_entities)
        word_emb_e1_e2 = np.zeros((96,))
        for token in doc:
            if token.tag_ == "IN":
                if prep.get(token.text) is not None:
                    sentence_prep[i, prep[token.text]] = 1

            if POS.get(token.pos_) is None:
                POS_beetween_e1e2[i,POS["other"]]=1
            else:
                POS_beetween_e1e2[i,POS[token.pos_]] = 1

            word_emb_e1_e2 = word_emb_e1_e2+token.vector
        e1_e2_embeding.append(word_emb_e1_e2)

    return sentence_prep , np.array(e1_e2_embeding), POS_beetween_e1e2

# Reading the train dataset
#file_name = "train.txt"
file_name = "semeval_train.txt"
file_ = open(file_name)
corpus = file_.read()
file_.close()
splited_data = corpus.split("\n\n\n")

# train set feature extraction :
train_prepositions, word_emb_e1_e2 , POS_beetween_e1e2 = prep_emb(splited_data)
relations_list = possible_relations(splited_data)
yTrain = data_relation_extract(splited_data, relations_list)
e1Type, e2Type, e1e2distance, train_sentence_length, e1_emb,e2_emb = et_ed_sl_em(splited_data)
features = np.concatenate((e1Type, e2Type, e1e2distance, train_prepositions, e1_emb,e2_emb,word_emb_e1_e2,POS_beetween_e1e2), axis=1)

#Reading the test dataset
#file_name_test = "test.txt"
file_name_test = "semeval_test.txt"
file_test = open(file_name_test)
corpus_test = file_test.read()
file_test.close()
splited_data_test = corpus_test.split("\n\n\n")

# test set feature extraction
Y_test = data_relation_extract(splited_data_test, relations_list)
test_prepositions , test_word_emb_e1_e2,test_POS_beetween_e1e2 = prep_emb(splited_data_test)
e1Type_test, e2Type_test, e1e2distance_test, test_sentence_length, e1_test_emb,e2_test_emb = et_ed_sl_em(splited_data_test)
X_test = np.concatenate((e1Type_test, e2Type_test,e1e2distance_test, test_prepositions, e1_test_emb,e2_test_emb,test_word_emb_e1_e2,test_POS_beetween_e1e2), axis=1)

# Hyper parameters
input_size = features.shape[1]
hidden_size = 500
hidden_size2 = 300
num_classes = len(relations_list)
num_epochs = 80
batch_size = 32
learning_rate = 0.001

# Reading the training set features in baches
custom_dataset = CustomDataset(features, yTrain)
train_loader = torch.utils.data.DataLoader(
    dataset=custom_dataset, batch_size=64, shuffle=True)

# Reading the test set features in baches
custom_dataset_test = CustomDataset(X_test, Y_test)
test_loader = torch.utils.data.DataLoader(
    dataset=custom_dataset_test, batch_size=batch_size, shuffle=False)

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
        loss = criterion(outputs,  Y[:, 0])

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for xTest, yTest in test_loader:
        xTest = xTest.to(torch.float32)
        yTest = yTest.to(torch.long)
        outputs_test = model(xTest)
        _, predicted = torch.max(outputs_test.data, 1)
        total += yTest.size(0)
        correct += (predicted == yTest[:, 0]).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(
        100 * correct / total))
