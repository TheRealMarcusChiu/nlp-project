import re
import numpy as np
import spacy
from spacy import displacy
from collections import Counter
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import networkx as nx
import statistics 

nlp = spacy.load("en_core_web_sm")
entity_types = {'PERSON': 0, 'NORP': 1, 'FAC': 2, 'ORG': 3, 'GPE': 4, 'LOC': 5, 'PRODUCT': 6, 'EVENT': 7, 'WORK_OF_ART': 8,
                'LAW': 9,'LANGUAGE': 10, 'DATE': 11, 'TIME': 12, 'PERCENT': 13, 'MONEY': 14, 'QUANTITY': 15, 'ORDINAL': 16, 
                'CARDINAL': 17, 'UNKNOWN': 18}
POS = {"PUNCT":0,"ADJ":1,"CCONJ":2,"NUM":3,"DET":4,"PRON":5,"ADP":6,"VERB":7,"NOUN":8,"PROPN":9,"ADV":10,"AUX":11,"SCONJ":12,
        "INTJ":13,"other":14}

prep = {"with": 0, "at": 1, "from": 2, "into": 3, "during": 4, "including": 5, "until": 6, "against": 7, "among": 8, 
        "throughout": 9,"of": 10, "to": 11, "in": 12, "for": 13, "on": 14, "by": 15, "despite": 16, "towards": 17, "upon": 18, 
        "about": 19, "like": 20,"through": 21, "over": 22, "before": 23, "between": 24, "after": 25, "since": 26, "without": 27,
        "under": 28, "around": 29, "near": 30}



# defining the neural network model: 3 layers fully connected
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.Dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.relu2 = nn.ReLU()
        self.Dropout2 = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.Dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.Dropout2(out)
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
        dict_key = data[i].split("\n")[1].split("(")[0]
        if relation_types.get(dict_key) is None:
            index += 1
            relation_types[dict_key] = index
    return relation_types

# data_relation_extract will return the data's relation
def data_relation_extract(data, relation_types):
    y = np.zeros((len(data)-1, 1))
    for j in range(len(data)-1):
        dict_key = data[j].split("\n")[1].split("(")[0]
        y[j] = relation_types[dict_key]
    return y


def entityType(en1,en2,i,e1Type,e2Type):
    doc = nlp(en1+','+en2)
    for ent in doc.ents:
        if ent.text == en1:
            e1_type = ent.label_
            e1Type[i, entity_types[e1_type]] = 1
        if ent.text == en2:
            e2_type = ent.label_
            e2Type[i, entity_types[e2_type]] = 1
    if sum(e1Type[i]) == 0:
        e1Type[i, entity_types['UNKNOWN']] = 1
    if sum(e2Type[i]) == 0:
        e2Type[i, entity_types['UNKNOWN']] = 1

def location(e1search,e2search,i,dis_matrix):
    e1_location = e1search.span()[1]
    e2_location = e2search.span()[1]
    dis_matrix[i] = e2_location - e1_location

def entity_embedding(e,mat_emb):
    word_emb = np.zeros((96, 1))
    word_emb = word_emb[:, 0]
    tokens = nlp(e)
    for token in tokens:
        word_emb = word_emb+token.vector
    if mat_emb is None:
        mat_emb = word_emb
    else:
        mat_emb = np.vstack((mat_emb, word_emb))
    return mat_emb


def entity_head_tag(e11,e22,e1,e2,data,i,matrix_heads): 
    #input1 = re.sub(r'</e2>|<e2>|</e1>|<e1>', "", data)
    #input1 = input1.replace("  "," ")
    indx = 0
    e11 = e11.replace("-","")
    e11 = e11.replace("_","")
    e11 = e11.replace(" ","")
    e22 = e22.replace("-","")
    e22 = e22.replace("_","")
    e22 = e22.replace(" ","")
    input1=data.replace(e2,' '+e22+' ')
    input1=input1.replace(e1,' '+e11+' ')
    input1 = input1.replace("  "," ")
    statement =nlp(input1)
    for token in statement:
        if '<e1>'+token.text+'</e1>' == e1 :
            if POS.get(token.head.pos_) is not None:
                matrix_heads[i , POS[token.head.pos_]]=1
            else:
                matrix_heads[i , POS["other"]]=1
            indx+=1
        elif '<e2>'+token.text+'</e2>' == e2 :
            if POS.get(token.head.pos_) is not None:
                matrix_heads[i , POS[token.head.pos_]]=1
            else:
                matrix_heads[i,POS["other"]]=1
            indx+=1
        if indx==2:
            break


def sdp(e11,e22,e1,e2,data,i,sdp_matrix):
    data1 = data.split("\n")[0]
    e11 = e11.replace("-","")
    e11 = e11.replace("_","")
    e11 = e11.lower()
    e22 = e22.replace("-","")
    e22 = e22.replace("_","")
    e22 = e22.lower()
    
    input1=data1.replace(e2,' '+e22+' ')
    input1=input1.replace(e1,' '+e11+' ')
    input1 = input1.replace("  "," ")
    doc = nlp(input1)
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                        '{0}'.format(child.lower_)))

    graph = nx.Graph(edges)
    try :        
        len_sdp = nx.shortest_path_length(graph, source=e11, target=e22)
    except : 
        len_sdp = -1
    sdp_matrix[i] = len_sdp
    #print(nx.shortest_path(graph, source=entity1, target=entity2))

# will extract the propesitions between e1 and e2 and will extract the tokens between
#  e1,e2 entities and represent them using word embeding
def prep_emb(data,i,POS_beetween_e1e2,sentence_prep,e1_e2_embeding):    
    data_between_entities = re.search(
        r'</e1>[\W\S.,`\':$<>/]+<e2>', data).group()
    data_between_entities2 = re.sub(
        r'</e1>\s|\s<e2>', "", data_between_entities)
    doc = nlp(data_between_entities2)
    word_emb_e1_e2 = np.zeros((96, 1))
    word_emb_e1_e2 = word_emb_e1_e2[:, 0]
    for token in doc:
        if token.tag_ == "IN":
            if prep.get(token.text) is not None:
                sentence_prep[i, prep[token.text]]+=1
        if POS.get(token.pos_) is None:
            POS_beetween_e1e2[i,POS["other"]]+=1
        else:
            POS_beetween_e1e2[i,POS[token.pos_]]+=1

        word_emb_e1_e2 = word_emb_e1_e2+token.vector
    if e1_e2_embeding is None:
        e1_e2_embeding = word_emb_e1_e2
    else:
        e1_e2_embeding = np.vstack((e1_e2_embeding, word_emb_e1_e2))
    return e1_e2_embeding

def emb_after_before(data,before_e1_embeding,aft_e2_embeding):
    data = data.split("\n")[0]
    before_e1 = re.search(r'[\W\S.,`\':$<>/]+<e1>', data).group()
    data_before_split_e1 = before_e1.split(" ")
    try:
        data_before_e1 = data_before_split_e1[-2]
    except:
        data_before_e1 = 'start'
    try:
        data_before_e1 = data_before_e1.split("\t")[1]
    except:
        data_before_e1 = data_before_e1
    doc = nlp(data_before_e1)
    word_emb_bef_e1 = np.zeros((96, 1))
    word_emb_bef_e1 = word_emb_bef_e1[:, 0]
    for token in doc:
        word_emb_bef_e1 = word_emb_bef_e1+token.vector
    if before_e1_embeding is None:
        before_e1_embeding = word_emb_bef_e1
    else:
        before_e1_embeding = np.vstack((before_e1_embeding, word_emb_bef_e1))


    after_e2 = re.search(r'</e2>[\W\S.,`\':$<>/]+', data).group()
    data_after_split_e2 = after_e2.split(" ")
    try:
        data_after_e2 = data_after_split_e2[1]
    except:
        data_after_e2 = 'end'

    doc = nlp(data_after_e2)
    word_emb_aft_e2 = np.zeros((96, 1))
    word_emb_aft_e2 = word_emb_aft_e2[:, 0]
    for token in doc:
        word_emb_aft_e2 = word_emb_aft_e2+token.vector
    if aft_e2_embeding is None:
        aft_e2_embeding = word_emb_aft_e2
    else:
        aft_e2_embeding = np.vstack((aft_e2_embeding, word_emb_aft_e2))

    return before_e1_embeding,aft_e2_embeding



# Reading the train dataset
#file_name = "train.txt"
file_name = "semeval_train.txt"
file_ = open(file_name)
corpus = file_.read()
file_.close()
data = corpus.split("\n\n\n")

relations_list = possible_relations(data)
yTrain = data_relation_extract(data, relations_list)

train_sentence_length = np.zeros((len(data)-1, 1))
train_e1S_type = np.zeros((len(data)-1, len(entity_types)))
train_e2S_type = np.zeros((len(data)-1, len(entity_types)))
train_e1_e2_distance = np.zeros((len(data)-1, 1))
train_e1_embeding = None
train_e2_embeding = None
train_POS_head = np.zeros((len(data)-1, len(POS)))
train_sdp_e1_e2 = np.zeros((len(data)-1, 1))
train_POS_beetween_e1e2 = np.zeros((len(data)-1, len(POS)))
train_sentence_prep = np.zeros((len(data)-1, len(prep)))
train_e1_e2_embeding = None
train_Waft_e2 = None
train_Wbef_e1= None


for i in range(len(data)-1):
    train_sentence_length[i] = len(data[i].split("\n")[0])
    e1_search = re.search(r'<e1>[\W\S.,`\':$]+</e1>', data[i])
    e1 = e1_search.group()
    e11 = re.sub(r'</e1>|<e1>', "", e1)
    e11 = e11.replace(" ", "") 
    e2_search = re.search(r'<e2>[\W\S.,`\':$]+</e2>', data[i])
    e2 = e2_search.group()
    e22 = re.sub(r'</e2>|<e2>', "", e2)
    e22 = e22.replace(" ", "")
    entityType(e11,e22,i,train_e1S_type,train_e2S_type)
    location(e1_search,e2_search,i,train_e1_e2_distance)
    train_e1_embeding = entity_embedding(e11,train_e1_embeding) 
    train_e2_embeding = entity_embedding(e22,train_e2_embeding)
    entity_head_tag(e11,e22,e1,e2,data[i],i,train_POS_head)
    sdp(e11,e22,e1,e2,data[i],i,train_sdp_e1_e2)
    train_e1_e2_embeding = prep_emb(data[i],i,train_POS_beetween_e1e2,train_sentence_prep,train_e1_e2_embeding)
    train_Wbef_e1,train_Waft_e2 = emb_after_before(data[i],train_Wbef_e1,train_Waft_e2)

features = np.concatenate((train_e1S_type, train_e2S_type, train_sentence_prep, train_e1_embeding,
                        train_e2_embeding,train_e1_e2_embeding,train_POS_beetween_e1e2,train_POS_head,train_sdp_e1_e2,train_Wbef_e1,train_Waft_e2), axis=1)
#features = np.concatenate((train_e1_embeding,
#                        train_e2_embeding,train_e1_e2_embeding,train_Wbef_e1,train_Waft_e2,train_POS_beetween_e1e2), axis=1)

mean_features =np.mean(features, axis=0)
std_features = np.std(features, axis=0)
std_features[std_features ==0 ] = 1.0
features = (features-mean_features)/std_features 

#Reading the test dataset
#file_name_test = "test.txt"
file_name_test = "semeval_test.txt"
file_test = open(file_name_test)
corpus_test = file_test.read()
file_test.close()
data_test = corpus_test.split("\n\n\n")

# test set feature extraction
Y_test = data_relation_extract(data_test, relations_list)

test_sentence_length = np.zeros((len(data_test)-1, 1))
test_e1S_type = np.zeros((len(data_test)-1, len(entity_types)))
test_e2S_type = np.zeros((len(data_test)-1, len(entity_types)))
test_e1_e2_distance = np.zeros((len(data_test)-1, 1))
test_e1_embeding = None
test_e2_embeding = None
test_POS_head = np.zeros((len(data_test)-1, len(POS)))
test_sdp_e1_e2 = np.zeros((len(data_test)-1, 1))
test_POS_beetween_e1e2 = np.zeros((len(data_test)-1, len(POS)))
test_sentence_prep = np.zeros((len(data_test)-1, len(prep)))
test_e1_e2_embeding = None
test_Waft_e2 = None
test_Wbef_e1 = None

for i in range(len(data_test)-1):
    test_sentence_length[i] = len(data_test[i])
    e1_search = re.search(r'<e1>[\W\S.,`\':$]+</e1>', data_test[i])
    e1 = e1_search.group()
    e11 = re.sub(r'</e1>|<e1>', "", e1)
    e11 = e11.replace(" ", "") 
    e2_search = re.search(r'<e2>[\W\S.,`\':$]+</e2>', data_test[i])
    e2 = e2_search.group()
    e22 = re.sub(r'</e2>|<e2>', "", e2)
    e22 = e22.replace(" ", "")
    entityType(e11,e22,i,test_e1S_type,test_e2S_type)
    location(e1_search,e2_search,i,test_e1_e2_distance)
    test_e1_embeding = entity_embedding(e11,test_e1_embeding) 
    test_e2_embeding = entity_embedding(e22,test_e2_embeding)
    entity_head_tag(e11,e22,e1,e2,data_test[i],i,test_POS_head)
    sdp(e11,e22,e1,e2,data_test[i],i,test_sdp_e1_e2)
    test_e1_e2_embeding = prep_emb(data_test[i],i,test_POS_beetween_e1e2,test_sentence_prep,test_e1_e2_embeding)
    test_Wbef_e1,test_Waft_e2 = emb_after_before(data_test[i],test_Wbef_e1,test_Waft_e2)


X_test = np.concatenate((test_e1S_type, test_e2S_type, test_sentence_prep, test_e1_embeding,
                        test_e2_embeding,test_e1_e2_embeding,test_POS_beetween_e1e2,test_POS_head,test_sdp_e1_e2,test_Wbef_e1,test_Waft_e2), axis=1)

#X_test = np.concatenate((test_e1_embeding,
#                        test_e2_embeding,test_e1_e2_embeding,test_Wbef_e1,test_Waft_e2,test_POS_beetween_e1e2), axis=1)
X_test = (X_test-mean_features)/std_features 


# Hyper parameters
input_size = features.shape[1]
hidden_size = 500
hidden_size2 = 300
num_classes = len(relations_list)
num_epochs = 80
batch_size = 16
learning_rate = 0.0005

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
    model.train()
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

    model.eval()
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

        print('Accuracy of the network: {} %'.format(
            100 * correct / total))
