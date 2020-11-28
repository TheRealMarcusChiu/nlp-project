import string

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np

file = open("../../data/semeval_train.txt")
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
df = pd.DataFrame(data)

n_most_common_words = 20000  # vocabulary size
max_len = 100

# Initialization
tokenizer = Tokenizer(num_words=n_most_common_words, filters=string.punctuation, lower=True)
# Fit and transformation
tokenizer.fit_on_texts(df['sentences'].values)
sequences = tokenizer.texts_to_sequences(df['sentences'].values)
print('%s unique tokens.' % len(tokenizer.word_index))
print('unique labels: ', df['labels'].unique())
# Padding
X = pad_sequences(sequences, maxlen=max_len)
print(X[0])
exit(0)

# textFile = open('out-sentences-sequences.txt', 'w')
# for item in sequences:
#     textFile.write(str(item) + "\n")
# textFile.close()

labels = to_categorical(df['labels'], num_classes=len(df.labels.unique()))

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.20, random_state=42)

embedding_size = 300
model = Sequential()
model.add(Embedding(n_most_common_words, embedding_size, input_length=X.shape[1]))
model.add(Bidirectional(LSTM(128, dropout=0.7, recurrent_dropout=0.7)))
model.add(Dense(labels.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1)]
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=callbacks)
model.save('trained-model')
model = keras.models.load_model("trained-model")

prediction_probas = model.predict(X_test)
y_pred = [np.argmax(pred) for pred in prediction_probas]

y_test = [np.argmax(y_t) for y_t in y_test]

print(y_test)
print(y_pred)

cr = classification_report(y_test, y_pred, digits=3)
print(cr)
file = open('out-classification-report.txt', 'w')
file.write(cr)
file.close()

cm = multilabel_confusion_matrix(y_test, y_pred)  # from sklearn.metrics
print(cm)
file = open('out-confusion-matrix.txt', 'w')
file.write(cm)
file.close()
