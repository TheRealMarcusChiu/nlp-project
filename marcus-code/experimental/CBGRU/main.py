from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Embedding, Dropout, Convolution1D, CuDNNGRU, Bidirectional, Activation, \
    GlobalMaxPooling1D, Dense, Permute, Lambda, Flatten, Concatenate

words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
words_input_mask = Input(shape=(max_sentence_len,), dtype='int32', name='words_input_mask')

words = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding], trainable=True, embeddings_regularizer=embeddings_reg, name="words_Embedding")(words_input)
words = Dropout(dropout_emb)(words)

output = Convolution1D(filters=256, kernel_size=3, activation=activation_fn, padding='same', strides=1, kernel_regularizer=kernel_reg)(words)
output = Dropout(dropout_model)(output)

output = Bidirectional( CuDNNGRU(units, return_sequences=True, recurrent_regularizer=recurrent_reg), merge_mode='concat') (output)
output_h = Activation('tanh') (output)

output1 = GlobalMaxPooling1D()(output_h)
output2 = MaskMaxPoolingLayer()([output_h, words_input_mask])

output = Dense(1, kernel_regularizer=kernel_reg)(output_h)
output = Permute((2, 1))(output)
output = Activation('softmax', name="attn_softmax")(output)
output = Lambda(lambda x: tf.matmul(x[0], x[1])) ([output, output_h])
output3 = Flatten() (output)

output = Concatenate()([output1, output2, output3])
output = Dropout(dropout_pen)(output)

output = Dense(300, kernel_regularizer=kernel_reg, activation='relu')(output)
output = Dense(n_out, kernel_regularizer=kernel_reg)(output)
output = Activation('softmax')(output)

model = Model(inputs=[words_input, words_input_mask], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])
model.summary(line_length=120)