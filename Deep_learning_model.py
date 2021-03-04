import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding, Dense, Flatten, Bidirectional, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

from import_data import df_blog

#%%
#df_blog = df_blog.sample(5, random_state=42)

#%% One hot
vocab_size = 10000
df_blog['one_hot_rep'] = df_blog.text.apply(lambda x: one_hot(x, vocab_size))

#%%
sent_len = 100
embedded_docs = pad_sequences(df_blog.one_hot_rep, padding='pre', maxlen=sent_len)

#%%
dim = 32

#%%
model = Sequential()
model.add(Embedding(input_dim= vocab_size, output_dim= dim, input_length= sent_len))
model.add(Flatten())
#model.add(Bidirectional(LSTM(dim)))
#model.add(Dense(dim, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

#%%
model.summary()

#%%
labels = pd.get_dummies(df_blog.gender)['female'].to_numpy()
model.fit(embedded_docs, labels, verbose = 1, epochs=10)

#%%

#%%
loss, accuracy = model.evaluate(embedded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))