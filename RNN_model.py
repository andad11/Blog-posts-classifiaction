import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

from import_data import df_blog

#%%

df_blog = df_blog.sample(5, random_state=42)

#%% One hot
vocab_size = 10000
df_blog['encoded'] = df_blog.text.apply(lambda x: one_hot(x, vocab_size))