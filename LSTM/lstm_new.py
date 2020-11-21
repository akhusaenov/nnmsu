import pandas as pd
import numpy as np
import sys
sys.path.append("/")
import encoder

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.models import load_model


positive, negative, dictionary = ('data/positive.csv','data/negative.csv','data/dictionary.csv')
#encoder = encoder.encoder_twitter(dictionary)

encoder = encoder.encoder_twitter()
# if there are no encoded files uncomment the following lines

#encoder.encode(data, url,limit,save)
# data - 2D array Numpy ot list
# url - twitts file
# limit - # limit - upper limit for dataset (more quickly train)
# save - overwrites files after encoding

#pos = encoder.encode(positive, save=True)
#neg = encoder.encode(negative, save=True)

# and comment on the following 2 lines

# read  positive and negative dataset
pos = pd.read_csv('data/positive_set.csv',sep=';',index_col='index')
neg = pd.read_csv('data/negative_set.csv',sep=';',index_col='index')

# можете добавить предложение в датасет
# пример
#twitts = [
#    ['Потрясающий фильм схожу еще раз'],
#    ['Фильм не понравился']
#]

#new_pos = encoder.encode(data=twitts)
#new_pos = sequence.pad_sequences(new_pos,len(pos))
#pos = np.r_[new_pos,pos]
#
#new_neg = encoder.encode(data=twitts)
#new_neg = sequence.pad_sequences(new_pos,len(pos))
#neg = np.r_[new_neg,neg]


pos = sequence.pad_sequences(pos.values, maxlen = 40)
neg = sequence.pad_sequences(neg.values, maxlen = 40)

pos = np.c_[pos,np.ones(len(pos))]
neg = np.c_[neg,np.zeros(len(neg))]
alls = np.r_[pos,neg]

randomize = np.arange(len(alls))
np.random.shuffle(randomize)
alls = alls[randomize]

x_train = alls[:28000]
y_train = x_train[:,40]
x_train = np.delete(x_train,40,axis=1)

x_test = alls[28000:30000]
y_test = x_test[:,40]
x_test = np.delete(x_test,40,axis=1)

max_features = 200000

batch_size = 64


model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))

#test NN
print(model.predict(x_test))

#test = encoder.encode(data=twitts)
#test = sequence.pad_sequences(test.values,maxlen = 40)
#print(model.predict(test))

# example decode
#decode_array = encoder.decode(x_test)
#print(decode_array)

#model.save('data/save_model.h5')