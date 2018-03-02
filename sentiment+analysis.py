
# coding: utf-8

# In[36]:


import keras


# In[37]:


import numpy


# In[38]:


from keras.datasets import imdb


# In[39]:


from matplotlib import pyplot


# In[40]:


(x_train,y_train), (x_test,y_test)= imdb.load_data()


# In[41]:


x=numpy.concatenate((x_train,x_test),axis=0)
y=numpy.concatenate((y_train,y_test),axis=0)


# In[42]:


#summarize the size
print("training_data:")
print(x.shape)
print(y.shape)


# In[43]:


#summarize number of words
print(len(numpy.unique(numpy.hstack(x))))


# In[46]:


result=[len(x) for x in x]
print("mean %.2f words ( %f)" % (numpy.mean(result),numpy.std(result)))
#plot review length
pyplot.boxplot(result)
pyplot.show()


# In[60]:


imdb.load_data(nb_words=5000)


# In[62]:


from keras.layers import Embedding

from keras.preprocessing import sequence

from keras.preprocessing.sequence import pad_sequences
pad_sequences




# In[64]:




x_train =pad_sequences(x_train, maxlen=500)
x_test = pad_sequences(x_test, maxlen=500)


# In[65]:


Embedding (5000,32, input_length=500)


# In[66]:


# MLP for the IMDB problem
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[67]:




# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


# In[68]:


#We will bound reviews at 500 words, truncating longer reviews and zero-padding shorter reviews.


max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


# In[69]:





# create the model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[ ]:




# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:




