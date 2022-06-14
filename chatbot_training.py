from random import shuffle
import numpy as np 
import pandas as pd
import nltk
#nltk.download('punkt')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
import pickle
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# read the json fie using pandas library
data=pd.read_json('intents.json')

words=[]
corpus=[]
classes=[]
for intent in data['intents']:
   for pattern in  intent['patterns']:
      # tokenize words
      word= nltk.word_tokenize(pattern)
      words.extend(word)

      # add each document to corpus
      corpus.append((word,intent['tag']))
      
      # add each tag as a class
      if intent['tag'] not in classes:
          classes.append(intent['tag'])


# preprocess words
stop_word=['?','!','#','@','%','(',')',',','.','*','&']
lemmatizer=WordNetLemmatizer()
# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in stop_word and len(word)>1]
words = sorted(list(set(words)))
print(len(words))

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

##########################################################  training  ##########################################################
training=[]
# create an empty list for output
output=[0]*(len(classes))

for doc in corpus:
    #preprocess the words in each document
    doc_words=[lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    bag=[]
    for word in words:
       if word in doc_words:
           bag.append(1)
       else:
           bag.append(0)
    # create the output list for each document 
    output_doc=list(output)
    output_doc[classes.index(doc[1])]=1
    training.append((bag,output_doc))
    
# shuffle data and split x and y
training=np.array(training)
np.random.shuffle(training)
x_train=list(training[:,0])
y_train=list(training[:,1])
print(x_train)

# Neural Network
model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))
   
#SGD optimizer
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Training and saving the model 
hist = model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)


