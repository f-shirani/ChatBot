import numpy as np
import pandas as pd
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import random

class predict:
    def __init__(self):
        self.words= pickle.load(open('words.pkl','rb'))
        self.data=pd.read_json('intents.json')
        self.classes = pickle.load(open('classes.pkl','rb'))
        self.model = load_model('chatbot_model.h5')

    def sentence_prepraed(self,sentence):

        lemmatizer=WordNetLemmatizer()
        # tokenize the pattern - splitting words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stemming every word - reducing to base form
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
     
        # create bag of the words of sentence
        bag=[0]*len(self.words)
        for s in sentence_words:
            for i,w in enumerate(self.words):
                if s==w:
                    bag[i]=1
        return bag

    def predict_class(self,sentence):

        sentence_bag=self.sentence_prepraed(sentence)
        result = self.model.predict(np.array([sentence_bag]))[0]
        ERROR_THRESHOLD = 0.25
        results = ([[i,r] for i,r in enumerate(result)  if r>ERROR_THRESHOLD])

        # sort the result in decending order
        results.sort(key=lambda x: x[1], reverse=True)
        # return the index and probebilty of first result 
        results=results[0]
        return_result={"intent": self.classes[results[0]], "probability": str(results[1])}
        
        return return_result

    def response(self,ints):

        tag = ints['intent']
        list_of_intents = self.data['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
               response = random.choice(i['responses'])
               break
        return response


                
            








