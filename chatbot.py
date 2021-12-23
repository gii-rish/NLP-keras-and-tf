import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer =  WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentance_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for word in sentance_words:
        for index, w in enumerate(words):
            if w == word:
                bag[index] = 1
                
    return np.array(bag)


def predict_class(sentence):
    bagOfWords = bag_of_words(sentence)    
    res = model.predict(np.array([bagOfWords]))[0]    
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    print(results)
    results.sort(key = lambda x: x[1], reverse=True)
    
    return_list = []
    for result in results:
        return_list.append({'intent': classes[result[0]], 'probability': str(result[1])})
    return return_list


def get_response(intents_list, intents_json):    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Go! Bot is Running")

while True:
    message = input()    
    ints = predict_class(message)
    print(ints)
    res = get_response(ints, intents)
    print(res)
    print()