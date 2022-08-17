import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

import requests
from bs4 import BeautifulSoup
import urllib
import re

with open("intents.json") as file:
    data = json.load(file)

try:
    x
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")



#-----------------------------------------------



def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def search(znp):
    inp=znp.replace("-"," ")
    Inp=inp.split(" ")
    inp=[]
    for i in range(len(Inp)):
        inp.append(Inp[i].capitalize())
    if inp[0]=="Bmw":
        inp[0]="BMW"
    inp1='_'.join(inp)
    print(inp1)

    inp2=znp.replace(" ","/")
    print(inp2)


    # Wiki search----------------------------------------------
    try:
        source = urllib.request.urlopen('https://en.wikipedia.org/wiki/'+inp1).read()
        soup1 =  BeautifulSoup(source, 'lxml')
        text = ""
        a=0
        for para in soup1.find_all('p'):
            text+= para.text
            if a==2:
                break
            a+=1
        text = re.sub(r'\[[0-9]*\]','',text)
        opp=str(text)
    except:
        opp="Sorry, couldn't get the car description."


    # Carinfo search--------------------------------------------
    try:
        url2 = 'https://www.car.info/en-se/'+inp2+'/specs'

        response2 = requests.get(url2)

        soup2 = BeautifulSoup(response2.text, 'html.parser')
        #print(soup2)
        z=soup2.find_all(style="font-weight:bold;")
        #print(z)

        x2=str(z[1]).split(">")
        x2=x2[1].split("<")
        x2="Horsepower : "+str(x2[0])

        x1=str(z[0]).split(">")
        x1=x1[1].split("<")
        x1="Power(kW) : "+str(x1[0])

        x3=str(z[4]).split(">")
        x3=x3[1].split("<")
        x3="Torque(nm) : "+str(x3[0])

        x4=str(z[7]).split(">")
        x4=x4[1].split("<")
        x4="Displacement(litres) : "+str(x4[0])

        x5=str(z[9]).split(">")
        x5=x5[1].split("<")
        x5="Cylinders : "+str(x5[0])

        x6=str(x2)+"\n"+str(x1)+"\n"+str(x3)+"\n"+str(x4)+"\n"+str(x5)
    except:
        x6 = "\nSorry, couldn't get the car performance specs"

    opp=opp+x6+"\n"
    return opp



def chat(user_resp):
    
    while True:
        x=user_resp.split(" ")
        if x[-1].lower()=="dets" or x[-1].lower()=="detail" or x[-1].lower()=="details":
            x.pop()
            x=' '.join(x)
            return search(x)
        else:
            if user_resp.lower() == "quit":
                break

            results = model.predict([bag_of_words(user_resp, words)])
            results_index = numpy.argmax(results)
            print(results)
            print(results_index)
            tag = labels[results_index]

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    break

            return random.choice(responses)


