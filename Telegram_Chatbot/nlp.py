import json
import numpy as np
import numpy
import nltk
nltk.download('punkt')
import pickle
import tensorflow as tf
import random
from snowballstemmer import TurkishStemmer
stemmer = TurkishStemmer()
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Dropout
from tensorflow.keras.optimizers import SGD

data_file = open('intents.json' ,encoding='iso8859-9').read()
intents = json.loads(data_file)
words=[]
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [stemmer.stemWord(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [stemmer.stemWord(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training , dtype=object,)
output = np.array(output_empty , dtype=object,)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")



model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
model.save('model.keras', hist)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    # print(s_words)
    s_words = [stemmer.stemWord(word.lower()) for word in s_words]
    # print(s_words)

    for se in s_words:
        # print(se)
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


graph = tf.Graph()
sess = tf.compat.v1.Session(graph=graph)

def prediction(x):
    with graph.as_default():
        with sess.as_default():
            results = model.predict(np.asanyarray([bag_of_words(x, words)]))[0]
            results_index = np.argmax(results)
            tag = classes[results_index]

            if results[results_index] > 0.70:
                for tg in intents["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                return random.choice(responses)
            else:
                return "Tam olarak anlayamadım"



ERROR_THRESHOLD = 0.30
def chat(x):
    while True:

        # results = model.predict([bag_of_words(inp, words)])
        results = model.predict(np.asanyarray([bag_of_words(x, words)]))[0]
        results_index = numpy.argmax(results)
        tag = classes[results_index]

        if results[results_index] > 0.70:
            for tg in intents["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            return random.choice(responses)
        else:
            return "Tam olarak anlayamadım"





