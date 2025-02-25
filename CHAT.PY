import random
import json
import nltk
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# **Expanded Intents Dataset**
intents = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hello", "Hey", "What's up?", "Hola"], 
         "responses": ["Hello!", "Hey there!", "Hi! How can I help?", "Hi, how's your day?"]},

        {"tag": "goodbye", "patterns": ["Bye", "See you", "Goodbye", "Take care"], 
         "responses": ["Goodbye!", "See you later!", "Bye! Have a great day!"]},

        {"tag": "thanks", "patterns": ["Thanks", "Thank you", "I appreciate it"], 
         "responses": ["You're welcome!", "No problem!", "Glad to help!"]},

        {"tag": "how_are_you", "patterns": ["How are you?", "How's it going?", "How do you feel today?"], 
         "responses": ["I'm doing great!", "All good! How about you?", "I'm fine, thanks for asking!"]},

        {"tag": "ai_definition", "patterns": ["What is AI?", "Explain artificial intelligence", "What does AI mean?"], 
         "responses": ["AI stands for Artificial Intelligence, which enables machines to learn and make decisions like humans."]},

        {"tag": "programming_languages", "patterns": ["What programming languages do you know?", "Tell me about coding languages"], 
         "responses": ["I know Python, Java, C++, JavaScript, and many more! What do you want to learn?"]},

        {"tag": "python", "patterns": ["Tell me about Python", "What is Python used for?"], 
         "responses": ["Python is a versatile programming language used in AI, data science, web development, and more."]},

        {"tag": "machine_learning", "patterns": ["What is machine learning?", "Explain ML", "How does machine learning work?"], 
         "responses": ["Machine learning is a field of AI that enables computers to learn from data without being explicitly programmed."]},

        {"tag": "math_help", "patterns": ["Solve 2+2", "What is 10 divided by 2?", "Calculate 5*5"], 
         "responses": ["That's easy! The answer is 4.", "10 divided by 2 is 5.", "5 multiplied by 5 is 25."]},

        {"tag": "jokes", "patterns": ["Tell me a joke", "Make me laugh"], 
         "responses": ["Why don’t skeletons fight each other? They don’t have the guts!", "I told my wife she should embrace her mistakes. She gave me a hug."]},

        {"tag": "fun_facts", "patterns": ["Tell me a fun fact", "Give me an interesting fact"], 
         "responses": ["Did you know? Honey never spoils! Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly good to eat!"]},

        {"tag": "who_are_you", "patterns": ["Who are you?", "What are you?"], 
         "responses": ["I'm a chatbot designed to assist you!", "I'm an AI-powered assistant here to help with your queries."]},

        {"tag": "chatting", "patterns": ["What are you doing?", "Are you busy?", "Can we chat?"], 
         "responses": ["I'm always here to chat! What’s on your mind?", "Just waiting for your questions!"]},

        {"tag": "weather", "patterns": ["How's the weather?", "Is it raining today?", "What's the temperature?"], 
         "responses": ["I can’t check the weather directly, but you can use a weather app for live updates."]},

        {"tag": "age", "patterns": ["How old are you?", "When were you created?"], 
         "responses": ["I'm as old as technology allows me to be!", "I was created recently, but I learn fast!"]}
    ]
}

# **Step 1: Data Preprocessing**
words, classes, documents = [], [], []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)  
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# **Create Training Data**
training = []
output_empty = [0] * len(classes)

for doc in documents:
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    bag = [1 if w in word_patterns else 0 for w in words]
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

x_train = np.array(list(training[:, 0]))
y_train = np.array(list(training[:, 1]))

print(f"Training Data Shapes: X={x_train.shape}, Y={y_train.shape}")

# **Step 2: Check & Load Model**
try:
    model = load_model('chatbot_model.h5')
    print("Model loaded successfully!")
except:
    print("Training new model...")
    model = Sequential()
    model.add(Dense(128, input_shape=(len(words),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5')

# **Step 3: Chatbot Functions**
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bow = [1 if w in sentence_words else 0 for w in words]
    return np.array(bow)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list):
    if not intents_list:
        return "I'm not sure how to respond to that."
    
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure."

# **Step 4: Chatbot Execution**
print("Chatbot is ready! Type 'exit' to stop.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    ints = predict_class(user_input)
    response = get_response(ints)
    print("Bot:", response)
