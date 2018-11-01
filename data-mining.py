import glob
import os
import re
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

path = './txt/'
stemmer = PorterStemmer()
cv = CountVectorizer(max_features=1000, encoding='latin1')
skf = StratifiedKFold(n_splits=10, shuffle=True)
nb = GaussianNB()

# Pega o tema do artigo através do nome do arquivo
def get_class (filename):
    return filename.split('-')[0].split('\\')[1]

def get_data (filename):
    with open(filename, 'r', encoding='latin1') as f:

        data = f.read()
        data = re.sub('[^A-Za-z]', ' ', data) # Retira caracteres não alfanumericos
        data = data.lower() # Torna todas as palavras minúsculas

        data = word_tokenize(data) # Remoção das stop words
        for token in data:
            if token in stopwords.words('english'):
                data.remove(token)

        for i in range(len(data)): # Processo de Stemming
            data[i] = stemmer.stem(data[i])

        plain_text = " ".join(data) # Transforma o array de tokens em uma string única por artigo
        return plain_text

names = [f for f in glob.glob(os.path.join(path, '*.txt'))]

dataset = pd.DataFrame({'themes' : [get_class(f) for f in names], 'data' : [get_data(f) for f in names]})

X = cv.fit_transform(dataset.data).toarray() # Bag of words de cada artigo
y = dataset.iloc[:, 0] # Classificação dos artigos

accuracy = 0
for train_index, test_index in skf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    nb.fit(X_train, y_train) # Fase de treinamento

    y_pred = nb.predict(X_test) # Fase de teste

    cr = classification_report(y_test, y_pred)
    print(cr)

    accuracy = accuracy + accuracy_score(y_test, y_pred)

print (accuracy/10)