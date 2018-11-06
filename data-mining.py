import glob
import os
import re
import pandas as pd
import nltk

# download dos pacotes para stemming e remoção de stopwords
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# caminho da pasta com os artigos
articles_dir = './txt/'
# cria uma lista com os caminhos dos arquivos de artigos
filenames = [f for f in glob.glob(os.path.join(articles_dir, '*.txt'))]

stemmer = PorterStemmer()

# Lê o texto de um arquivo, fazendo os devidos pré-processamentos
def get_data (filename):
    with open(filename, 'r', encoding='latin1') as f:
        corpus = f.read() # leitura do corpuso do arquivo
        corpus = re.sub('[^A-Za-z]', ' ', corpus) # Retira caracteres não alfanumericos
        corpus = corpus.lower() # Torna todas as palavras minúsculas

        corpus = word_tokenize(corpus) # tokenização do corpuso
        # remoção das stopwords
        for token in corpus:
            if token in stopwords.words('english'):
                corpus.remove(token)

        for i in range(len(corpus)): # Processo de Stemming
            corpus[i] = stemmer.stem(corpus[i])

        plain_corpus = " ".join(corpus) # Transforma o array de tokens em uma string única por artigo
        return plain_corpus

# Pega o tema do artigo através do nome do arquivo
def get_class (filename):
    return filename.split('-')[0].split('\\')[1]

# criação de um dataframe do pandas com uma coluna de temas e de textos
dataset = pd.DataFrame({'themes': [get_class(f) for f in filenames],
                        'data': [get_data(f) for f in filenames]})

# transformação em matriz de termo/documento
cv = CountVectorizer(max_features=1000, encoding='latin1')
# gerador de 'folds' normalizadas para um k-fold cross-validation
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
nb = GaussianNB()

X = cv.fit_transform(dataset.data).toarray() # Bag of words de cada artigo
y = dataset.iloc[:, 0] # Classificação dos artigos

accuracy = 0
for train_index, test_index in skf.split(X, y):
    # separação entre teste e treinamento
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    nb.fit(X_train, y_train) # Fase de treinamento

    y_pred = nb.predict(X_test) # Fase de teste

    cr = classification_report(y_test, y_pred)
    print(cr)

    accuracy += accuracy_score(y_test, y_pred)

print (accuracy/n_splits)
