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

print("lendo arquivos e executando pré-processamento...")
# criação de um dataframe do pandas com uma coluna de temas e de textos
dataset = pd.DataFrame({'theme': [get_class(f) for f in filenames],
                        'data': [get_data(f) for f in filenames]})

print("transformando textos em matriz termo-documento...")
# transformação em matriz de termo/documento
cv = CountVectorizer(max_features=1000, encoding='latin1')

X = cv.fit_transform(dataset.data).toarray() # Bag of words de cada artigo
words = cv.get_feature_names() # palavras presentes em X
y = dataset.theme # Classificação dos artigos

# matriz de termo/documento com coluna de classe
tdm = pd.concat([y, pd.DataFrame(X, columns=words)], axis=1)

# determina a classe de um documento utilizando naive bayes
def bayes(document, train_set, themes):
    # dicionário que guarda as probabilidades de cada tema
    probabilities = {}

    # calcula a probabilidade condicional (parcial) de cada tema
    for theme in themes:
        # separa a porção do dataset associada ao tema atual
        theme_set = train_set[train_set.theme == theme]
        # calcula a probabilidade do tema atual (P(A))
        theme_prob = (len(theme_set) / len(train_set))

        # calcula a probabilidade condicional para cada palavra (P(B|A))
        for word in theme_set.columns[1:]:
            theme_prob *= ((len(theme_set[theme_set[word] == document[word]]) + 0.1) /
                           (len(theme_set) + 0.1))

        # guarda o valor no dicionário de probabilidades
        probabilities[theme] = theme_prob

    # retorna a entrada no dicionário
    return max(probabilities, key=lambda k: probabilities[k])

# gerador de 'folds' normalizadas para um k-fold cross-validation
n_splits = 10 # quantidade de folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

print(f"realizando {n_splits}-fold cross-validation...")

# inicialização da acurácia
accuracy = 0
# criação do conjunto de temas possíveis
themes = list(set(tdm.theme))
count = 0 # contagem de folds
for train_index, test_index in skf.split(X, y):
    # separação entre teste e treinamento
    train_set = tdm.iloc[train_index]
    test_set = tdm.iloc[test_index]

    print(f"fold {count} ({len(test_set)} itens):")

    correct_count = 0 # contagem de exemplos corretamente previstos
    for index, doc in test_set.iterrows():
        # calcula a predição para um exemplo do conjunto de tetes
        pred = bayes(doc, train_set, themes)
        # incrementa contagem de previsões corretas caso esta seja correta
        correct = (pred == doc.theme)
        correct_count += 1 if correct else 0
        print(f"{'.' if correct else 'x'}", end='', flush=True)

    correct_count /= len(test_set)
    print(f"\nfold {count} accuracy: {correct_count}")

    # adiciona à acurácia total a acurácia obtida neste fold
    accuracy += correct_count / len(test_set)

    # incrementa a contagem de folds
    count += 1

# divide a acurácia pela quantidade de folds, efetivamente calculando a média
# das acurácias
accuracy /= n_splits

print (accuracy)
