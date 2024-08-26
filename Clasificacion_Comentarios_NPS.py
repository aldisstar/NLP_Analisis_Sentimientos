# %%
# 1. Preparación del ambiente (Librerias + Data)
import numpy as np
import pandas as pd
import re
import sklearn
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline

import codecs
from unidecode import unidecode
import nltk
nltk.download('stopwords')

# %%

# Ruta del archivo CSV
file_path = r'C:\Users\Aldis\Documents\Master Data Science\GitHub\NLP_Analisis_Sentimientos\Data\Solo_comentarios_clasificados.csv'
# Leer el archivo CSV con diferentes opciones
try:
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8', on_bad_lines='skip')
    print(df.head())
except Exception as e:
    print(f"Ocurrió un error al leer el archivo: {e}")
    
df.columns
# %%
# Revisar la información relevante para el análisis (comentarios y etiquetas)
data = df[['COMENTARIOS', 'ETIQUETAS']]
data = data.dropna() # elimina las filas que figuran como NAN
data.sample(5)
# %%

# Posibles etiquetas
sentiment_values = data["ETIQUETAS"].unique() 
print(f"Los posibles valores de etiquetas son: {sentiment_values}")
# %%

# Separación de datos para test = 15%
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.15)
# %%

# 2. Limpieza de los datos
# Separacion para visualizar cada set
train_pos = train[train['ETIQUETAS'] == 'Positivo']
train_pos = train_pos['COMENTARIOS']
train_neg = train[train['ETIQUETAS'] == 'Negativo']
train_neg = train_neg['COMENTARIOS']
train_neu = train[train['ETIQUETAS'] == 'Neutro']
train_neu = train_neu['COMENTARIOS']

test_pos = test[test['ETIQUETAS'] == 'Positivo']
test_pos = test_pos['COMENTARIOS']
test_neg = test[test['ETIQUETAS'] == 'Negativo']
test_neg = test_neg['COMENTARIOS']
test_neu = test[test['ETIQUETAS'] == 'Neutro']
test_neu = test_neu['COMENTARIOS']

# Visualizar comentarios positivos
print(train_pos.sample(5))
# Visualizar comentarios negativos
print(train_neg.sample(5))
# Visualizar comentarios neutros
print(train_neu.sample(5))
# %%

# Limpieza
def corregir_codificacion(texto):
    try:
        texto_corregido = texto.encode('latin-1').decode('utf-8')
        return texto_corregido
    except UnicodeDecodeError:
        return texto

def eliminar_tildes_y_especiales(texto):
    texto_limpio = unidecode(texto)
    return texto_limpio

def clean(dataset):
    comentarios = []

    # Tomamos un listado de stopwords
    stopwords_set = set(stopwords.words("spanish"))

    # Crear una copia de la tabla original con el texto corregido
    dataset_corregido = dataset.copy()
    dataset_corregido['COMENTARIOS'] = dataset_corregido['COMENTARIOS'].apply(corregir_codificacion)
    dataset_corregido['COMENTARIOS'] = dataset_corregido['COMENTARIOS'].apply(eliminar_tildes_y_especiales)


    for index, row in dataset_corregido.iterrows():
        # Filtramos palabras muy cortas y transformamos a minúscula
        words_filtered = [e.lower() for e in row.COMENTARIOS.split() if len(e) > 3]

        # Eliminamos stopwords
        words_without_stopwords = [word for word in words_filtered if not word in stopwords_set]

        # Guardamos en el vector comentarios con el label correspondiente (positivo o negativo)
        comentarios.append((words_without_stopwords, row.ETIQUETAS))

    # Eliminar palabras de longitud 3 o menos de la tabla original
    dataset_corregido['COMENTARIOS'] = dataset_corregido['COMENTARIOS'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 3]))
    return comentarios, dataset_corregido
# %%

# Limpiar los sets
train_clean, dataset_train = clean(train)
test_clean, dataset_test = clean(test)

# Valor original
print(train.iloc[245].COMENTARIOS)
# Texto corregido original
print(dataset_train.iloc[245].COMENTARIOS)
# Texto corregido vector
print(train_clean[245])
# %%

# 3. Visualizacion wordcloud
train_pos = dataset_train[dataset_train['ETIQUETAS'] == 'Positivo']
train_pos = train_pos['COMENTARIOS']
train_neg = dataset_train[dataset_train['ETIQUETAS'] == 'Negativo']
train_neg = train_neg['COMENTARIOS']

test_pos = dataset_test[dataset_test['ETIQUETAS'] == 'Positivo']
test_pos = test_pos['COMENTARIOS']
test_neg = dataset_test[dataset_test['ETIQUETAS'] == 'Negativo']
test_neg = test_neg['COMENTARIOS']

def wordcloud_draw(data, color = 'black'):
    """"
    Función para crear wordcloud
    """
    words = ' '.join(data)

    wordcloud = WordCloud(
        stopwords=STOPWORDS,
        background_color=color,
        width=2500,
        height=2000
        ).generate(words)

    # Plotear wordcloud
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

print("Palabras positivas")
wordcloud_draw(train_pos,'white')

print("Palabras negativas")
wordcloud_draw(train_neg)
# %%

# 4. Clasificador bayesiano ingenuo
# Lista de palabras
def get_words_in_comentarios(comentarios):
    all = []
    for (words, sentiment) in comentarios:
        all.extend(words)
    return all

all_words = get_words_in_comentarios(train_clean)

# Cálculo de la frecuencia de aparición de cada palabra
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features

w_features = get_word_features(all_words)

# Extractor de features
def extract_features(document):
    """
    Se utiliza el set de un documento en vez de búsqueda en una lista por temas de eficiencia.
    """
    document_words = set(document)
    features = {}
    for word in w_features:
        features['contiene(%s)' % word] = (word in document_words)
    return features

# Apply features
training_set = nltk.classify.apply_features(extract_features, train_clean)

# Entrenamiento
classifier = nltk.NaiveBayesClassifier.train(training_set)
# %%

# 5. Resultados
classifier.show_most_informative_features(10)
# %%
# Accuracy del training set
print(nltk.classify.accuracy(classifier, training_set))
# %%
# Accuracy del test set
test_set = nltk.classify.apply_features(extract_features, test_clean)
print(nltk.classify.accuracy(classifier, test_set))
# %%
