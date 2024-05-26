import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import BigramCollocationFinder
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Marca de tiempo al inicio del código
start_time_total = time.time()

# Lectura del conjunto de datos
inicia_tiempo_carga_datos = time.time()
DATASET_COLUMNAS = ['target', 'title', 'text']
DATASET_CODIFICACION = "ISO-8859-1"
df = pd.read_csv('reviews.csv', encoding=DATASET_CODIFICACION, names=DATASET_COLUMNAS)
termina_tiempo_carga_datos = time.time()
# Calcular el tiempo de lectura de datos
tiempo_carga_datos = termina_tiempo_carga_datos - inicia_tiempo_carga_datos


# Descarga de recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Preprocesamiento de datos
inicia_tiempo_preprocesamiento_datos = time.time()
# Preprocesamiento de datos
data = df[['text', 'target']]
data['target'] = data['target'].replace(1, 0)
data['target'] = data['target'].replace(2, 1)

# Separar datos en positivos y negativos
data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == 0]

# Unificar el conjunto de datos
dataset = pd.concat([data_pos, data_neg])
dataset['text'] = dataset['text'].str.lower()

# Lista de stopwords y otras palabras que no ofrecen información
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
                'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
                's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
                't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
                'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
                "youve", 'your', 'yours', 'yourself', 'yourselves']
STOPWORDS = set(stopwordlist)

def limpiar_stopwords(texto):
    return " ".join([palabra for palabra in str(texto).split() if palabra not in STOPWORDS])

dataset['text'] = dataset['text'].apply(lambda text: limpiar_stopwords(text))

puntuacion_ingles = string.punctuation
lista_puntuaciones = puntuacion_ingles

def limpiar_puntuacion(text):
    traductor = str.maketrans('', '', lista_puntuaciones)
    return text.translate(traductor)

dataset['text'] = dataset['text'].apply(lambda x: limpiar_puntuacion(x))

def limpiar_URLs(data):
    return re.sub('((www.[^\s]+)|(https?://[^\s]+))',' ',data)

dataset['text'] = dataset['text'].apply(lambda x: limpiar_URLs(x))

def limpiar_numeros(data):
    return re.sub('[0-9]+', '', data)

dataset['text'] = dataset['text'].apply(lambda x: limpiar_numeros(x))

tokenizador = RegexpTokenizer(r'\w+')
dataset['text'] = dataset['text'].apply(tokenizador.tokenize)

st = nltk.PorterStemmer()
def stemming(data):
    return [st.stem(palabra) for palabra in data]

dataset['text'] = dataset['text'].apply(lambda x: stemming(x))

lm = nltk.WordNetLemmatizer()
def lematizador(data):
    return [lm.lemmatize(palabra) for palabra in data]

dataset['text'] = dataset['text'].apply(lambda x: lematizador(x))
termina_tiempo_preprocesamiento_datos = time.time()
# Calcular el tiempo de preprocesamiento de datos
tiempo_preprocesamiento_datos = termina_tiempo_preprocesamiento_datos - inicia_tiempo_preprocesamiento_datos


# Separar el texto en positivos y negativos después del preprocesamiento
resenas_pos = dataset[dataset['target'] == 1]['text']
resenas_neg = dataset[dataset['target'] == 0]['text']

# Registrar el tiempo antes del procesamiento
inicia_tiempo_procesamiento = time.time()
# Crear BigramCollocationFinder para reseñas positivas
buscador_bigramas_positivos = BigramCollocationFinder.from_documents(resenas_pos)
buscador_bigramas_positivos.apply_freq_filter(5)

# Crear BigramCollocationFinder para reseñas negativas
buscador_bigramas_negativos = BigramCollocationFinder.from_documents(resenas_neg)
buscador_bigramas_negativos.apply_freq_filter(5)

# Registrar el tiempo después de procesamiento
termina_tiempo_procesamiento = time.time()
# Calcular la diferencia de tiempo
tiempo_procesamiento = termina_tiempo_procesamiento - inicia_tiempo_procesamiento

# Registrar el tiempo antes de análisis
inicia_tiempo_analisis = time.time()
terminos_interes = ["quality", "price", "performance", "design", "functionality"]

# Función para filtrar bigramas por términos de interés
def filtrar_bigramas_por_termino(buscador_bigramas, terminos, top_num=20):
    termino_bigramas = {}
    bigramas = buscador_bigramas.ngram_fd.items()

    for termino in terminos:
        bigramas_filtrados = [(bigrama, frecuencia) for bigrama, frecuencia in bigramas if termino in bigrama]
        bigramas_ordenados = sorted(bigramas_filtrados, key=lambda x: x[1], reverse=True)[:top_num]
        termino_bigramas[termino] = bigramas_ordenados

    return termino_bigramas


# Filtrar bigramas por términos de interés para reseñas positivas
bigramas_positivos_filtrados = filtrar_bigramas_por_termino(buscador_bigramas_positivos, terminos_interes)

# Filtrar bigramas por términos de interés para reseñas negativas
bigramas_negativos_filtrados = filtrar_bigramas_por_termino(buscador_bigramas_negativos, terminos_interes)

print(bigramas_positivos_filtrados)
print(bigramas_negativos_filtrados)

# Función para crear gráficos de barras para bigramas de cada término de interés
def imprimir_bigramas_por_terminos(termino_bigramas, sentimiento):
    for termino, bigramas in termino_bigramas.items():
        if bigramas:
            bigramas_df = pd.DataFrame(bigramas, columns=['bigrama', 'frecuencia'])
            bigramas_df['bigrama'] = bigramas_df['bigrama'].apply(lambda x: ' '.join(x))
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='frecuencia', y='bigrama', data=bigramas_df, palette='viridis')
            plt.title(f'Top 20 Bigramas para "{termino}" en Reseñas {sentimiento}')
            plt.xlabel('Frecuencia')
            plt.ylabel('Bigrama')
            

# Visualizar bigramas más comunes para términos de interés en reseñas positivas
imprimir_bigramas_por_terminos(bigramas_positivos_filtrados, "Positivas")

# Visualizar bigramas más comunes para términos de interés en reseñas negativas
imprimir_bigramas_por_terminos(bigramas_negativos_filtrados, "Negativas")

# Registrar el tiempo después de la sección de interés
termina_tiempo_analisis = time.time()
# Calcular la diferencia de tiempo
tiempo_analisis = termina_tiempo_analisis - inicia_tiempo_analisis

# Marca de tiempo al inicio del código
termina_tiempo_total = time.time()
# Calcular la diferencia de tiempo
tiempo_total = termina_tiempo_total - start_time_total

print(f"Tiempo de lectura de datos: {tiempo_carga_datos} segundos")
print(f"Tiempo de preprocesamiento de datos: {tiempo_preprocesamiento_datos} segundos")
print("Tiempo de procesamiento:", tiempo_procesamiento, "segundos")
print("Tiempo de análisis:", tiempo_analisis, "segundos")
print("Tiempo total:", tiempo_total, "segundos")

terminos_interes = ["quality", "price", "performance", "design", "functionality"]
plt.show()
