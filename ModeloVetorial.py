""" Para implementar o modelo Vetorial, vamos seguir alguns passos:
    Primeiro: processamento do texto ;
    Segundo: Calcular o TF-IDF;
    Terceiro: Calcular a similaridade de coscenos;
    Quarto: Ranquear os documentos pela similaridade.
"""
# Para executar o código colocar na linha de comando: python ModeloVetorial.py

# Essa biblioteca (nltk), auxilia no trabalho com a linguagem natural
import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('rslp')
nltk.download('wordnet')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words # Lista de stopwords
stopwords = nltk.corpus.stopwords.words("portuguese") # Pegando as stopwords em português
stopwords.sort() # Ordenando as stopwords por ordem alfabética 

# Função para ler as questões de um arquivo de texto
def read_txt(name_file):
    with open(name_file, 'r', encoding='utf-8') as file:
        questions = file.readlines()
    # Remover o caractere de nova linha de cada questão
    questions = [q.strip() for q in questions]
    return questions

# Use a função para ler as questões
name_file = 'questions.txt'
documents = read_txt(name_file)

# Entrada da consulta
query = input("O que deseja consultar no banco de questões?\n")

# Primeiro passo: Tokenizar e processar o texto
from nltk.tokenize import word_tokenize
tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
tokenized_query = word_tokenize(query.lower())

# Função para remover stopwords e fazer a radicalização
def preprocess(text):
    stemmer = nltk.stem.RSLPStemmer()
    return [stemmer.stem(word) for word in text if word not in stopwords]

# Preprocessar a consulta
processed_query = preprocess(tokenized_query)

# Preprocessar os documentos
processed_documents = [preprocess(doc) for doc in tokenized_documents]

# Segundo passo: Calcular o TF-IDF (Term Frequency — Inverse Document Frequency) 
# Converter documentos tokenizados em texto
preprocessed_documents = [' '.join(doc) for doc in processed_documents]
preprocessed_query = ' '.join(processed_query)
print("\n",preprocessed_query,"\n")

# Criar um TF-IDF em forma de vetor 
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)

# Transformando a consulta em um vetor TF-IDF
query_vector = tfidf_vectorizer.transform([preprocessed_query])

# Terceiro passo: Calcular similaridade de cosseno 
cosine_similarity = cosine_similarity(query_vector, tfidf_matrix)

# Quarto passo: Ranquear documentos pela similaridade 
results = [(documents[i], cosine_similarity[0][i]) for i in range (len(documents))]
results.sort(key=lambda x : x[1], reverse=True)

# Printa os documentos ranqueados
found = False
for doc, similarity in results: 
    if similarity != 0.0:
        print(f"Grau de similaridade: {similarity:.2f}\n{doc}\n")
        found = True
# Se não encontrar nenhum documento com similaridade com a entrada printa essa mensagem para o usuário
if found == False:
    print(f"Nenhum resultado encontrado na base de dados para essa consulta!\n")

