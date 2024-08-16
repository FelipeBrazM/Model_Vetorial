import os
import nltk 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('rslp')
nltk.download('wordnet')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words("portuguese") 
stopwords.sort()

# Função para ler as questões de arquivos individuais na pasta 'questions_OP'
def read_questions_from_folder(folder_path):
    questions = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                question = file.read().strip()
                questions.append(question)
    return questions

# Função para perguntar se deseja salvar as respostas e salvar em um arquivo se sim
def ask_to_save_results(results):
    save_results = input("Deseja salvar os resultados em um arquivo .txt? (s/n): ").strip().lower()
    if save_results == 's':
        output_file = "resultados_consulta.txt"
        with open(output_file, 'w', encoding='utf-8') as file:
            for doc, similarity in results:
                if similarity != 0.0:
                    file.write(f"Grau de similaridade: {similarity:.2f}\n{doc}\n")
                    file.write("=====================================================================================================================================================================\n")
        print(f"Resultados salvos em {output_file}\n")

# Use a função para ler as questões da pasta 'questions_OP'
folder_path = 'questions_OP'
documents = read_questions_from_folder(folder_path)

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

# Segundo passo: Calcular o TF-IDF 
preprocessed_documents = [' '.join(doc) for doc in processed_documents]
preprocessed_query = ' '.join(processed_query)

# Criar um TF-IDF em forma de vetor 
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)

# Transformando a consulta em um vetor TF-IDF
query_vector = tfidf_vectorizer.transform([preprocessed_query])

# Terceiro passo: Calcular similaridade de cosseno 
cosine_sim = cosine_similarity(query_vector, tfidf_matrix)

# Quarto passo: Ranquear documentos pela similaridade 
results = [(documents[i], cosine_sim[0][i]) for i in range(len(documents))]
results.sort(key=lambda x: x[1], reverse=True)

# Printa os documentos ranqueados
found = False
for doc, similarity in results: 
    if similarity != 0.0:
        print(f"Grau de similaridade: {similarity:.2f}\n{doc}\n")
        print("=====================================================================================================================================================================\n")
        found = True

# Se não encontrar nenhum documento com similaridade com a entrada printa essa mensagem para o usuário
if not found:
    print(f"Nenhum resultado encontrado na base de dados para essa consulta!\n")

# Pergunta ao usuário se deseja salvar os resultados
ask_to_save_results(results)
