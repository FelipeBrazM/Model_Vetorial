import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
documents = [

    # **Algoritmos e Estruturas de Dados:**

    "Quais são os diferentes tipos de algoritmos de classificação e qual é o mais adequado para cada tipo de problema?",
    "Como podemos utilizar estruturas de dados eficientes, como árvores binárias e tabelas hash, para otimizar o tempo de busca e acesso a informações?",
    "Explique a diferença entre recursão e iteração e quando cada uma deve ser utilizada na resolução de problemas.",
    "Como podemos analisar a complexidade de tempo e espaço de um algoritmo e qual a importância dessa análise?",
    "Descreva diferentes algoritmos de busca, como busca binária e busca em largura, e suas aplicações práticas.",

    # **Programação:**

    "Quais são os paradigmas de programação mais utilizados e quais as características de cada um?",
    "Como podemos utilizar funções e módulos para organizar e reutilizar código em Python?",
    "Explique a diferença entre variáveis ​​locais e globais e como o escopo de variável impacta o programa.",
    "Quais são os diferentes tipos de exceções em Python e como podemos tratá-las de forma eficiente?",
    "Descreva as boas práticas de programação em Python para escrever código limpo, legível e fácil de manter.",

    # **Sistemas Operacionais:**

    "Como os sistemas operacionais gerenciam memória, processos e arquivos?",
    "Explique a diferença entre multitarefa preemptiva e multitarefa cooperativa.",
    "Quais são os diferentes tipos de sistemas de arquivos e como cada um organiza e armazena dados?",
    "Como a segurança é implementada em sistemas operacionais para proteger o sistema e seus dados?",
    "Descreva os principais componentes de um kernel de sistema operacional e suas funções.",

    # **Redes de Computadores:**

    "Como os dados são transmitidos na internet e quais protocolos são utilizados para essa comunicação?",
    "Explique a diferença entre endereços IP e nomes de domínio e como eles são utilizados para identificar computadores na rede.",
    "Quais são os diferentes tipos de topologias de rede e como cada uma impacta o desempenho e a confiabilidade da rede?",
    "Como a segurança é implementada em redes de computadores para proteger contra ataques e proteger dados?",
    "Descreva os diferentes protocolos de rede, como TCP/IP e HTTP, e suas funções.",

    # **Inteligência Artificial:**

    "Como os algoritmos de aprendizado de máquina podem ser utilizados para prever eventos futuros e tomar decisões?",
    "Explique a diferença entre aprendizado supervisionado e aprendizado não supervisionado e qual tipo é mais adequado para cada problema.",
    "Como o processamento de linguagem natural pode ser utilizado para entender e processar linguagem humana?",
    "Descreva os diferentes tipos de redes neurais artificiais e como elas aprendem com dados.",
    "Quais são as implicações éticas do desenvolvimento e uso da inteligência artificial?",

    # **Banco de Dados:**

    "Quais são os diferentes modelos de banco de dados, como relacional, NoSQL e grafo, e qual é o mais adequado para cada tipo de aplicação?",
    "Explique como o SQL é utilizado para consultar e manipular dados em bancos de dados relacionais.",
    "Como podemos garantir a integridade e consistência dos dados em um banco de dados?",
    "Descreva as diferentes técnicas de otimização de desempenho para consultas em bancos de dados.",
    "Quais são as medidas de segurança que devem ser tomadas para proteger um banco de dados contra acessos não autorizados?",

    # **Segurança da Informação:**

    "Quais são os diferentes tipos de ameaças à segurança da informação e como podemos nos proteger contra elas?",
    "Explique como a criptografia pode ser utilizada para proteger dados confidenciais.",
    "Como podemos implementar um sistema de autenticação e autorização seguro para controlar o acesso a recursos?",
    "Descreva as melhores práticas para garantir a segurança da informação em uma organização.",
    "Quais são as leis e regulamentos relacionados à segurança da informação que devem ser cumpridos?",

    # **Desenvolvimento Web:**

    "Quais são as diferentes tecnologias front-end e back-end utilizadas para desenvolver websites e aplicativos web?",
    "Explique como o HTML, CSS e JavaScript são utilizados para criar interfaces web interativas.",
    "Como podemos utilizar frameworks web, como Django e React, para agilizar o desenvolvimento web?",
    "Descreva as melhores práticas para garantir a performance, acessibilidade e segurança de websites e aplicativos web.",
    ]

# Sample query
query = "Web"

# Step 1: Tokenize and preprocess the text
nltk.download('punkt')
from nltk.tokenize import word_tokenize
tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
tokenized_query = word_tokenize(query.lower())

# Step 2: Calculate TF-IDF
# Convert tokenized documents to text
preprocessed_documents = [' '.join(doc) for doc in tokenized_documents]
preprocessed_query = ' '.join(tokenized_query)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_documents)

# Transform the query into a TF-IDF vector
query_vector = tfidf_vectorizer.transform([preprocessed_query])

# Step 3: Calculate cosine similarity
cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)

# Step 4: Rank documents by similarity
results = [(documents[i], cosine_similarities[0][i]) for i in range(len(documents))]
results.sort(key=lambda x: x[1], reverse=True)

# Print the ranked documents
for doc, similarity in results:
    if similarity != 0.0:
        print(f"Similarity: {similarity:.2f}\n{doc}\n")