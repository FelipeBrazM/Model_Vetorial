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
stopwords = nltk.corpus.stopwords.words # Lista de stopwords
stopwords = nltk.corpus.stopwords.words("portuguese") # Pegando as stopwords em português
stopwords.sort() # Ordenando as stopwords por ordem alfabética 

# Exemplo de documentos
documents = [
    # **Algoritmos e Estruturas de Dados:**

    "Quais são os diferentes tipos de algoritmos de classificação e qual é o mais adequado para cada tipo de problema?",
    "Explique a diferença entre recursão e iteração e quando cada uma deve ser utilizada na resolução de problemas.",
    "Como podemos analisar a complexidade de tempo e espaço de um algoritmo e qual a importância dessa análise?",
    "Descreva diferentes algoritmos de busca, como busca binária e busca em largura, e suas aplicações práticas.",
    "Explique o funcionamento da busca binária e da busca em largura, destacando suas principais vantagens e desvantagens.",
    "Compare e contraste os algoritmos de busca binária e busca em profundidade. Quais são as situações ideais para usar cada um?",
    "Quais são as aplicações mais comuns dos algoritmos de busca binária e busca em largura na ciência da computação?",
    "Discuta como a busca binária e a busca em largura podem ser otimizadas em termos de desempenho.",
    "Explique como funcionam a busca em largura e a busca em profundidade, e forneça exemplos práticos de uso.",
    "Descreva o algoritmo de busca em largura e sua importância na teoria dos grafos.",
    "Compare a eficiência da busca binária com a busca linear em diferentes cenários práticos.",
    "Como a busca em largura é utilizada em algoritmos de inteligência artificial, como a busca em árvore de decisão?",
    "Qual é a diferença entre busca binária e busca ternária, e quando seria vantajoso usar uma sobre a outra?",
    "Discuta as aplicações da busca binária na organização e gerenciamento de dados em bancos de dados.",

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
    "Como os endereços IP e os nomes de domínio trabalham juntos para facilitar a navegação na internet?",
    "Quais são as diferenças entre IPv4 e IPv6 e como essas versões afetam a identificação de dispositivos na rede?",
    "O que é DNS (Sistema de Nomes de Domínio) e qual sua importância na tradução de nomes de domínio para endereços IP?",
    "Descreva o processo de resolução de nomes de domínio e como um nome de domínio é convertido em um endereço IP.",
    "Como os endereços IP estáticos e dinâmicos diferem e quais são seus usos na rede?",
    "Quais são os principais componentes de um endereço IP e como eles contribuem para a identificação de computadores na rede?",
    "Explique como funciona o DHCP (Protocolo de Configuração Dinâmica de Host) na atribuição de endereços IP.",
    "O que é um roteador e como ele usa endereços IP para encaminhar pacotes de dados na rede?",
    "Qual é a função de um servidor de nomes de domínio (DNS) na infraestrutura da internet?",
    "Quais são as vantagens e desvantagens de usar endereços IP públicos versus privados na rede?",
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
    "Como escolher entre um banco de dados relacional e um banco de dados NoSQL para um projeto específico?",
    "Quais são as principais diferenças entre bancos de dados relacionais e bancos de dados de grafo?",
    "Em quais cenários é mais vantajoso utilizar um banco de dados NoSQL em vez de um relacional?",
    "Quais são os benefícios e desvantagens de utilizar bancos de dados de grafo para representar dados complexos?",
    "Quais são os casos de uso típicos para bancos de dados relacionais, NoSQL e de grafo?",
    "Como a estrutura dos dados influencia na escolha entre um banco de dados relacional, NoSQL e de grafo?",
    "Quais são os principais exemplos de bancos de dados NoSQL e quais são suas características distintivas?",
    "Em que situações um banco de dados de grafo é mais eficiente que um banco de dados relacional ou NoSQL?",
    "Quais são as vantagens e desvantagens de utilizar um banco de dados relacional em comparação com um banco de dados NoSQL?",
    "O que são sistemas de gerenciamento de banco de dados NoSQL e quais são suas vantagens?",

    # **Segurança da Informação:**

    "Quais são os diferentes tipos de ameaças à segurança da informação e como podemos nos proteger contra elas?",
    "Explique como a criptografia pode ser utilizada para proteger dados confidenciais.",
    "Como podemos implementar um sistema de autenticação e autorização seguro para controlar o acesso a recursos?",
    "Descreva as melhores práticas para garantir a segurança da informação em uma organização.",
    "Quais são as leis e regulamentos relacionados à segurança da informação que devem ser cumpridos?",

    # **Outros tópicos:**

    "Quais são as diferentes tecnologias front-end e back-end utilizadas para desenvolver websites e aplicativos web?",
    "Explique como o HTML, CSS e JavaScript são utilizados para criar interfaces web interativas.",
    "Como podemos utilizar frameworks web, como Django e React, para agilizar o desenvolvimento web?",
    "Descreva as melhores práticas para garantir a performance, acessibilidade e segurança de websites e aplicativos web.",
    "Como podemos utilizar estruturas de dados eficientes, como árvores binárias e tabelas hash, para otimizar o tempo de busca e acesso a informações?",
    "Como podemos analisar a complexidade de tempo e espaço de um algoritmo e qual a importância dessa análise?",
    "O que são grafos e como podemos utilizá-los para representar e resolver problemas de redes?",
    "Explique o funcionamento do algoritmo de Dijkstra para encontrar o caminho mais curto em um grafo.",
    "Quais são as diferenças entre algoritmos guloso e de programação dinâmica?",
    "Como o algoritmo de Kruskal é utilizado para encontrar árvores geradoras mínimas em grafos?",
    "Descreva o algoritmo de ordenação merge sort e suas vantagens em relação a outros métodos de ordenação.",
    "Como o balanceamento de árvores binárias impacta a eficiência das operações de inserção, busca e exclusão?",
    "Explique o conceito de hash table e como colisões podem ser tratadas em sua implementação.",
    "Quais são as vantagens e desvantagens de utilizar listas ligadas em vez de arrays?",
    "Como o algoritmo quicksort funciona e quais são seus casos de pior e melhor desempenho?",
    "Explique a diferença entre grafos direcionados e não direcionados e suas aplicações.",
    "Como podemos utilizar heaps para implementar uma fila de prioridade?",
    "Descreva o algoritmo de Floyd-Warshall para encontrar caminhos mínimos entre todos os pares de vértices em um grafo.",
    "Quais são as principais diferenças entre árvores AVL e árvores vermelho-preto?",
    "Como a técnica de backtracking pode ser aplicada para resolver problemas de otimização?",
    "Explique a importância do algoritmo de Huffman na compressão de dados.",
    "Quais são os casos de uso típicos para uma fila (queue) em programação?",
    "Como podemos utilizar matrizes de adjacência e listas de adjacência para representar grafos?",
    "Descreva o algoritmo de Bellman-Ford para encontrar caminhos mais curtos e como ele lida com pesos negativos.",
    "Quais são as aplicações práticas de árvores trie na busca de padrões em textos?",
    "Explique a diferença entre busca em profundidade (DFS) e busca em largura (BFS) em grafos.",
    "Como a técnica de memoization pode ser usada para otimizar algoritmos recursivos?",
    "Quais são os benefícios de utilizar um grafo bipartido em problemas de correspondência?",
    "Descreva como funciona a técnica de dividir e conquistar e dê exemplos de algoritmos que a utilizam.",
    "Como o algoritmo de Ford-Fulkerson é utilizado para resolver problemas de fluxo máximo em redes de transporte?",
    "Quais são as diferenças entre busca binária e busca linear e quando cada uma deve ser utilizada?",
    "Explique a importância de balanceamento em árvores B-trees para sistemas de banco de dados.",
    "Como podemos utilizar a técnica de branch and bound para resolver problemas de programação inteira?",
    "Quais são os principais algoritmos de ordenação e como eles se comparam em termos de complexidade de tempo e espaço?",
    "Como a técnica de programação dinâmica difere da abordagem gulosa?",
    "Descreva como a técnica de poda alfa-beta é utilizada em algoritmos de busca de jogos.",
    "Quais são as aplicações de grafos direcionados acíclicos (DAG) na computação?",
    "Como a técnica de força bruta pode ser aplicada a problemas de busca e quais são suas limitações?",
    "Explique a diferença entre filas de prioridade implementadas com heaps e com listas ligadas.",
    "Quais são os benefícios e desvantagens de utilizar conjuntos disjuntos para representar partições?",
    "Como a análise de complexidade assintótica ajuda na avaliação de algoritmos?",
    "Descreva a técnica de hash duplo e suas vantagens sobre o hashing simples.",
    "Quais são os principais tipos de árvores de pesquisa balanceadas e suas diferenças?",
    "Como a técnica de backtracking pode ser combinada com memoization para resolver problemas de programação dinâmica?",
    "Explique a diferença entre árvore geradora mínima (MST) e caminho mínimo em grafos.",
    "Quais são as vantagens de utilizar uma estrutura de dados de conjunto disjunto (union-find)?",
    "Descreva o algoritmo de ordenação heap sort e suas aplicações.",
    "Como a técnica de busca local pode ser aplicada a problemas de otimização combinatória?",
    "Quais são as diferenças entre um grafo ponderado e um grafo não ponderado?",
    "Explique o conceito de árvore de sufixos e suas aplicações em processamento de textos.",
    "Como podemos utilizar listas duplamente ligadas para implementar estruturas de dados mais complexas?",
    "Descreva a técnica de ordenação por contagem (counting sort) e suas limitações.",
    "Quais são as diferenças entre algoritmos de busca exata e busca aproximada?",
    "Como a técnica de busca tabu pode ser utilizada para evitar soluções repetidas em problemas de otimização?",
    "Explique a importância de árvores B+ na indexação de bancos de dados.",
    "Quais são as vantagens de utilizar uma fila de prioridade com heaps binários?",
    "Descreva o algoritmo de ordenação radix sort e como ele lida com diferentes bases.",
    "Como a técnica de branch and bound difere da busca em profundidade com poda?",
    "Quais são os benefícios de utilizar uma estrutura de dados deque (double-ended queue)?",
    "Explique a importância da complexidade de tempo polinomial em algoritmos.",
    "Como a técnica de programação linear é utilizada para resolver problemas de otimização?",
    "Descreva o algoritmo de busca A* e suas aplicações em inteligência artificial.",
    "Quais são as diferenças entre grafos completos e grafos bipartidos completos?",
    "Como a técnica de hashing universal pode ser utilizada para evitar colisões?",
    "Explique a diferença entre algoritmos de busca heurística e exata.",
    "Quais são as vantagens de utilizar uma estrutura de dados trie para busca de prefixos?",
    "Descreva como a técnica de busca em largura pode ser utilizada para resolver problemas de labirinto.",
    "Como a técnica de redução de problema pode ser utilizada para provar NP-completude?",
    "Quais são as diferenças entre algoritmos de compressão com perda e sem perda?",
    "O que é a arquitetura RISC (Reduced Instruction Set Computer) e como ela se diferencia da arquitetura CISC?",
    "Quais são os desafios de segurança associados à Internet das Coisas (IoT) e como podem ser mitigados?",
    "Explique o conceito de recursão em linguagens de programação e forneça exemplos de sua aplicação.",
    "Como funcionam os sistemas de recomendação em plataformas online e quais são suas principais abordagens?",
    "O que é a teoria dos grafos e como ela é aplicada em problemas de ciência da computação?",
    "Quais são as técnicas de otimização de consultas em bancos de dados e como elas podem melhorar o desempenho?",
    "Como as técnicas de visão computacional são aplicadas em problemas de reconhecimento de imagem?",
    "O que é o princípio de localidade na ciência da computação e como ele influencia o desempenho de sistemas?",
    "Quais são os principais tipos de vulnerabilidades de segurança em software e como podem ser exploradas?",
    "Explique o conceito de escalabilidade em sistemas distribuídos e como ela é alcançada.",
    "Qual é a importância da normalização de banco de dados e como ela ajuda na eficiência e integridade dos dados?",
    "O que são algoritmos de aprendizado por reforço e quais são suas aplicações?",
    "Como funcionam os sistemas de detecção de intrusão baseados em assinaturas e em comportamento?",
    "Quais são os métodos de autenticação mais comuns em sistemas de segurança de computadores?",
    "O que é a arquitetura de computação em cluster e em que casos ela é utilizada?",
    "Explique o conceito de balanceamento de carga em servidores web e quais são suas técnicas mais populares.",
    "Quais são os diferentes tipos de testes de software e como eles contribuem para a qualidade do produto?",
    "Como os algoritmos de busca heurística são aplicados em problemas de otimização?",
    "O que são linguagens de programação de script e quais são exemplos populares?",
    "Explique o conceito de sincronização de threads em programação concorrente.",
    "Quais são os benefícios e desafios da computação quântica em relação à computação clássica?",
    "Como funcionam os algoritmos de detecção de colisão em jogos de computador?",
    "O que são sistemas de gerenciamento de conteúdo (CMS) e como eles facilitam a criação de sites?",
    "Explique o funcionamento do algoritmo de classificação K-means e em quais situações é utilizado.",
    "Quais são as técnicas de segurança mais comuns para proteger redes sem fio?",
    "Como os algoritmos de aprendizado profundo são aplicados em problemas de processamento de linguagem natural?",
    "Quais são as principais técnicas de aprendizado profundo usadas em processamento de linguagem natural?",
    "Como as redes neurais convolucionais são utilizadas no processamento de linguagem natural?",
    "Quais são os desafios enfrentados ao aplicar aprendizado profundo em processamento de linguagem natural?",
    "De que forma os modelos de linguagem como BERT e GPT são aplicados em tarefas de processamento de linguagem natural?",
    "Como a transformação de texto em vetores (word embeddings) facilita o uso de aprendizado profundo no processamento de linguagem natural?",
    "Qual é o papel das redes neurais recorrentes no processamento de linguagem natural?",
    "Como o aprendizado profundo melhora a precisão dos sistemas de tradução automática?",
    "Quais são as diferenças entre os métodos tradicionais de processamento de linguagem natural e os baseados em aprendizado profundo?",
    "Como os modelos de aprendizado profundo são treinados para entender o contexto em processamento de linguagem natural?",
    "De que forma o aprendizado profundo contribui para o desenvolvimento de chatbots e assistentes virtuais no processamento de linguagem natural?",
    "O que são as primitivas de sincronização e como elas são usadas para garantir a consistência de dados?",
    "Explique o conceito de computação heterogênea e como ela é aplicada em sistemas de alto desempenho.",
    "Quais são as técnicas de prevenção de ataques de negação de serviço (DDoS) e como elas funcionam?",
    "Como funcionam os algoritmos de filtragem colaborativa em sistemas de recomendação?",
    "O que são algoritmos de aprendizado não supervisionado e em que tipo de problemas são aplicados?",
    "Quais são os desafios de escalabilidade em sistemas de banco de dados distribuídos?",
    "Explique o que é uma linguagem de consulta de banco de dados e como ela difere da SQL.",
    "Como as técnicas de mineração de dados são aplicadas na análise de grandes conjuntos de dados?",
    "O que são sistemas de detecção de intrusão baseados em anomalias e como eles funcionam?",
    "Quais são os princípios do desenvolvimento seguro de software e como são implementados?",
    "Explique o conceito de criptografia de curva elíptica e suas vantagens sobre outras técnicas.",
    "Como funcionam os sistemas de armazenamento distribuído e quais são suas vantagens e desvantagens?",
    "O que é a programação dinâmica e em quais problemas ela é aplicada?",
    "Quais são os diferentes tipos de linguagens de programação e suas respectivas categorias?",
    "Como os algoritmos genéticos são utilizados para resolver problemas de otimização em engenharia?",
    "O que são as métricas de avaliação de algoritmos de aprendizado de máquina e por que são importantes?",
    "Explique o conceito de criptomoeda e como funciona a tecnologia blockchain.",
    "Quais são os protocolos de roteamento mais comuns em redes de computadores e como eles operam?",
    "Como funcionam os sistemas de recomendação baseados em filtragem colaborativa e conteúdo?",
    "Quais são os desafios de privacidade associados ao uso de grandes volumes de dados pessoais?",
    "Explique o conceito de balanceamento de carga global em arquiteturas de nuvem.",
    "Como os sistemas distribuídos lidam com problemas de consistência de dados e concorrência?",
    "O que é a arquitetura de microsserviços e como ela facilita a escalabilidade e manutenção de sistemas?",
    "Quais são os diferentes tipos de algoritmos de aprendizado supervisionado e suas aplicações?",
    "Como funcionam os sistemas de gerenciamento de banco de dados distribuídos e quais são seus desafios?",
    "O que é o algoritmo de árvore B e como ele é utilizado para operações eficientes em bancos de dados?",
    "Explique o conceito de computação em memória e suas implicações no desempenho de sistemas.",
    "Quais são os métodos de autenticação de dois fatores e como eles aumentam a segurança de contas online?",
    "Como os algoritmos de roteamento de pacotes são utilizados na internet para entregar dados aos seus destinos?",
    "O que são algoritmos de aprendizado por transferência e em que cenários são aplicados?",
    "Quais são os desafios de segurança associados à inteligência artificial e como podem ser mitigados?",
]

# Entrada da consulta
query = input("O que deseja consultar no banco de questões?\n")

# Primeiro passo: Tokenizar e processar o texto
nltk.download('punkt')
from nltk.tokenize import word_tokenize
tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
tokenized_query = word_tokenize(query.lower())

palavras_not_stopwords = [] # Criação de uma lista para armazenar os documentos da base sem "stopwords"
#palavras_not_rad = [] # Criação de uma lista para armazenar os documentos da base com apenas os radicais

for k in range(0, len(tokenized_query)): 
        if(tokenized_query[k] not in stopwords): # Se a palavra ou caracter da posição k na lista paluvras não for "stopwords" (variável criada)
            palavras_not_stopwords.append(tokenized_query[k]) # Utilizo a função append para acrescentar a "palavra" dentro da minha lista de palavras sem stopwords

#print("\n",palavras_not_stopwords,"\n")

# Utilizando o WordNetLemmatizer para obter os radicais das palavras
"""lemmatizer = nltk.stem.WordNetLemmatizer()
for k in range(0, len(palavras_not_stopwords)):
    palavras_not_rad.append(lemmatizer.lemmatize(palavras_not_stopwords[k]))


for k in range(0, len(palavras_not_stopwords)):
        stemmer = nltk.stem.RSLPStemmer() # Criação de uma variável que permite que eu possa tirar os radicais das palavras
        palavras_not_rad.append(stemmer.stem(palavras_not_stopwords[k])) # Colocando as palavras sem as stopwords e com seus radicais extraídos na lista palavras_not_rad

print("\n",palavras_not_rad,"\n")"""

# Segundo passo: Calcular o TF-IDF (Term Frequency — Inverse Document Frequency)
# Converter documentos tokenizados em texto 
preprocessed_documents = [' '.join(doc) for doc in tokenized_documents]
preprocessed_query = ' '.join(palavras_not_stopwords)

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

