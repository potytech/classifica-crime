import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os
import pickle
import re
from nltk.corpus import stopwords

# Caminho do arquivo CSV
caminho_arquivo_csv = "treino2.csv"

# Função para carregar o arquivo CSV
def carregar_dados_csv(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    data = pd.read_csv(caminho_arquivo)
    return data

# Função para pré-processamento de texto
def preprocessar_texto(texto):
    texto = texto.lower()  # Converte para minúsculas
    texto = re.sub(r'\d+', '', texto)  # Remove números
    texto = re.sub(r'\W+', ' ', texto)  # Remove caracteres especiais
    texto = ' '.join([word for word in texto.split() if word not in stopwords.words('portuguese')])  # Remove stopwords
    return texto

# Função para treinar o modelo SVM com validação cruzada e ajuste de hiperparâmetros
def treinar_modelo_svm(dados_csv):
    # Pré-processar as descrições
    dados_csv["Descrição"] = dados_csv["Descrição"].apply(preprocessar_texto)
    descricoes = dados_csv["Descrição"].tolist()
    tipos_crime = dados_csv["Tipo de Crime"].tolist()

    # Criar um pipeline com vectorizer e modelo SVM
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', LinearSVC())
    ])

    # Definir os hiperparâmetros para busca em grid
    parametros = {
        'tfidf__max_df': [0.8, 0.9, 1.0],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'svm__C': [0.1, 1, 10]
    }

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(descricoes, tipos_crime, test_size=0.3, random_state=42)

    # Fazer a busca em grid com validação cruzada
    grid_search = GridSearchCV(pipeline, parametros, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Melhor modelo
    melhor_modelo = grid_search.best_estimator_

    # Avaliar o modelo no conjunto de teste
    y_pred = melhor_modelo.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Salvar o modelo treinado
    with open("modelo_svm.pkl", "wb") as f:
        pickle.dump(melhor_modelo.named_steps['svm'], f)

    # Salvar o vectorizer treinado
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(melhor_modelo.named_steps['tfidf'], f)

    return melhor_modelo

# Exemplo de uso
dados_csv = carregar_dados_csv(caminho_arquivo_csv)
modelo_treinado = treinar_modelo_svm(dados_csv)

print("Modelo SVM e Vectorizer treinados com sucesso!")
