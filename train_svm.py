import pandas as pd
import numpy as np  # Import numpy library
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import os
import pickle

# Caminho do arquivo CSV
caminho_arquivo_csv = "treino.csv"

# Carregando o arquivo CSV
def carregar_dados_csv(caminho_arquivo):
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")
    data = pd.read_csv(caminho_arquivo)
    return data

# Treinando o modelo de classificação de crimes usando SVM
def treinar_modelo_svm(dados_csv):
    descricoes = dados_csv["Descrição"].tolist()
    tipos_crime = dados_csv["Tipo de Crime"].tolist()

    # Criando e ajustando o vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(descricoes)  # Ajusta o vectorizer às descrições

    # Convertendo descrições e tipos de crime em vetores numéricos
    X_train = vectorizer.transform(descricoes)
    y_train = np.array(tipos_crime)

    # Criando e treinando o modelo SVM
    model = LinearSVC()
    model.fit(X_train, y_train)

    # Salvando o modelo treinado
    with open("modelo_svm.pkl", "wb") as f:
        pickle.dump(model, f)

    # Salvando o vectorizer treinado
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return vectorizer

# Exemplo de uso
dados_csv = carregar_dados_csv(caminho_arquivo_csv)
vectorizer = treinar_modelo_svm(dados_csv)

print("Modelo SVM e Vectorizer treinados com sucesso!")
