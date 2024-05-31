import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import os
import pickle


# Caminho do modelo treinado 
caminho_modelo_svm = "modelo_svm.pkl"
# Caminho do vectorizer treinado (adjust as needed)
caminho_vectorizer = "vectorizer.pkl"  # Assuming you saved the vectorizer separately

# Predição de crimes usando modelo SVM treinado
def prever_tipo_crime_svm(nova_descricao, vectorizer, caminho_modelo=caminho_modelo_svm):
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Modelo não encontrado: {caminho_modelo}")

    # Carregando o modelo treinado (already included)
    with open(caminho_modelo, "rb") as f:
        model = pickle.load(f)

    # Loading the vectorizer (changed)
    with open(caminho_vectorizer, "rb") as f:
        vectorizer = pickle.load(f)  # Load the vectorizer

    nova_descricao_vetor = vectorizer.transform([nova_descricao])
    predicao = model.predict(nova_descricao_vetor)
    return predicao[0]

# Exemplo de uso

try:
    # Carregando o vectorizer treinado
    with open(caminho_vectorizer, "rb") as f:
        vectorizer = pickle.load(f)
    
    while True:
        nova_descricao = input("Digite uma nova ocorrência: ")
        tipo_crime_predito = prever_tipo_crime_svm(nova_descricao, vectorizer)
        print(f"Tipo de crime predito: {tipo_crime_predito}")

except KeyboardInterrupt:
    print("\nSaindo do programa.")