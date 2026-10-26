import pandas as pd
import numpy as np
import spacy

print("Carregando modelo spaCy...")
nlp = spacy.load("pt_core_news_lg")


def extrair_features(texto):
    texto_str = str(texto) if pd.notnull(texto) else ""
    doc = nlp(texto_str)
    tamanho = len(doc)

    if tamanho == 0:
        return {'tamanho_doc': 0, 'frac_verb': 0, 'densidade_passiva': 0}

    vozes_passivas = sum(1 for token in doc if token.dep_ in [
                         'nsubj:pass', 'aux:pass'])
    verbos = sum(1 for token in doc if token.pos_ == 'VERB')

    return {
        'tamanho_doc': tamanho,
        'frac_verb': verbos / tamanho,
        'densidade_passiva': vozes_passivas / tamanho
    }


print("Processando treino...")
df_train = pd.read_excel('train.xlsx')
features_train = [extrair_features(txt) for txt in df_train['resp_text']]
pd.DataFrame(features_train).to_csv('metadados_treino.csv', index=False)

print("Processando teste 1...")
df_test = pd.read_excel('test1-rotulado-para-teste-estatistico.xlsx')
features_test = [extrair_features(txt) for txt in df_test['resp_text']]
pd.DataFrame(features_test).to_csv('metadados_teste1.csv', index=False)

print("Ficheiros 'metadados_treino.csv' e 'metadados_teste1.csv' gerados com sucesso.")
