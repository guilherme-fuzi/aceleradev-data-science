#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[4]:


df = black_friday


# In[5]:


df.columns


# In[6]:


# Visualizando os dados
df.tail(5)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[7]:


def q1():
    return tuple(df.shape)


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[8]:


def q2():
    return len(df[(df['Gender'] == 'F') & (df['Age'] == '26-35')])


# In[9]:


len(df[(df['Gender'] == 'F') & (df['Age'] == '26-35')]['User_ID'].unique())


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[10]:


def q3():
    return len(df['User_ID'].value_counts())


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[11]:


def q4():
    return len(df.dtypes.unique())


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[30]:


total_null = df['Product_Category_3'].isnull().value_counts()[1]
total_null / df.shape[0]


# In[17]:


def q5():
    return float(total_null / df.shape[0])


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[18]:


# Verificando valores nulos nas colunas e armazenando em um dicionário
dictionary_false_values = {}
for col in df.columns:
    if len(df[col].isnull().value_counts()) > 1:
        dictionary_false_values[col] = df[col].isnull().value_counts().to_dict()

dictionary_false_values


# In[19]:


# A quantidade de valores True correspondem à quantidade de valores nulos, 
# então é evidente que a coluna com o maior número de nulos é a Product_Category_3
# abaixo o algoritmo pega o maior número da lista de valores True.
list_true = []
for value in dictionary_false_values.values():
    list_true.append(value[True])
max(list_true)


# In[20]:


def q6():
    return max(list_true)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[22]:


dict_prod_cat3 = df['Product_Category_3'].value_counts().to_dict()
dict_prod_cat3


# In[23]:


# Pegando a qtde de ocorrencias mais frequente
max_freq_value = max(dict_prod_cat3.values())

# Agora invertendo o key-value para pegar o valor mais frequente atraves da ocorrencia
aux = {v: k for k, v in dict_prod_cat3.items()}
aux


# In[24]:


# Agora conseguimos pegar o valores mais frequente através da maior ocorrência
aux[max_freq_value]


# In[25]:


def q7():
    return aux[max_freq_value]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[27]:


# Media de Purchase
df['Purchase'].mean()


# In[29]:


df2 = df.copy()


# In[32]:


min_purchase = min(df2['Purchase'])
max_purchase = max(df2['Purchase'])

df2['purchase_norm'] = (df2['Purchase'] - min_purchase) / (max_purchase - min_purchase)
df2[['Purchase', 'purchase_norm']].head(10)


# In[34]:


def q8():
    return float(df2['purchase_norm'].mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[36]:


media_purchase = df2['Purchase'].mean()
devpad_purchase = df2['Purchase'].std()

df2['pad_norm'] = (df2['Purchase'] - media_purchase) / devpad_purchase
df2[['Purchase', 'pad_norm']].head(5)


# In[38]:


df2[(df2['pad_norm'] > -1) & (df2['pad_norm'] < 1)].shape[0]


# In[40]:


def q9():
    return df2[(df2['pad_norm'] > -1) & (df2['pad_norm'] < 1)].shape[0]


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[42]:


# Primeiro peguei somente os valores nulos da categoria 2 e depois pego as duas colunas que quero comparar
df_aux = df[df['Product_Category_2'].isnull()][['Product_Category_2', 'Product_Category_3']]
df_aux.head(5)


# In[43]:


# Para responder, basta achar um contra-exemplo, ou seja, se encontrarmos um valor diferente de NaN na categoria 3
# a afirmação é Falsa, senão é verdadeira.
df_aux['Product_Category_3'].unique()

# Portanto verificamos que a afirmação é verdadeira


# In[44]:


df[['Product_Category_2', 'Product_Category_3']]


# In[45]:


def q10():
    return True

