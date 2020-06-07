#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[2]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.
countries.dtypes


# In[6]:


countries.head()


# In[7]:


for col in countries.columns:
    if countries[col].dtypes == 'object':
        countries[col] = countries[col].str.replace(',', '.')


# In[8]:


sns.heatmap(countries.isnull(), cbar=False)


# In[9]:


for col in countries.columns:
    countries[col] = pd.to_numeric(countries[col], errors='ignore', downcast='integer')
countries.dtypes


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[25]:


def q1():
    countries['Country'] = countries['Country'].str.strip()
    countries['Region'] = countries['Region'].str.strip()
    
    return sorted(list(countries['Region'].unique()))


# In[26]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[13]:


from sklearn.preprocessing import KBinsDiscretizer


# In[14]:


# Aqui estamos dividindo nossos dados de Pop_density em 10 intervalos
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
pop_density_discret = discretizer.fit_transform(countries[['Pop_density']])
pop_density_discret[:5]


# In[15]:


# Olhando para os limites de cada um dos 10 intervalos
discretizer.bin_edges_


# In[16]:


# Para pegar o 90 percentil, i.e., o último intervalo dos 10 que dividimos, basta fazer
bin_10 = []
for i in pop_density_discret:
    bin_10.append(i == 9) 
sum(bin_10)


# In[17]:


def q2():
    return sum(bin_10)[0]


# In[18]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# No one-hot-enconding, cada valor único da coluna será um novo atributo

# In[19]:


len(countries['Region'].unique())


# In[20]:


len(countries['Climate'].unique())


# In[21]:


def q3():
    return 7 + 11


# In[22]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[38]:


def padronizar(col):
    median = col.median()
    std = col.std()
    
    return (col - median) / std


# In[39]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[40]:


test_country_df = pd.DataFrame(data=[test_country], columns=countries.columns)
test_country_df


# In[42]:


padronizar(test_country_df._get_numeric_data())


# In[27]:


def q4():
    # Retorne aqui o resultado da questão 4.
    pass


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[28]:


countries['Net_migration'].describe()


# In[29]:


net_migration_q1 = countries['Net_migration'].describe()[4]
net_migration_q3 = countries['Net_migration'].describe()[6]
net_migration_iqr = net_migration_q3 - net_migration_q1


# In[30]:


sns.boxplot(countries['Net_migration'])


# In[36]:


outliers_abaixo = net_migration_q1 - (1.5 * net_migration_iqr)
outliers_acima = net_migration_q3 + (1.5 * net_migration_iqr)

print('valores abaixo: ' + str(outliers_abaixo) + ' | valores acima: ' + str(outliers_acima))


# In[32]:


valores_abaixo_q1 = countries['Net_migration'].apply(lambda x: x < outliers_abaixo).value_counts()[1]
valores_abaixo_q1


# In[33]:


valores_acima_q3 = countries['Net_migration'].apply(lambda x: x > outliers_acima).value_counts()[1]
valores_acima_q3


# In[34]:


def q5():
    return (valores_abaixo_q1, valores_acima_q3, False)


# In[35]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[ ]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[ ]:


vectorizer = CountVectorizer()


# In[ ]:


len(newsgroup.data)


# In[ ]:


X = vectorizer.fit_transform(newsgroup.data)


# In[ ]:


palavras = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())


# In[ ]:


palavras['phone'].sum()


# In[ ]:


def q6():
    return palavras['phone'].sum()


# In[ ]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


tfid_vectorizer = TfidfVectorizer()


# In[ ]:


tfid = tfid_vectorizer.fit_transform(newsgroup.data)
tfid.toarray()


# In[ ]:


newgroup_df = pd.DataFrame(tfid.toarray(), columns=tfid_vectorizer.get_feature_names())
newgroup_df


# In[ ]:


newgroup_df['phone'].sum()


# In[ ]:


def q7():
    return round(newgroup_df['phone'].sum(), 3)


# In[ ]:


q7()


# In[ ]:




