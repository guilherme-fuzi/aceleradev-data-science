#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[4]:


# Sua análise da parte 1 começa aqui.
df = dataframe
df.head()


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[5]:


df.describe()


# In[6]:


# Quartis variável normal
q1_norm = df.normal.quantile(0.25)
q2_norm = df.normal.quantile(0.5)
q3_norm = df.normal.quantile(0.75)

print(q1_norm, q2_norm, q3_norm)

# Quartis variável binomial
q1_binom = df.binomial.quantile(0.25)
q2_binom = df.binomial.quantile(0.5)
q3_binom = df.binomial.quantile(0.75)
print(q1_binom, q2_binom, q3_binom)


# In[7]:


def q1():
    diferenca_quartil = (q1_norm - q1_binom, q2_norm - q2_binom, q3_norm - q3_binom)
    quartil_lista = []
    for dq in diferenca_quartil:
        quartil_lista.append(round(dq, 3))

    tupla_q1 = tuple(quartil_lista)
    
    return tupla_q1


# In[8]:


q1()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[9]:


normal_mean = df.normal.mean()
normal_std = df.normal.std()


interval = [normal_mean - normal_std, normal_mean + normal_std]
interval


# In[10]:


df.mean()


# In[11]:


ecdf_normal = ECDF(df.normal)


# In[12]:


probab_intervalo = ecdf_normal(interval[1]) - ecdf_normal(interval[0])
probab_intervalo


# In[13]:


def q2():
    probab_intervalo = ecdf_normal(interval[1]) - ecdf_normal(interval[0])
    return round(probab_intervalo, 3)


# In[14]:


q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[15]:


binom_mean = df.binomial.mean()
binom_var = df.binomial.var()

norm_mean = df.normal.mean()
norm_var = df.normal.var()

(float(round(binom_mean - norm_mean, 3)), float(round(binom_var - norm_var, 3)))


# In[16]:


def q3():
    binom_mean = df.binomial.mean()
    binom_var = df.binomial.var()

    norm_mean = df.normal.mean()
    norm_var = df.normal.var()
    
    return (round(binom_mean - norm_mean, 3), round(binom_var - norm_var, 3))


# In[17]:


q3()


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[18]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[19]:


# Sua análise da parte 2 começa aqui.
stars.head()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[20]:


# Filtrando onde target == 0
stars_profile = stars[stars.target == False]['mean_profile']


# In[21]:


# Padronizando mean_profile
false_pulsar_mean_profile_standardized = (stars_profile - stars_profile.mean()) / stars_profile.std()


# In[22]:


ecdf = ECDF(false_pulsar_mean_profile_standardized)


# In[23]:


q1_ppf = sct.norm.ppf(0.80) 
q2_ppf = sct.norm.ppf(0.90)
q3_ppf = sct.norm.ppf(0.95)

print(q1_ppf, q2_ppf, q3_ppf)


# In[24]:


ecdf_quartil = ecdf([q1_ppf, q2_ppf, q3_ppf])


# In[25]:


ecdf_list = []
for e in ecdf_quartil:
    ecdf_list.append(round(e, 3))
    
tupla_ecdf = tuple(ecdf_list)
tupla_ecdf


# In[26]:


def q4():
    return tupla_ecdf


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[27]:


false_pulsar_mean_profile_standardized.describe()


# In[28]:


q1_pad = false_pulsar_mean_profile_standardized.quantile(q=0.25)
q2_pad = false_pulsar_mean_profile_standardized.quantile(q=0.5)
q3_pad = false_pulsar_mean_profile_standardized.quantile(q=0.75)

print(q1_pad, q2_pad, q3_pad)


# In[29]:


n1 = sct.norm.ppf(0.25)
n2 = sct.norm.ppf(0.5)
n3 = sct.norm.ppf(0.75)

print(n1, n2, n3)


# In[30]:


(round(q1_pad - n1, 3), round(q2_pad - n2, 3), round(q3_pad - n3, 3))


# In[31]:


def q5():
    return (round(q1_pad - n1, 3), round(q2_pad - n2, 3), round(q3_pad - n3, 3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
