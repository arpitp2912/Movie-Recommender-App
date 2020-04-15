#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from requests import get
from bs4 import BeautifulSoup


# In[ ]:


import pandas as pd
data = pd.read_csv('Data/movie_metadata.csv', engine='python')


# In[ ]:


data.columns


# In[ ]:


from tqdm import tqdm_notebook as tqdm


# In[ ]:


for link in tqdm(data.movie_imdb_link):
    desc = []
    response = get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    plot = soup.find('meta',property="og:description")
    desc.append(plot["content"])

