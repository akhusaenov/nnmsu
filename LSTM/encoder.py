# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:25:52 2018

@author: Xiaomi
"""

import pandas as pd
import Stemmer
import re
from collections import OrderedDict
import os
import numpy as np

class encoder_twitter:
    
    def __init__(self, dictionary = 'data/dictionary.csv'):
        
        self.stemmer = Stemmer.Stemmer('russian')
        self.dictionary = dictionary
        
    def exists(self, path):
        try:
            os.stat(path)
        except OSError:
            return 'Unsuccessfully'
        return 'Successfully'
    
    # filter only RU and pull base words
    def stem(self,data):
        #data - row from dataframe or str line
        
        # filter not only RU
        # data = re.sub("[^\w]", " ", data).split()
        data = re.findall("[а-яА-Я]+", data)
        return self.stemmer.stemWords(data)
    
    def update_dict(self, data):
        # data - numpy array 2D or list 2D
        print('Load ' + self.dictionary + ' ' + self.exists(self.dictionary))
        dictionary = pd.read_csv(self.dictionary, ';',index_col='index')
        data = pd.DataFrame(data)
        
        data = data['ttext'].map(self.stem)
        
        all_words = []
        for twitt in data:
            for word in twitt:
                all_words.append(word)
                
        upd_dict = np.append(dictionary.values,np.asarray(all_words))
        dictionary = list(OrderedDict.fromkeys(upd_dict))
        
        dct = pd.DataFrame(dictionary, columns=['words'])
        dct.index.name = 'index'
        dct.to_csv(self.dictionary, sep=';')
        return dictionary,data
    
    def decode(self, data):
        # data - numpy array 2D
        print('Load ' + self.dictionary + ' ' + self.exists(self.dictionary))
        dictionary = pd.read_csv(self.dictionary, ';',index_col='index')
        n,m = data.shape
        data = data.astype(int).astype(str)
        for i in range(n):
            for j in range(m):
                data[i][j] = dictionary.at[int(data[i][j]),'words']

        return data
    
    def encode(self, url=None, data=None, limit = 0, save = False):
        if(url == None and data == None):
            print('set one arg, url or data')
            pass
        # data - 2D array Numpy ot list
        # url - twitts file
        # limit - # limit - upper limit for dataset (more quickly train)
        # save - overwrites files after encoding
        if(url!=None):
            print('Load ' + url + ' ' + self.exists(url))
        
        if(data == None):
            data = pd.read_csv(url, sep=';')
        else:
            data = pd.DataFrame(data,columns=['ttext'])
        
        if(limit>0):
            data = data[:limit]
        
        dictionary, data = self.update_dict(data)
        
        twitt_to_number = []
        for row in data:
            lst = []
            for word in row:
                lst.append(dictionary.index(word))
            twitt_to_number.append(lst)
        # convert twitts to numbers
            
        df = (pd.DataFrame(twitt_to_number)).fillna(0)
        df.index.name = 'index'
        # save positive, negative and dictionary dataset
        if(save):
            df.to_csv(url,sep=';')
        
        return df