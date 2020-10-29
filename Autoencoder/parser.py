# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:46:12 2018

@author: Xiaomi
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime

# Словарь хранящий в себе жанры
# работаем по прицинпу ключ - значение
GENRES = {
            'Art & Design':1,
            'Auto & Vehicles':2,
            'Beauty':3,
            'Books & Reference':4,
            'Business':5,
            'Comics':6,
            'Communication':7,
            'Dating':8,
            'Education':9,
            'Entertainment':10,
            'Events':11,
            'Finance':12,
            'Food & Drink':13,
            'Health & Fitness':14,
            'House & Home':15,
            'Libraries & Demo':16,
            'Lifestyle':17,
            'Adventure':18,
            'Arcade':19,
            'Casual':20,
            'Card':21,
            'Action':22,
            'Strategy':23,
            'Puzzle':24,
            'Sports':25,
            'Music':26,
            'Word':27,
            'Racing':28,
            'Simulation':29,
            'Board':30,
            'Trivia':31,
            'Role Playing':32,
            'Educational':33,
            'Music & Audio':34,
            'Video Players & Editors':35,
            'Medical':36,
            'Social':37,
            'Shopping':38,
            'Photography':39,
            'Travel & Local':40,
            'Tools':41,
            'Personalization':42,
            'Productivity':43,
            'Parenting':44,
            'Weather':45,
            'News & Magazines':46,
            'Maps & Navigation':47,
            'Casino':48,
            'Varies with device':49,
            'Mature 17+':50
        }
# Словарь хранящий в себе категории
CATEGORY = {
            'ART_AND_DESIGN':1,
            'AUTO_AND_VEHICLES':2,
            'BEAUTY':3,
            'BOOKS_AND_REFERENCE':4,
            'BUSINESS':5,
            'COMICS':6,
            'COMMUNICATION':7,
            'DATING':8,
            'EDUCATION':9,
            'ENTERTAINMENT':10,
            'EVENTS':11,
            'FINANCE':12,
            'FOOD_AND_DRINK':13,
            'Without Internet':14,
            'HEALTH_AND_FITNESS':15,
            'HOUSE_AND_HOME':16,
            'LIBRARIES_AND_DEMO':17,
            'LIFESTYLE':18,
            'GAME':19,
            'FAMILY':20,
            'MEDICAL':21,
            'SOCIAL':22,
            'SHOPPING':23,
            'PHOTOGRAPHY':24,
            'SPORTS':25,
            'TRAVEL_AND_LOCAL':26,
            'TOOLS':27,
            'PERSONALIZATION':28,
            'PRODUCTIVITY':29,
            'PARENTING':30,
            'WEATHER':31,
            'VIDEO_PLAYERS':32,
            'NEWS_AND_MAGAZINES':33,
            'MAPS_AND_NAVIGATION':34
            }
# Словарь хранящий в себе возрастной рейтинг
CONTENT_RATING = {
        'Everyone':1,
        'Teen':2,
        'Everyone 10+':3,
        'Mature 17+':4,
        'Adults only 18+':5,
        'Unrated':6,
        'Paid':7
        }

# очистка столбца рейтинга
def clean_rating(rating):
    # если полученный элемент имеет тип строки 
    # или кол-во символов в рейтинге не равно 3 возвращаем NaN
    if(type(rating)==str and len(rating) != 3):
        return np.nan
    # иначе меняем тип на вещ. число
    return float(rating)

# очистка столбца reviews
def clean_reviews(reviews):
    # считаем кол-во буквенных символов в reviews
    # '\D' — все, кроме цифры
    count_symb = len(re.findall('\D',reviews))
    # если reviews хранит в себе хотя бы 1 букву то возвращаем NaN
    if(count_symb>0):
        return np.nan
    #меняем тип на инт и возвращаем
    return int(reviews)

#очистка столбца size
def clean_size(size):
    # считаем кол-во буквенных символов в reviews
    count_symb = len(re.findall('\D',size))
    # если последний символ = 'M', то удаляем его и умножаем размер на 1024
    if(size[-1] == 'M'):
        size = size.replace('M','')
        # переводим тип на вещ. числа
        size = float(size)*1024
        #возвращаем размер
        return size
    # если последнмй символ 'k', то удаляем его и возвращаем число
    if(size[-1] == 'k'):
        size = size.replace('k','')
        size = float(size)
        return size
    # если ни одно из условий выше не сработало и в переменной есть буквы, то возвращаем Nan
    if(count_symb>0):
        return np.nan

#функция очистки столбца установок    
def clean_installs(value):
    # если value состоит только из символов, то возвращаем NaN
    if(value.isalpha()):
        return np.nan
    # если value содержит + и оно находится в конце (последний элемент)
    # то удаляем его
    if(value.find('+')!=-1):
        value = value.replace('+','')
    # если после удаления оставлись какие-то буквы, то возвращаем NaN
    count_symb = len(re.findall('\D',value))
    if(count_symb>0):
        return np.nan
    # возвращаем value с типом инт
    return int(value)

# очистка столбца price
def clean_price(price):
    # если цена не равно '0'
    if(price!='0'):
        # если в цене содержится символ '$' и он в конце то удаляем его и меняет тип на float
        if(price.find('$')!=-1):
            price = price.replace('$','')
            price = float(price)
        # иначе возвращаем NaN
        else:
            return np.nan
    return float(price)

# очистка стобца content_rating
def clean_content_rating(content):
    # ищем в словаре, по ключу наше числовое значение
    # пример CONTENT_RATING.get('Everyone') вернет число 1
    # второй аргумент функции .get(content,0) - означет если не найдет ничего похожего в словаре, вернет 0
    content = CONTENT_RATING.get(content,0)
    #возвращает числовое значение content
    return content

#очистка столбца update
def clean_update(date):
    #функция try: - означает попытка :-)
    try:
        #datetime.strptime() - парсит данные по шаблону
        # 1 аргумент date - стркоа в котором парсятся данные
        # 2 аргумент шаблон по которому парсятся данные
        # %B - Month as locale’s full name. (пример January, February)
        # %d - Day of the month as a zero-padded decimal number.  (01, 02, …, 31)
        # %Y - Year without century as a zero-padded decimal number. (0001, 0002, …, 2013, 2014)
        dt = datetime.strptime(date, '%B %d %Y')
        # считаем кол-во пройденных дней после обновления
        # datetime.now() возвращает дату на данный момент
        # .days возвращает кол-во дней
        date = (datetime.now() - dt).days
        # вовзвращаем дни
        return date
    #если попытка не удалась то вернет NaN
    except:
        return np.nan
# очистка столбца CATEGORY    
def clean_category(category):
    # ищем в словаре, по ключу наше числовое значение
    # пример CATEGORY.get('GAME') вернет число 19
    category = CATEGORY.get(category,0)    
    # возврващаем число
    return category

# очистка столбца жанры
def clean_genres(genres):
    # ищем в словаре, по ключу наше числовое значение
    genres = GENRES.get(genres,0)
    # возвращаем число
    return genres

# функция считает разряд числа
def digit(value):
    # масимум от value
    value = value.astype(int)
    get_max = max(value)
    #переводим в строку
    int_to_str = str(get_max)
    # разряд числа равен кол-ву символов в строке
    get_digit = len(int_to_str)
    # возвращаем число с разрядностью на 1 больше
    return 10**(get_digit + 1) #up digit

# функция нормализации датасета
def norm(data):
    # цикл который проходит по слобцам датасета
    for col in data:
        # делим столбец на число, с разрядностью на 1 больше, чем масимум данного столбца
        data[col] = data[col]/digit(data[col])
    # возврващаем нормализованный датасет
    return data

# Шаблон по которому структурируются данные, не идеальный
reg = '".+?"|[^"]+?(?=,)|(?<=,)[^"]+'
# открываем файл googleplaystore
# параметр 'r' означает только для чтения
f = open('googleplaystore.csv', 'r',encoding='utf-8')
# копируем все строки в массив
lines = f.readlines()
# закрываем файл после копирования строк
f.close()
# создаем новый файл googleplaystorenew, параметр 'w' - создает новый файл если такого нет
new_file = open('googleplaystorenew.csv', 'w',encoding='utf-8')

# result будет хранить обработнные строки
result = []
# цикл крутится по всем строкам
for line in lines:
# удаляем ';' в строках
    line = line.replace(';',' ')
    # re.findall() возвращает список данных подходящих по шаблону
    #  принимает на вход 2 аргумента
    # 1-й это шаблон, 2-й это строка в котором будет делаться поиск
    line = re.findall(reg, line)
    # объединяем получившийся список с разделитем ';'
    line = ';'.join(line)
    # удаляем все запятые из строки
    line = line.replace(',','')
    # добавляем полученную строку в список
    result.append(line)
# записываем в новый файл полученные строки
new_file.writelines(result)
# закрываем новый файл
new_file.close()
# считываем структурированные данные с помощью pandas
# параметр sep=';' указывает на разделитель данных
# error_bad_lines = False - пропускать линии которые не считываются
# это необходимо из-за того что некоторые символы в названиях имеют смайлики и пр.
data = pd.read_csv('googleplaystorenew.csv', sep=';', error_bad_lines=False)
#вывод в консоль строку № 5
print(data.iloc[5])

#Последующая чистка структурированных данных
#очистка столбцов датафрейма
# функция .map(name_function) применяет функцию ко всем строка выбранного столбца
data['Category'] = data['Category'].map(clean_category)
data['Rating'] = data['Rating'].map(clean_rating)
data['Reviews'] = data['Reviews'].map(clean_reviews)
#очистка столбца size
data['Size'] = data['Size'].map(clean_size)
# очистка столбца installs
data['Installs'] = data['Installs'].map(clean_installs)
# очистка столбца type
# if type - free, then 1, else 0
data.loc[data['Type']!='Free','Type'] = 0
data.loc[data['Type']=='Free','Type'] = 1
data['Price'] = data['Price'].map(clean_price)
data['Genres'] = data['Genres'].map(clean_genres)
data['Content Rating'] = data['Content Rating'].map(clean_content_rating)
data['Last Updated'] = data['Last Updated'].map(clean_update)
#вывод в консоль строку № 5 после чистки
print(data.iloc[5])

# отбрасываются все строки в которых пропущенны некоторые данные
data = data.dropna()
# удаляются не нужные столбцы для NN
data = data.drop(columns=['App','Android Ver','Current Ver'])
# даем название столбцу индексов
data.index.name = 'index'
#сохранение чистых данных
data.to_csv("googleplaystore_clean.csv", sep='\t', encoding='utf-8')
# нормализация данных (приведение к типу от 0 до 1)
data = norm(data)

#вывод в консоль строку № 5 после нормализации
print(data.iloc[5])
#сохранение нормализованных данных
data.to_csv("googleplaystore_norm.csv", sep='\t', encoding='utf-8')
