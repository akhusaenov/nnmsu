# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
# библиотека для работы с dataframe
import pandas as pd

# активационная функция (1/(e^(-x)))
def f(x):
    return sp.expit(x)

# производная активационной функции
def f1(x):
    return x * (1 - x)

# функция инициализации весов
# на вход имеет несколько аргументов:
#    inputs - количество входных узлов
#    hiddens - количество узлов 1го скрытого слоя
#    hiddens2 - количество узлов 2го скрытого слоя
#    outputs - количество узлов выходного слоя
def init_weight(inputs, hiddens, hiddens2, outputs):
#    матрица весов от входного слоя к 1му скрытому слою
#    принимает рандомные значения и имеет размерность [inputs х hiddens]
    w1 = np.random.random((inputs, hiddens))
#    матрица весов от 1го скрытого слоя к 2му скрытому слою
#    принимает рандомные значения и имеет размерность [hiddens+1 х hiddens2]
#    эта матрица имеет кол-во строк на единицу больше т.к. нужно учитывать мнимую единицу
    w2 = np.random.random((hiddens + 1, hiddens2))
#    матрица весов от 2го скрытого слоя к выходному слою
#    принимает рандомные значения и имеет размерность [hiddens2+1 х outputs]
#    эта матрица имеет кол-во строк на единицу больше т.к. нужно учитывать мнимую единицу
    w3 = np.random.random((hiddens2 + 1, outputs))
    return w1, w2, w3


# функция тренировки сети
# на вход имеет несколько аргументов:
#    inputs_list - обучающее множество (входные сигналы)
#    w1 - матрица весов от входного слоя к 1му скрытому слою
#    w2 - матрица весов от 1го скрытого слоя к 2му скрытому слою
#    w3 - матрица весов от 2го скрытого слоя к выходному слою
#    targets_list - целевое множество
#    lr - скорость обучения сети
#    error - допустимая погрешность в обучении
def train(inputs_list, w1, w2, w3, targets_list, lr, error):
    #    счетчик эпох
    era = 0
    #    глобальная ошибка
    global_error = 1
    #    список ошибок
    list_error = []
    #   �"лавный цикл обучения, повторяется пока глобальная ошибка больше погрешности
    while global_error > error:
        #        локальная ошибка
        local_error = np.array([])
        # побочный цикл, прогоняющий данные с input_list
        # функция enumerate(matrix) возвращает индекс и значение строк
        # которая сохраняется в переменные i, value
        # i - индекс строки input_list
        # value - переменная которая хранит в себе строки матрицы input_list
        for i, inputs in enumerate(inputs_list):
            # переводит листа inputs в двумерный вид (для возможности проведения операции транспонирования)
            inputs = np.array(inputs, ndmin=2)
#            targets - содержит локальный таргет для данного инпута
            targets = np.array(targets_list[i], ndmin=2)

#            прямое распространение
#            скалярное произведение строки на матрицу весов
            hidden_in = np.dot(inputs, w1)
#            применение активационной функции к вектору
            hidden_out = f(hidden_in)
#            добавление в начало вектора мнимой единицы для обучения сети
            hidden_out = np.array(np.insert(hidden_out, 0, [1]), ndmin=2)

#            скалярное произведение строки на матрицу весов
            hidden_in2 = np.dot(hidden_out, w2)
#            применение активационной функции к вектору
            hidden_out2 = f(hidden_in2)
#            добавление в начало вектора мнимой единицы
            hidden_out2 = np.array(np.insert(hidden_out2, 0, [1]), ndmin=2)

#            скалярное произведение строки на матрицу весов
            final_in = np.dot(hidden_out2, w3)
#            активационная функция выходного слоя это прямая y = x, поэтому
#            здесь значение "out" равно значение "in"
            final_out = final_in
            
            
#            вычисление ошибки выходного слоя
            output_error = targets - final_out
#            вычисление ошибки второго скрытого слоя
            hidden_error2 = np.dot(output_error, w3.T)
#            вычисление ошибки первого скрытого слоя
            hidden_error = np.dot(hidden_error2[:, 1:], w2.T)
#            добавление в список локальных ошибок текущую ошибку
            local_error = np.append(local_error, output_error)
#            обратного распространение ошибки
#            изменение матрицы весов 3 т.к. производная активационный функции (y = x)
#            y` = 1 в dW = lr*output_error*hidden_out2.T не умножается на эту производную
            w3 += lr * output_error * hidden_out2.T
#            в методе обратного распространения ошибки исключается мнимая единичка для совпадения размерностей
#            hidden_error2[:,1:] - означает весь вектор за исключением первого элемента
            w2 += lr * hidden_error2[:, 1:] * f1(hidden_out2[:, 1:]) * hidden_out.T
            w1 += lr * hidden_error[:, 1:] * f1(hidden_out[:, 1:]) * inputs.T
#        глобальная ошибка - это средняя по модуля от всех локальных ошибок
        global_error = abs(np.mean(local_error))
#        global_error = np.sqrt(((local_error) ** 2).mean())
#        эпоха увеличивается на 1
        era += 1
#        вывод в консоль текущую глобальную ошибку
        print('era=',era, 'global_error=', global_error)
#        в список ошибок добавляется глобальная ошибка
        list_error.append(global_error)
#        если при обучении количество эпох превысит порог 10000 то обучение прекратится
        if era > 10000: break
#    возвращает измененные веса, количество эпох, и список ошибок
    return w1, w2, w3, era, list_error


# функция для проверки обученной сети и вывода результата
def query(inputs_list, w1, w2, w3):
#    создаем список в котором будем хранить "outs" для тестового множества
    final_out = np.array([])
    for i, inputs in enumerate(inputs_list):
#       прямое распространение так же как и при обучении для получении "out"
        inputs = np.array(inputs, ndmin=2)

        hidden_in = np.dot(inputs, w1)
        hidden_out = f(hidden_in)
        hidden_out = np.array(np.insert(hidden_out, 0, [1]), ndmin=2)

        hidden_in2 = np.dot(hidden_out, w2)
        hidden_out2 = f(hidden_in2)
        hidden_out2 = np.array(np.insert(hidden_out2, 0, [1]), ndmin=2)

        final_in = np.dot(hidden_out2, w3)

        final_out = np.append(final_out, final_in)
#    возвращаем значение вектора "out" округленные до целого числа
    return np.around(final_out)


# считываем данные с csv с помощью библиотеки pandas
# Данные о пассажирах Титаник
# данные предоставлены Яндекс курсом
# задаем столбец по которому будет вести индексирование index_col='PassengerId'
data = pd.read_csv('titanic_data.csv', index_col='PassengerId')
# столбец Survived из data 
# .values означает что данные из dataframe конвертируются в numpy array
target_data = data['Survived'].values
# удаляем из датасета столбец Survived и конвертируем в array
# data = data.drop(columns=['Survived']).values
data = data.drop('Survived', 1).values

# составляем выборку обучающего множества из первых 600 строк датасета
inputs = data[0:600]
# добавляем столбец мнимых единичек для множества
inputs = np.c_[np.ones(600),inputs]
# составляем целевое множество
targets = target_data[0:600]

# из оставшихся 114 строк составляем тестовое множество
test = data[600:714]
test = np.c_[np.ones(114),test]
targets_test = target_data[600:714]

# скорость обучения
lr = 0.3
# допустимая погрешность обучения (** - это степень)
eps = 10**(-8)

# количество узлов в входном слое с учетом единичке
# т.е. кол-во столбцов датасета +1 мнимая единичка
input_layer = 7
# количество узлов в скрытом слое 1
hidden_layer = 9
# количество узлов в скрытом слое 2
hidden_layer2 = 4
# количество узлов в выходном слое
output_layer = 1

# инициализация весов в зависимости от количества узлов в слоях сети
w1, w2, w3 = init_weight(input_layer, hidden_layer, hidden_layer2, output_layer)

# тренировка сети
# train network
w1, w2, w3, era, lst = train(inputs, w1, w2, w3, targets, lr, eps)
# вывод количества пройденных эпох
print("Количество пройденных эпох = " + str(era))
# result_test - сохранит значение "outs"
result_test = query(test,w1,w2,w3)
# проверка совпадают ли значения targets_test с result_test
# Сумма всех совпадений, разделенная на количество выборки дает точность обучения в среднем 85%
eq = sum(result_test == targets_test)/len(test)
# вывод точности
print("Результат тестирования (в %) = " + str(eq*100))

#отрисовка побочных графиков
#plt.plot(np.arange(114),result_test,color='r')
#plt.plot(np.arange(114),targets_test,color='b')

# отрисовка графика кривой ошибки
fig = plt.figure(figsize=(20,20))
plt.plot(np.arange(era),lst)
plt.show()
