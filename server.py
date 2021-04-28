#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:08:06 2021

@author: krishna
"""
import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow import keras
import time


#loading the dataset
fashion_mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] /255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

x_data, y_data=get_non_iid_data(x_data_temp,y_data_temp,5)


import dataset_divider
#getting iid data
x_data, y_data=dataset_divider.divide_without_label(5,X_train, y_train)

#getting non-iid data
x_data_temp, y_data_temp=dataset_divider.divide_with_label(5,X_train, y_train)
x_data,y_data=dataset_divider.get_non_iid_data(x_data_temp,y_data_temp,5)


#function for averaging
def model_average(client_weights):
    average_weight_list=[]
    for index1 in range(len(client_weights[0])):
        layer_weights=[]
        for index2 in range(len(client_weights)):
            weights=client_weights[index2][index1]
            layer_weights.append(weights)
        average_weight=np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_list.append(average_weight)
    return average_weight_list
            

def create_model():
    model=keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28,28]),
        keras.layers.Dense(200,activation='relu'),
        keras.layers.Dense(200,activation='relu'),
        keras.layers.Dense(10,activation='softmax')
    ])
    
    weight=model.get_weights()
    return weight
    
def evaluate_model(accuracy_list,weight):
    model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(200,activation='relu'),
    keras.layers.Dense(200,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])
    
    model.set_weights(weight)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.1),metrics=['accuracy'])
    result=model.evaluate(X_valid, y_valid)
    
    if len(accuracy_list)==0:
        accuracy_list.append(0)
        if result[1] > accuracy_list[len(accuracy_list)-1]:
            return True
        
    elif result[1] > accuracy_list[len(accuracy_list)-1]:
            return True
    else:
        return False
    
    
#initializing the client automatically
from client import Client
def train_server(training_rounds,epoch,learning_rate):
    accuracy_list=[]
    client_weight_for_sending=[]
    
    for index1 in range(1,training_rounds):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged=[]
        for index in range(len(y_data)):
            print('-------Client-------', index)
            if index1==1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight=create_model()
                client=Client(x_data[index],y_data[index],epoch,learning_rate,initial_weight)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
            else:
                client=Client(x_data[index],y_data[index],epoch,learning_rate,client_weight_for_sending[index1-2])
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
    
        #calculating the avearge weight from all the clients
        client_average_weight=model_average(client_weights_tobe_averaged)
        client_weight_for_sending.append(client_average_weight)


        #validating the model with avearge weight
        model=keras.models.Sequential([
                keras.layers.Flatten(input_shape=[28,28]),
                keras.layers.Dense(200,activation='relu'),
                keras.layers.Dense(200,activation='relu'),
                keras.layers.Dense(10,activation='softmax')
            ])

        model.set_weights(client_average_weight)
        model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.1),metrics=['accuracy'])
        result=model.evaluate(X_valid, y_valid)
        accuracy=result[1]
        print('#######-----Acccuracy for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list.append(accuracy)
    return accuracy_list

def train_server_weight_discard(training_rounds,epoch,learning_rate):
    accuracy_list=[]
    client_weight_for_sending=[]
    
    for index1 in range(1,200):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged=[]
        for index in range(len(y_data)):
            print('-------Client-------', index)
            if index1==1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight=create_model()
                client=Client(x_data[index],y_data[index],epoch,learning_rate,initial_weight)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
            else:
                client=Client(x_data[index],y_data[index],epoch,learning_rate,client_weight_for_sending[index1-2])
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
        
        #calculating the avearge weight from all the clients
        client_average_weight=model_average(client_weights_tobe_averaged)
        if evaluate_model(accuracy_list,client_average_weight)==True:
            client_weight_for_sending.append(client_average_weight)
            
            #validating the model with avearge weight
            model=keras.models.Sequential([
            keras.layers.Flatten(input_shape=[28,28]),
            keras.layers.Dense(200,activation='relu'),
            keras.layers.Dense(200,activation='relu'),
            keras.layers.Dense(10,activation='softmax')
                ])
            
            model.set_weights(client_average_weight)
            model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.1),metrics=['accuracy'])
            result=model.evaluate(X_valid, y_valid)
            accuracy=result[1]
            print('#######-----Acccuracy for round ', index1, 'is ', accuracy, ' ------########')
            accuracy_list.append(accuracy)
            
        else:
            print('Weight discarded due to low accuarcy')
            client_weight_for_sending.append(client_weight_for_sending[len(client_weight_for_sending)-1])
            
    return accuracy_list
        

        
#initializng the traiing work
start=time.time()
training_accuracy=train_server(200,2,0.1)
end=time.time()
print('TOTAL TIME ELPASED = ', end-start)

#plotting the graph
plt.pyplot.plot(accuracy_list,label='train')
plt.legend()
plt.show()









