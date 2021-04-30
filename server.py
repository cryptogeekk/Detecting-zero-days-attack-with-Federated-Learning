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
import pickle
from tensorflow.keras.regularizers import l1,l2

#loading the dataset ##should be in the form of X_train, y_train, X_valid,y_valid
import clean_data
X_train, y_train, X_valid,y_valid=clean_data.nsl_kdd_train_data()

import dataset_divider
#getting iid data
x_data, y_data=dataset_divider.divide_without_label(10,X_train, y_train)
    

#getting non-iid data when there are 5 clients and each client have one attack. Here we don't need 
#shuflle so dataset.divider.divide with label is sufficient
x_data, y_data=dataset_divider.divide_with_label(5,X_train, y_train)

def non_iid(clients):    
    file = open("train_x_non_iid_5_client.txt",'rb')
    x_data= pickle.load(file)
    
    file = open("train_y_non_iid_5_client.txt",'rb')
    y_data= pickle.load(file)

    y_data_list=[]
    for index in range(0,len(y_data)):
        temp=np.array(y_data[index])
        y_data_list.append(temp)
        
    y_data=y_data_list
    return x_data[:clients], y_data[:clients]

x_data, y_data=non_iid(5)

#temp work----------------
for index in range(len(y_data)):
    print(pd.Series(y_data[index]).value_counts())
    
pd.Series(y_valid).value_counts()

#deleting the 2,3,4 of test dataset

#---------------------------

#getting non-iid data when one client have multiple attack classes and the attack need to be shuffled.
x_data_temp, y_data_temp=dataset_divider.divide_with_label(4,X_train, y_train)
x_data,y_data=dataset_divider.get_non_iid_data(x_data_temp,y_data_temp,5)



#function for averaging
def get_model():
    model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[122,]),
    keras.layers.Dense(200,activation='relu',kernel_regularizer=l1(0.01)),
    keras.layers.Dense(100,activation='relu',kernel_regularizer=l1(0.01)),
    keras.layers.Dense(5,activation='softmax',kernel_regularizer=l1(0.01))
    ])
    
    return model
    

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
    model=get_model()
    weight=model.get_weights()
    return weight
    
def evaluate_model(accuracy_list,weight):
    model=get_model()  
    model.set_weights(weight)
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
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
def train_server(training_rounds,epoch,batch):
    #temp_variable
    # training_rounds=2 
    # epoch=5 
    # batch=128
    
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
                client=Client(x_data[index],y_data[index],epoch,initial_weight,batch)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
            else:
                client=Client(x_data[index],y_data[index],epoch,client_weight_for_sending[index1-2],batch)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
    
        #calculating the avearge weight from all the clients
        client_average_weight=model_average(client_weights_tobe_averaged)
        client_weight_for_sending.append(client_average_weight)


        #validating the model with avearge weight
        model=get_model()

        model.set_weights(client_average_weight)
        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        result=model.evaluate(X_valid, y_valid)
        accuracy=result[1]
        print('#######-----Acccuracy for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list.append(accuracy)
    return accuracy_list


def train_server_weight_discard(training_rounds,epoch,batch):
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
                client=Client(x_data[index],y_data[index],epoch,initial_weight,batch)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
            else:
                client=Client(x_data[index],y_data[index],epoch,client_weight_for_sending[index1-2],batch)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
        
        #calculating the avearge weight from all the clients
        client_average_weight=model_average(client_weights_tobe_averaged)
        if evaluate_model(accuracy_list,client_average_weight)==True:
            client_weight_for_sending.append(client_average_weight)
            
            #validating the model with avearge weight
            model=get_model()
            
            model.set_weights(client_average_weight)
            model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
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
training_accuracy=train_server(20,5,64)
end=time.time()
print('TOTAL TIME ELPASED = ', end-start)

#plotting the graph
plt.pyplot.plot(training_accuracy,label='train')
plt.legend()
plt.show()









