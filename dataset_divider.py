from tensorflow import keras
import pandas as pd
import numpy as np

def divide_without_label(parts, X_train_full,y_train_full):
    
        each_part_number=int(len(X_train_full)/parts)
        list_x_train=[]
        list_y_train=[]
        
        number_list=[]
        number_list.append(0)
        for x in range(1, parts+1):
            number_list.append(each_part_number*x)
        
        
        for x in range(0,parts):
            data_x=X_train_full[number_list[x]:number_list[x+1]]
            data_y=y_train_full[number_list[x]:number_list[x+1]]
            list_x_train.append(data_x)
            list_y_train.append(data_y)
            
        return list_x_train, list_y_train


def divide_with_label(parts, X_train_full, y_train_full):
    
    value_counts=pd.Series(y_train_full).value_counts()
    each_part_number=int(len(value_counts)/parts)
    labels=pd.Series(y_train_full).unique()

    if (len(labels)/parts)%each_part_number!=0:
        print('The entered parts is invalid. ----Closing the program----')
        
    else:
        x_train_list=[[],[],[],[],[],[],[],[],[],[]]
        y_train_list=[[],[],[],[],[],[],[],[],[],[]]
        
        for index in range(len(y_train_full)):
            for index1 in range(len(labels)):
                if y_train_full[index]==labels[index1]:
                    y_train_list[labels[index1]].append(y_train_full[index])
                    x_train_list[labels[index1]].append(X_train_full[index])
        
    
        each_part_number1=int(len(y_train_list)/parts)
    
        x_train_list1=[[],[],[],[],[],[],[],[],[],[]]
        y_train_list1=[[],[],[],[],[],[],[],[],[],[]]
        
        count=0
        for index in range(parts):
            for index1 in range(each_part_number1):
                x_train_list1[index].append(x_train_list[count])
                y_train_list1[index].append(y_train_list[count])
            
                count=count+1
        
        return x_train_list1,y_train_list1
        
def get_data(x_data,y_data,count,data_type):
    from sklearn.utils import shuffle
    if data_type=='non--iid':
        print('Non--IID Data')
        train_data_1=np.array(x_data[count])
        test_data_1=np.array(y_data[count])
        return train_data_1,test_data_1
    
    else:
        if(len(x_data[count]))!=0:
            train_data_1=np.concatenate((np.array(x_data[count][0]), np.array(x_data[count][1])), axis=0)
            test_data_1=np.concatenate((np.array(y_data[count][0]), np.array(y_data[count][1])), axis=0)
            train_data_1, test_data_1=shuffle(train_data_1,test_data_1)
    
        return train_data_1, test_data_1
    
def get_non_iid_data(x_data_temp,y_data_temp,clients):
    clients=5
    x_data=[]
    y_data=[]
    for index in range(0,clients):
        x_data_temp1,y_data_temp1=get_data(x_data_temp,y_data_temp,index,'non-iid')
        x_data.append(x_data_temp1)
        y_data.append(y_data_temp1)
    
    return x_data, y_data
    


