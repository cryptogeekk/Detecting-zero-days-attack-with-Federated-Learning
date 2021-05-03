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
    
def nsl_benign_data(x_data,y_data):
    from sklearn.utils import shuffle
    x_benign,y_benign=x_data[0],y_data[0]
    del x_data[0],y_data[0]
    x_benign,y_benign=shuffle(x_benign,y_benign)
    x_benign,y_benign=pd.DataFrame(x_benign),pd.DataFrame(y_benign)
    parts=4
    data_in_each_part=int(len(y_benign)/parts)
    for index in range(len(y_data)):
        print(index)
        x_temp_benign=x_benign[(index)*data_in_each_part:(index+1)*data_in_each_part]
        y_temp_benign=y_benign[(index)*data_in_each_part:(index+1)*data_in_each_part]
        x_data[index]=x_data[index].append(x_temp_benign,ignore_index=True)
        y_data[index]=np.concatenate((y_data[index],y_temp_benign),axis=None)
        x_data[index],y_data[index]=shuffle(x_data[index],y_data[index])
        
    return x_data,y_data


def divide_with_label(parts, X_train_full, y_train_full):
    
    # #temp_variable
    # X_train_full=X_train
    # y_train_full=y_train
    # parts=5
    
    #finding the name of column
    column_name=X_train_full.columns
    
    #assigning index from 0 instead of 0 in X_train_full
    X_train_full.index=np.arange(0,len(X_train_full))
    
    value_counts=pd.Series(y_train_full).value_counts()
    each_part_number=int(len(value_counts)/parts)
    labels=pd.Series(y_train_full).unique()
    
    if (len(labels)/parts)%each_part_number!=0:
        print('The entered parts is invalid. ----Closing the program----')
        
    else:
        #creating required number of dataframe as per the number of client
        x_train_list=[]
        y_train_list=[]
        for x in range (0,parts):
            x_train_list.append(pd.DataFrame(columns=column_name))
            y_train_list.append([])
        
        for index in range(len(y_train_full)):
            print((index/len(y_train_full))*100)
            for index1 in range(len(labels)):
                if y_train_full[index]==labels[index1]:
                    y_train_list[labels[index1]].append(y_train_full[index])
                    x_train_list[labels[index1]]=x_train_list[labels[index1]].append(X_train_full.iloc[index],ignore_index=True)

        
    return x_train_list,y_train_list
        
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
    # clients=5
    x_data=[]
    y_data=[]
    
    #temp_variables
    
    
    for index in range(0,clients):
        x_data_temp1,y_data_temp1=get_data(x_data_temp,y_data_temp,index,'non-iid')
        x_data.append(x_data_temp1)
        y_data.append(y_data_temp1)
    
    return x_data, y_data



    


