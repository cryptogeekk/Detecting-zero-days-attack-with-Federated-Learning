#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:08:06 2021

@author: krishna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred']
column_names=np.array('header names')

def create_category(training_dataset):
    category_type=training_dataset['attack_type'].tolist()  #taking attack_type data from dataframe and converting it into list
    #category=['u2r','r2l','probe','dos','benign']
    
    benign=['normal']
    probe=['nmap', 'ipsweep', 'portsweep', 'satan','mscan', 'saint', 'worm']
    r2l=['ftp_write', 'guess_passwd', 'snmpguess','imap', 'spy', 'warezclient', 'warezmaster','multihop', 'phf', 'imap', 'named', 'sendmail','xlock', 'xsnoop', 'worm']
    u2r=['ps', 'buffer_overflow', 'perl', 'rootkit','loadmodule', 'xterm', 'sqlattack', 'httptunnel']
    dos=['apache2', 'back', 'mailbomb', 'processtable','snmpgetattack', 'teardrop', 'smurf', 'land','neptune', 'pod', 'udpstorm']
    
    for type in range(0,len(training_dataset)):
         if category_type[type] in probe:
             category_type[type]='probe'
         elif category_type[type] in r2l:
             category_type[type]='r2l'
         elif category_type[type] in u2r:
             category_type[type]='u2r'
         elif category_type[type] in dos:
             category_type[type]='dos'
         else:
             category_type[type]='benign'

    category_type_series=pd.Series(category_type)
    training_dataset['attack_category']=category_type_series
    return training_dataset

def nsl_kdd_train_data():
    #Reading the trainning dataset
    training_dataset=pd.read_csv("/home/krishna/Desktop/CIC AWS 2018/zero-days/Data/KDDTrain+.csv")
    training_dataset.columns=header_names  #Adding a headers to a dataframe.
    training_dataset_prepared=create_category(training_dataset)
    
    #handling the categorical columns of service,flag and protocol_type.
        #service
    train_service=training_dataset_prepared['service']
    train_service_unique=sorted(train_service.unique())
    
    service_columns=['Service_' + x for x in train_service_unique]
    
    train_service_encoded=pd.get_dummies(train_service)
    train_service_encoded=pd.DataFrame(train_service_encoded)
    train_service_encoded.columns=service_columns
    
        #flag
    train_flag=training_dataset_prepared['flag']
    train_flag_unique=sorted(train_flag.unique())
    
    flag_column=['Flag_' + x for x in train_flag_unique]
    
    train_flag_encoded=pd.get_dummies(train_flag)
    train_flag_encoded=pd.DataFrame(train_flag_encoded)
    train_flag_encoded.columns=flag_column
    
        #protocol_type
    train_protocol=training_dataset_prepared['protocol_type']
    train_protocol_unique=sorted(train_protocol.unique())
    
    protocol_columns=['Protocol_' + x for x in train_protocol_unique]
    
    train_protocol_encoded=pd.get_dummies(train_protocol)
    train_protocol_encoded=pd.DataFrame(train_protocol_encoded)
    train_protocol_encoded.columns=protocol_columns
    
    #removing the service,flag and protocol columns
    training_dataset_prepared.drop(['service','protocol_type','flag'], axis=1, inplace=True)
    
    #joining the categorical encoded attribute into main dataframe
    frames=[train_service_encoded,train_flag_encoded,train_protocol_encoded]
    training_dataset_prepared=pd.concat([training_dataset_prepared,train_service_encoded,train_flag_encoded,train_protocol_encoded], axis=1, sort=False)
    
    #handling the missing and infinite value and deleting unnecessary values
    # info=training_dataset_prepared.describe()
    training_dataset_prepared.drop(['num_outbound_cmds'], axis=1, inplace=True)     #Dropping the num_outbound coumn since it only contains 0 value.
    
    training_dataset_prepared.replace([np.inf,-np.inf],np.nan,inplace=True)                  #handling the infinite value
    training_dataset_prepared.fillna(training_dataset_prepared.mean(),inplace=True)
    
    training_dataset_prepared['attack_type'].value_counts()
    
    #Doing the feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_x=StandardScaler()
    
    
    #splitting the dataset into train set and test set
    from sklearn.model_selection import train_test_split
    train_set,test_set=train_test_split(training_dataset_prepared,test_size=0.2,random_state=42)
        #sorting the train_set and test set
    pd.DataFrame.sort_index(train_set,axis=0,ascending=True,inplace=True) 
    pd.DataFrame.sort_index(test_set,axis=0,ascending=True,inplace=True) 
    
    train_set['attack_category'].value_counts()
    training_dataset_prepared['attack_category'].value_counts()
    
    train_set.drop(['attack_type'], axis=1, inplace=True)
    test_set.drop(['attack_type'], axis=1, inplace=True)
    
    train_y=train_set['attack_category']
    train_set.drop(['attack_category'], axis=1, inplace=True)
    train_x=train_set
    
        #for test set
    
    test_y=test_set['attack_category']
    test_set.drop(['attack_category'], axis=1, inplace=True)
    test_x=test_set
    
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    train_y=le.fit_transform(np.array(train_y))
    test_y=le.fit_transform(np.array(test_y))
    
    return train_x, train_y, test_x,test_y
    



