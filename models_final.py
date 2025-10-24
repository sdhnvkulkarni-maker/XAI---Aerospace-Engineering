#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 06:19:35 2024

@author: snk
"""

# Imports required libraries 
import pandas as pd
import seaborn as sns
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
mpl.rcParams.update(mpl.rcParamsDefault)
np.random.seed(42)
torch.manual_seed(42)
mpl.rcParams['font.family'] = 'Arial'

#~~~~~~~~~~~~~~~~~~ Import data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#reads the COMPILED_AIRFOIL_DATA.csv file as df_1
df_1 = pd.read_csv('COMPILED_AIRFOIL_DATA_sk.csv', header = 0)

#renames the features in df_1 
df_1.rename(columns= {'CST Coeff 1':'c1', 'CST Coeff 2': 'c2', 'CST Coeff 3': 'c3',
                    'CST Coeff 4':'c4', 'CST Coeff 5': 'c5', 'CST Coeff 6': 'c6',
                    'CST Coeff 7':'c7', 'CST Coeff 8': 'c8',
                    'AoA':'aoa', 'Cl': 'c_l'}, inplace = True)

#creates df_2 which contains the renamed columns from df_1
df_2 = df_1[['Filename', 'aoa', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8','c_l']]

#creates df which is cleaned and has no duplicates across features. 
df = df_2.drop_duplicates(subset=['aoa', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8','c_l'])


#'Filename', 'aoa', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8',
#       'c_l', 'c_d'
# Separating out the features
features = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'aoa']


#separates out target variable
x = df[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'aoa']].values


# Separating out the target coeff_of_lift
y = df[['c_l']].values


#creates a list which stores information about models used
results = []

'''
#~~~~~~~~~~~~~~~~~~~~~ EDA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#descriptive statistics
#sns.boxplot(df[features[0:7]])

#correlation between all features in the dataset

#x_axis_labels = [r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', 
#                                   r'$c_5$', r'$c_6$', r'$c_7$', r'$c_8$', 'aoa']
#sns.heatmap(df[features].corr(), annot = False, xticklabels = x_axis_labels, 
#            yticklabels = x_axis_labels)

#~~~~~~~~~~~~~~~~Split data into Train, Val, and Test ~~~~~~~~~~~~~~~~~~~~~
##select few variables after testin for correlations
x = df[['c1', 'c3', 'c4', 'c6', 'c7','aoa']].values
#splits data into x_train, x_test, y_train, y_test wth random state = 7 and test size = 0.25
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state = 7, shuffle = True)
print("x_train.shape", x_train.shape)
print("x_test.shape", x_test.shape)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~ Models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~ 1. Linear Regression ~~~~~~~~~~~~~~~~~~~~~
from sklearn import linear_model

# initialise a LR model
LR = linear_model.LinearRegression()

#fits linear regression model to x_train and y_train
LR.fit(x_train, y_train)

#predicts for y_test and y_train using x_test and x_train

y_test_pred = LR.predict(x_test)
y_train_pred = LR.predict(x_train)

# creates data frame for train and test 
#combines the y_train and y_train_pred array
#creates columns in the data frame : y_train, y_train_pred
df_res_train = pd.DataFrame(np.concatenate((y_train, y_train_pred), axis=1),
                  columns=['y_train', 'y_train_pred'])


#combines the y_test and y_test_pred array
#creates columns in the data frame : y_test, y_test_pred
df_res_test = pd.DataFrame(np.concatenate((y_test, y_test_pred), axis=1),
                  columns=['y_test', 'y_test_pred'])

#prints accuracy metrics for the model
print('-'*25)
#print("MSE (LR) test: %.4f" % mean_squared_error(y_test, y_test_pred))
print("R2_score (LR) test: %.4f" % r2_score(y_test, y_test_pred))
#print("MSE (LR) train: %.4f" % mean_squared_error(y_train, y_test_pred))
print("R2_score (LR) train: %.4f" % r2_score(y_train, y_train_pred))


# Plot outputs
#sns.regplot(data = df_res_train, x= 'y_train', y = 'y_train_pred' ).set_title("Linear Regression")
#plt.show()
#plt.close()

#shows the line of best fit
#sns.regplot(data = df_res_test, x= 'y_test', y = 'y_test_pred' ).set_title("Linear Regression")
#plt.show()
#plt.close()

results.append({
                    'model': 'LR',
                    'data': 'train',
                    'mean_squared_error': mean_squared_error(y_train, y_train_pred),
                    'R2_score': r2_score(y_train, y_train_pred)
                })

results.append({
                    'model': 'LR',
                    'data': 'test',
                    'mean_squared_error': mean_squared_error(y_test, y_test_pred),
                    'R2_score': r2_score(y_test, y_test_pred)
                })


#~~~~~~~~~~~~~~~~~~~~~~~~~~ 2. Decision Tree Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.tree import DecisionTreeRegressor

#uses decision tree rgeressor with random state as 42
DTR = DecisionTreeRegressor(random_state = 42)

#fits model to x_train and y_train
DTR.fit(x_train, y_train)

#predicts for y_test and y_train using x_test and x_train
y_test_pred = DTR.predict(x_test)
y_train_pred = DTR.predict(x_train)

#prints accuracy metrics before hyper parameter tuning 
print('-'*25)
print('DT results before hyper parameter tuning')
print("MSE: %.4f" % mean_squared_error(y_test.flatten(), y_test_pred))
##The R2_score: 1 is perfect prediction
print("R2_score: %.4f" % r2_score(y_test.flatten(), y_test_pred))
print('-'*25)



#implementing hyperparamter tuning using GridSearchCV
# from sklearn.model_selection import GridSearchCV
# ##Hyper parameters range intialization for tuning 
# defines the parameters to tune the model on
# parameters= {"max_depth" : [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
#             "min_samples_leaf":[5, 10, 15, 20, 25, 30],"max_features":[9]}

#tunes the model, defining cross validation and the detail of the output 
# tuning_model = GridSearchCV (DTR, param_grid = parameters,
#                             scoring='neg_mean_squared_error', cv = 5, verbose = 3)

#fits the model onto x_train and y_train
# tuning_model.fit(x_train, y_train)
# print('tuning_model.best_params_', tuning_model.best_params_)

#defines the best parameters from hyperparameter tuning and implements them
DTR_best = DecisionTreeRegressor(max_depth = 13, max_features = 9, min_samples_leaf = 10,
                                 splitter= 'best', random_state = 42)

#fits bsest DTR model onto x_train and y_train
DTR_best.fit(x_train, y_train)
y_test_pred = DTR_best.predict(x_test)
y_train_pred = DTR_best.predict(x_train)



#combines the y_train and y_train_pred array
#creates columns in the data frame : y_train, y_train_pred
df_res_train = pd.DataFrame(np.concatenate((y_train, np.row_stack(y_train_pred)), axis=1),
                  columns=['y_train', 'y_train_pred'])

#combines the y_test and y_test_pred array
#creates columns in the data frame : y_train, y_train_pred
df_res_test = pd.DataFrame(np.concatenate((y_test, np.row_stack(y_test_pred)), axis=1),
                  columns=['y_test', 'y_test_pred'])

# prints accuracy metrics after hyperparameter tuning
print('-'*25)
print("MSE: %.4f" % mean_squared_error(y_test.flatten(), y_test_pred))
# The R2_score: 1 is perfect prediction
print("R2_score: %.4f" % r2_score(y_test.flatten(), y_test_pred))
print('-'*25)

# Plot outputs
#sns.regplot(data = df_res_train, x= 'y_train', y = 'y_train_pred').set_title("DT Regressor")
#plt.show()
#plt.close()

#sns.regplot(data = df_res_test, x= 'y_test', y = 'y_test_pred').set_title("DT Regressor")
#plt.show()
#plt.close()

results.append({
    'model': 'DTR',
    'data': 'train',
    'mean_squared_error': mean_squared_error(y_train, y_train_pred),
    'R2_score': r2_score(y_train, y_train_pred)
})

results.append({
    'model': 'DTR',
    'data': 'test',
    'mean_squared_error': mean_squared_error(y_test, y_test_pred),
    'R2_score': r2_score(y_test, y_test_pred)
})




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~ 3. Random Forest Regression ~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.ensemble import RandomForestRegressor

#initialises random forest regressor with random state 42
RFR = RandomForestRegressor(random_state = 42)

#converts y_train and y_test into a continugous arrray
y_train = y_train.ravel()
y_test = y_test.ravel()

#fits RFR model onto x_train and y_train
RFR.fit(x_train, y_train)

#predicts for y_test and y_train using x_test and x_train
y_test_pred = RFR.predict(x_test)
y_train_pred = RFR.predict(x_train)


#prints accuracy metrics before hyperparamter tuning
print('-'*25)
print('RFR results before hyper parameter tuning')
##The MSE
print("MSE: %.4f" % mean_squared_error(y_test.flatten(), y_test_pred))
##The R2_score: 1 is perfect prediction
print("R2_score: %.4f" % r2_score(y_test.flatten(), y_test_pred))
print('-'*25)

#implementing hyperparamter tuning using GridSearchCV
# from sklearn.model_selection import GridSearchCV
# ##Hyper parameters range intialization for tuning 
# parameters= {'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90, 100],  
#             'max_depth': [5, 10, 15], 'min_samples_leaf': [10, 15, 20, 25, 30, 35],
#             'max_features': [9],  
#             'max_leaf_nodes': [2, 4, 6, 8, 10, 12, 14, 16, 18]}
     
#tunes the model, defining cross validation and the detail of the output  
# tuning_model = GridSearchCV (RFR, param_grid = parameters,
#                             scoring='neg_mean_squared_error', cv = 5, verbose = False)

#fits tuned models onto x_train and y_train
# tuning_model.fit(x_train, y_train)
#prints best parameters
# print('tuning_model.best_params_', tuning_model.best_params_)

##best hyperparameters 
##tuning_model.best_params_ {'max_depth': 10, 'max_features': 9, 
##'max_leaf_nodes': 18, 'min_samples_leaf': 10, 'n_estimators': 40}

#uses best paramters obtained from hyperparamter tuning and applies it to RFR best
RFR_best = RandomForestRegressor( max_depth = 10, max_features = 9, 
                                  min_samples_leaf = 10, n_estimators = 40, random_state = 42)

#fits best RFR onto x_train and y_train
RFR_best.fit(x_train, y_train)
y_test_pred = RFR_best.predict(x_test)
y_train_pred = RFR_best.predict(x_train)


#combines the y_train and y_train_pred array
df_res_train = pd.DataFrame(np.concatenate((y_train.reshape(-1, 1), y_train_pred.reshape(-1, 1)),
                                           axis = 1), columns=['y_train', 'y_train_pred'])

#combines the y_test and y_test_pred array
df_res_test = pd.DataFrame(np.concatenate((y_test.reshape(-1, 1), y_test_pred.reshape(-1, 1)),
                                          axis = 1), columns=['y_test', 'y_test_pred'])

# prints accuracy metrics after hyper parameter tuning 
print('-'*25)
print('RFR results after parameter tuning')
print("MSE: %.4f" % mean_squared_error(y_test.flatten(), y_test_pred))
# The R2_score: 1 is perfect prediction
print("R2_score: %.4f" % r2_score(y_test.flatten(), y_test_pred))

# Plot outputs
#sns.regplot(data = df_res_train, x= 'y_train', y = 'y_train_pred').set_title("RFR Regressor")
#plt.show()
#plt.close()

#sns.regplot(data = df_res_test, x= 'y_test', y = 'y_test_pred').set_title("RFR Regressor")
#plt.show()
#plt.close()

results.append({
                'model': 'RFR',
                'data': 'train',
                'mean_squared_error': mean_squared_error(y_train, y_train_pred),
                'R2_score': r2_score(y_train, y_train_pred)
                })

results.append({
                'model': 'RFR',
                'data': 'test',
                'mean_squared_error': mean_squared_error(y_test, y_test_pred),
                'R2_score': r2_score(y_test, y_test_pred)
                })




#~~~~~~~~~~~~~~~~~~~~~~~~~~ 4. Adaboost Regression ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model

#sets base estimator to linear regression
LR = linear_model.LinearRegression()

#intialises Adaboostregressor
ABR = AdaBoostRegressor(random_state = 42, estimator = LR)

#converts y_train and y_test into a continugous arrray
y_train = y_train.ravel()
y_test = y_test.ravel()

#fits ABR onto x_train and y_train
ABR.fit(x_train, y_train)


#predicts for y_test and y_train using x_test and x_train
y_test_pred = ABR.predict(x_test)
y_train_pred = ABR.predict(x_train)


#prints accuracy metrics before hyerparameter tuning
print('-'*25)
print('ABR results before hyper parameter tuning')
print("MSE: %.4f" % mean_squared_error(y_test.flatten(), y_test_pred))
##The R2_score: 1 is perfect prediction
print("R2_score: %.4f" % r2_score(y_test.flatten(), y_test_pred))
print('-'*25)


##implementing hyperparamter tuning using GridSearchCV
# from sklearn.model_selection import GridSearchCV
# ##Hyper parameters range intialization for tuning 
# parameters= {'n_estimators': [3,6,9,12,15,18,21],
#             'learning_rate': [0.001,0.01,0.1,0.5,1.,2,3],
#             'loss': ['linear', 'square','exponential']}


#tunes the model, defining cross validation and the detail of the output
# tuning_model = GridSearchCV (ABR, param_grid = parameters,
#                             scoring='neg_mean_squared_error', cv = 5, verbose = False)
# tuning_model.fit(x_train, y_train)
#prints best parameters
# print('tuning_model.best_params_', tuning_model.best_params_)



#defines the base estimator
##base_estimator = LR,learning_rate = 0.01,loss = 'square',n_estimators = 15
                  

#uses best paramters obtained from hyperparamter tuning and applies it to ABR best
ABR_best = AdaBoostRegressor (estimator = LR, learning_rate = 0.01, loss = 'square',
                                n_estimators = 15, random_state = 42)

#
ABR_best.fit(x_train, y_train)
y_test_pred = ABR_best.predict(x_test)
y_train_pred = ABR_best.predict(x_train)

df_res_train = pd.DataFrame(np.concatenate((y_train.reshape(-1, 1), y_train_pred.reshape(-1, 1)),
                                           axis = 1), columns=['y_train', 'y_train_pred'])

df_res_test = pd.DataFrame(np.concatenate((y_test.reshape(-1, 1), y_test_pred.reshape(-1, 1)),
                                          axis = 1), columns=['y_test', 'y_test_pred'])

#prints ABR after hyperparameter tuning 
print('-'*25)
print('ABR results after hyper parameter tuning')
print("MSE: %.4f" % mean_squared_error(y_test.flatten(), y_test_pred))
# The R2_score: 1 is perfect prediction
print("R2_score: %.4f" % r2_score(y_test.flatten(), y_test_pred))


# Plot outputs
#sns.regplot(data = df_res_train, x= 'y_train', y = 'y_train_pred').set_title("ABR Regressor")
#plt.show()
#plt.close()

#sns.regplot(data = df_res_test, x= 'y_test', y = 'y_test_pred').set_title("ABR Regressor")
#plt.show()
#plt.close()

#appends information about model to results list
results.append({
    'model': 'ABR',
    'data': 'train',
    'mean_squared_error': mean_squared_error(y_train, y_train_pred),
    'R2_score': r2_score(y_train, y_train_pred)
})

results.append({
    'model': 'ABR',
    'data': 'test',
    'mean_squared_error': mean_squared_error(y_test, y_test_pred),
    'R2_score': r2_score(y_test, y_test_pred)
})








#~~~~~~~~~~~~~~~~~~~~~~~~~~ 5. ANN  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import copy

import torch.nn as nn
import torch.optim as optim
import tqdm
from skorch import NeuralNetRegressor
from sklearn.preprocessing import StandardScaler


#sets manual seed to 42
torch.manual_seed(42)

#splits data with test size  = 0.25
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25,random_state = 7, shuffle = True)

#standardizes x_train and z_test
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


#implements hyper parameter tuning for different ANN architectures
#~~~~~~~~~~~~~~~~~~~~ Hyper-parameter tuning ~~~~~~~~~~~~~~~~~~~~~~`
# from sklearn.model_selection import GridSearchCV
# model = nn.Sequential(
#                         nn.Linear(in_features=9, out_features= 9),
#                         nn.LeakyReLU(negative_slope=0.01, inplace=False),

#                         nn.Linear(in_features= 9, out_features= 6),
#                         nn.LeakyReLU(negative_slope=0.01, inplace=False),
#                         nn.Linear(in_features= 6, out_features= 3),
#                         nn.LeakyReLU(negative_slope=0.01, inplace=False),

#                         nn.Linear(in_features= 3, out_features= 2),
#                         nn.LeakyReLU(negative_slope=0.01, inplace=False),

#                         nn.Linear(in_features= 2, out_features= 1))
# net = NeuralNetRegressor (module = model, criterion = nn.HuberLoss(delta = 0.1),
#                             optimizer = optim.Adam, batch_size=32,max_epochs=10)
# net.fit(X_train, y_train)
# y_test_pred = net.predict(X_test)#.detach().numpy()
# y_train_pred = net.predict(X_train)#.detach().numpy()


#prints accuracy metrics of ANN
# print('-'*25)
# print('ANN results')
# ## The MSE
# print("MSE for C_l: %.4f" % mean_squared_error(y_test, y_test_pred))
# # The R2_score: 1 is perfect prediction
# print("R2_score for C_l: %.4f" % r2_score(y_test, y_test_pred))

#sets different parameters for model to tune on
# net.set_params(train_split = False, verbose = 0)
# params = {'lr': [0.01, 0.001, 0.0001],
#             'max_epochs': [50],'batch_size': [32, 64],
#             'optimizer': [optim.SGD, optim.RMSprop, 
#                           optim.Adam, optim.NAdam],
#             }


#fits best parameters on ANN
# gs = GridSearchCV (net, params, refit = False, cv = 5, scoring ='r2', verbose = 2)
# gs.fit(X_train, y_train)
# print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))

##best score: 0.961, best params: {'batch_size': 32, 'lr': 0.001, 'max_epochs': 200}
##https://machinelearningmastery.com/how-to-grid-search-hyperparameters-for-pytorch-models/
##'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta,
#                  optim.Adam, optim.Adamax, optim.NAdam],


#final model for ANN defined using best 
#~~~~~~~~~~~~~~~~~~~~ Final model ~~~~~~~~~~~~~~~~~~~~~~`
# Define the model
model_ANN = nn.Sequential(
                        nn.Linear(in_features=9, out_features= 9),
                        nn.LeakyReLU(negative_slope=0.01, inplace=False),

                        nn.Linear(in_features= 9, out_features= 6),
                        nn.LeakyReLU(negative_slope=0.01, inplace=False),

                        nn.Linear(in_features= 6, out_features= 3),
                        nn.LeakyReLU(negative_slope=0.01, inplace=False),

                        nn.Linear(in_features= 3, out_features= 2),
                        nn.LeakyReLU(negative_slope=0.01, inplace=False),
                        
                        nn.Linear(in_features= 2, out_features= 1)
                    )
loss_fn = nn.HuberLoss(delta = 0.1)
optimizer = optim.Adam(model_ANN.parameters(), lr = 0.001)#0.005
n_epochs = 100   # number of epochs to run #100
batch_size = 32  # size of each batch #32

batch_start = torch.arange(0, len(X_train), batch_size)

# Hold the best model
best_mse = np.inf   # init to infinity
#best_r2_score = 1.0

best_weights = None
history = []

#trains and implements ANN model
for epoch in range(n_epochs):
    model_ANN.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model_ANN(X_batch.to(torch.float32)) #model(X_train.to(torch.float32))
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model_ANN.eval()
    y_pred = model_ANN(X_test.to(torch.float32))
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    
    r2_s = r2_score (y_pred.detach().numpy(), y_test.detach().numpy())
    r2_s = float(r2_s)
    
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_r2_s = r2_s
        best_weights = copy.deepcopy(model_ANN.state_dict())
        
# restore model and return best accuracy
model_ANN.load_state_dict(best_weights)

#prints accuracy metrics for the best ANN model
print("MSE for best model: %.4f" % best_mse)
#print("R2_score for best model: %.4f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()
plt.close()

#converst from tensor back to numpy 
y_test_pred = model_ANN(X_test).detach().numpy()
y_train_pred = model_ANN(X_train).detach().numpy()


#pints final accuracy metrics
print('-'*25)
print('ANN results')
# The MSE
print("MSE for C_l: %.4f" % mean_squared_error(y_test, y_test_pred))
# The R2_score: 1 is perfect prediction
print("R2_score for C_l: %.4f" % r2_score(y_test, y_test_pred))

# Plot outputs
df_res_train = pd.DataFrame(np.concatenate((y_train.reshape(-1, 1), y_train_pred.reshape(-1, 1)),
                                           axis = 1), columns=['y_train', 'y_train_pred'])

df_res_test = pd.DataFrame(np.concatenate((y_test.reshape(-1, 1), y_test_pred.reshape(-1, 1)),
                                          axis = 1), columns=['y_test', 'y_test_pred'])


#sns.regplot(data = df_res_train, x= 'y_train', y = 'y_train_pred').set_title("ANN")
#plt.show()
#plt.close()

#sns.regplot(data = df_res_test, x= 'y_test', y = 'y_test_pred').set_title("ANN")
#plt.show()
#plt.close()


results.append({
                'model': 'ANN',
                'data': 'train',
                'mean_squared_error': mean_squared_error(y_train, y_train_pred),
                'R2_score': r2_score(y_train, y_train_pred)
                })

results.append({
                'model': 'ANN',
                'data': 'test',
                'mean_squared_error': mean_squared_error(y_test, y_test_pred),
                'R2_score': r2_score(y_test, y_test_pred)
                })

results.append({
                'model': 'ANN (Baseline)',
                'data': 'train',
                'mean_squared_error': np.float32 (0.0071),
                'R2_score': np.float32 (0.9560)
                })

results.append({
                'model': 'ANN (Baseline)',
                'data': 'test',
                'mean_squared_error': np.float32 (0.0071),
                'R2_score': np.float32 (0.9566)
                })


'''



#~~~~~~~~~~~~~~~~~~~~~ KAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#https://medium.com/@rubenszimbres/kolmogorov-arnold-networks-a-critique-2b37fea2112e
#https://kindxiaoming.github.io/pykan/API_demo/API_7_pruning.html

#from sklearn.preprocessing import StandardScaler
from kan import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#~~~~~~~~~~~~~~~~Split data into Train, Val, and Test ~~~~~~~~~~~~~~~~~~~~~
torch.use_deterministic_algorithms(True)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.25,random_state = 7, shuffle = True)


#standardizes  x_train and x_test
#sc = StandardScaler()
#x_train = sc.fit_transform(x_train)
#x_test = sc.transform(x_test)

# Convert to 2D PyTorch tensors
X_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


#intializes empty dataset

dataset = {}
dataset['train_input'] = X_train 
dataset['test_input'] = X_test
dataset['train_label'] = y_train
dataset['test_label'] = y_test 

#seperates out input data and labels 
X = dataset['train_input']
y = dataset['train_label']
#plt.scatter(X[:,0].cpu().detach().numpy(), X[:,1].cpu().detach().numpy(), c=y[:,0].cpu().detach().numpy())
print("x_train.shape", X_train .shape)
print("x_test.shape", X_test.shape)

#-----------
#initial network architecture of KAN
model_KAN = KAN(width=[9, 9, 1], grid = 6, k = 2, device = device, 
                sparse_init = False, seed = 2024, base_fun='silu')
#base_fun='identity'

# functions to train model on training and test 
def train_acc():
    return mean_squared_error( model_KAN(dataset['train_input'])[:,0].detach().numpy(),
             dataset['train_label'][:,0].detach().numpy())

def test_acc():
    return mean_squared_error( model_KAN(dataset['test_input'])[:,0].detach().numpy(),
             dataset['test_label'][:,0].detach().numpy())

# model_KAN = KAN(width=[9, 9, 1], grid = 6, k = 2, device = device, 
#                 sparse_init = False, seed = 2024, base_fun='silu')
#steps = 50
# Training MSE KAN 0.00617
# Test MSE KAN 0.006342
# Training R2_score KAN 0.961429
# Testing R2_score  KAN 0.961655

# full KAN model without pruning 
results_kan = model_KAN.fit(dataset, opt="LBFGS", steps = 50, metrics = (train_acc, test_acc), 
                    lamb = 0.0, lamb_entropy = 0.0, 
                    loss_fn = torch.nn.HuberLoss(delta = 0.1))

# prints accuracy metrics for initial KAN model
print('-'*35)
print("Training MSE KAN", np.round(results_kan['train_acc'][-1], 6))
print("Test MSE KAN", np.round(results_kan['test_acc'][-1], 6))
print("Training R2_score KAN", np.round(r2_score(
                    model_KAN(dataset['train_input'])[:,0].detach().numpy(),
                    dataset['train_label'][:,0].detach().numpy()), 6))
print("Testing R2_score  KAN", np.round(r2_score(
                    model_KAN(dataset['test_input'])[:,0].detach().numpy(),
                    dataset['test_label'][:,0].detach().numpy()), 6))
print('-'*35)



#plots full KAN model
model_KAN.plot(beta = 3, in_vars=[r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', 
                                   r'$c_5$', r'$c_6$', r'$c_7$', r'$c_8$', 'aoa'], 
                                  out_vars=[r'$C_L$'], title = 'KAN (Full)')
plt.show()
plt.close()

#model_KAN.node_scores
#model_KAN.edge_scores
#model_KAN.feature_score



#pruning of KAN model
print('~~~~~~~~~~~~Pruning~~~~~~~~~~~~~~~~~')
# Run these step by step- one at a time

# picks the 25% of features according to the threshold
node_75 = np.quantile(model_KAN.node_scores[0].detach().numpy(), 0.75)
edge_75 = np.quantile(model_KAN.edge_scores[0].detach().numpy(), 0.75)
print('node_75=', np.round(node_75, 4))
print('edge_75=', np.round(edge_75, 4))


#threshold for nodes and edges 
#node_75= 0.2537
#edge_75= 0.0426

#threshold for 90% for nodes and edges 
#node_90= 0.5362
#edge_90= 0.0784

#prunes the model on the respective threshold
mkp = model_KAN.prune(node_th = 0.2537, edge_th = 0.0426)

#fits the pruned KAN model, re-running the process
results_mkp = mkp.fit(dataset, opt="LBFGS", steps = 50, metrics = (train_acc, test_acc), 
                    lamb = 0.00, lamb_entropy = 0.0, 
                    loss_fn = torch.nn.HuberLoss(delta = 0.1))

#prints the accuracy metrics for pruned KAN model
print('-'*35)
print("Training MSE pruned KAN", np.round(results_mkp['train_acc'][-1], 6))
print("Test MSE pruned KAN", np.round(results_mkp['test_acc'][-1], 6))
print("Training R2_score pruned KAN", np.round(r2_score(mkp(dataset['train_input'])[:,0].detach().numpy(),
         dataset['train_label'][:,0].detach().numpy()), 6))
print("Testing R2_score pruned KAN", np.round(r2_score(mkp(dataset['test_input'])[:,0].detach().numpy(),
         dataset['test_label'][:,0].detach().numpy()), 6))
print('-'*35)

#mkp.plot(beta = 5, in_vars=['C', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'aoa'], 
#               out_vars=['C_l'], title = 'KAN (Pruned)')

mkp.plot(beta = 5, in_vars=[r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', 
                                   r'$c_5$', r'$c_6$', r'$c_7$', r'$c_8$', 'aoa'], 
                                  out_vars=[r'$C_L$'], title = 'KAN (Pruned)')

plt.show()
plt.close()

#mkp.feature_score
#~~~~~~~~~~~ symbolic formula ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#https://kindxiaoming.github.io/pykan/Example/Example_5_special_functions.html

#lib contains a list of mathematical activation functions
lib = ['x', 'x^2', 'x^3', 'x^4', 'x^5', '1/x', '1/x^2', '1/x^3', '1/x^4',
       '1/x^5', 'sqrt', 'x^1.5', '1/sqrt(x)', 'exp', 
       'log', 'abs', 'sin', 'cos', 'tan', 'tanh', 'sgn','Logit' ,'arcsin', 'arccos', 
       'arctan', 'arctanh', '0', 'gaussian']
#SYMBOLIC_LIB.keys()
add_symbolic('Logit', torch.special.logit, c=3)

mkp.auto_symbolic(lib=lib, verbose=4)

#KAN suggests equation which link features to the output 
#mkp.suggest_symbolic(0,0,0, lib=lib, a_range=( x_train[:,0].min(), x_train[:,0].max()))


# for prune =75
# manual fix
mkp.fix_symbolic(0,0,0, 'sin')
mkp.fix_symbolic(0,1,0, 'cos') # sin
mkp.fix_symbolic(0,2,0, 'sqrt')
mkp.fix_symbolic(0,3,0, 'cos') # sin
mkp.fix_symbolic(0,4,0, 'cos')
mkp.fix_symbolic(0,5,0, 'sin')
mkp.fix_symbolic(0,6,0, 'sin') # sin
mkp.fix_symbolic(0,7,0, 'cos') # sin
mkp.fix_symbolic(0,8,0, 'x') # known by the physics of the problem
mkp.fix_symbolic(1,0,0, 'sin') # sin


results_mkp_2 = mkp.fit(dataset, opt="LBFGS", steps = 50, metrics = (train_acc, test_acc), 
                    lamb = 0.00, lamb_entropy = 0.0, 
                    loss_fn = torch.nn.HuberLoss(delta = 0.1))
print('-'*35)
print("Training MSE KAN", np.round(results_mkp_2['train_acc'][-1], 6))
print("Test MSE KAN", np.round(results_mkp_2['test_acc'][-1], 6))
print("Training R2_score KAN", np.round(r2_score(mkp(dataset['train_input'])[:,0].detach().numpy(),
          dataset['train_label'][:,0].detach().numpy()), 6))
print("Testing R2_score  KAN", np.round(r2_score(mkp(dataset['test_input'])[:,0].detach().numpy(),
          dataset['test_label'][:,0].detach().numpy()), 6))
print('-'*35)

mkp.symbolic_formula()
mkp.plot(beta = 5, in_vars=[r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$', 
                                   r'$c_5$', r'$c_6$', r'$c_7$', r'$c_8$', 'aoa'], 
                                  out_vars=[r'$C_L$'], title = 'KAN (Pruned + Symbolified)')
plt.show()
plt.close()

'''













'''

results.append({
                'model': 'KAN',
                'data': 'train',
                'mean_squared_error': np.round(results_kan['train_acc'][-1], 6),
                'R2_score': np.round(r2_score(model_KAN(dataset['train_input'])[:,0].detach().numpy(),
                         dataset['train_label'][:,0].detach().numpy()), 6)
                })

results.append({
                'model': 'KAN',
                'data': 'test',
                'mean_squared_error': np.round(results_kan['test_acc'][-1], 6),
                'R2_score': np.round(r2_score(model_KAN(dataset['test_input'])[:,0].detach().numpy(),
                         dataset['test_label'][:,0].detach().numpy()), 6)
                })

df_results = pd.DataFrame(results).sort_values('R2_score')

print(df_results)
df_results.to_excel("results.xlsx", sheet_name='Sheet_name_1')

#~~~~~~~~~~~~~~~~~~~~~~~~~ Analysis of results ~~~~~~~~~~~~~~~~~~~~~~~~~
df_results = pd.DataFrame(results).sort_values('R2_score')
print(df_results)


#Visualise
sns.barplot(data = df_results [df_results['data']== 'test'] , x = 'model' , 
            y = 'R2_score', hue = 'data').set(ylim=(0.93,0.97))
#plt.tick_params (axis="x", rotation=90)

plt.show()
plt.close()

sns.barplot(data = df_results [df_results['data']== 'test'] , x = 'model' ,
            y = 'mean_squared_error', hue = 'data').set(ylim=(0.004,0.012))
plt.show()
plt.close()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
'''
#~~~~~~~~~~~~~~~ KAN hyper-parameter tuning ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
results_kan_hp = []
# Hyperparameter grid
grids = [3, 4, 5, 6, 7]
kk = [2, 3, 4, 5]
widths = [[9, 12, 1], [9, 9, 1], [9, 6, 3, 1]]


# varying hyperparameter
for g in grids :
  for ks in kk:
    for w in widths:
                model = KAN (width = w, device = device,
                                              grid = g,
                                              k = ks, seed = 0)
                model.fit(dataset, opt="LBFGS", steps = 20, metrics = (train_acc, test_acc), 
                                    lamb = l, lamb_entropy = l_e, 
                                    loss_fn = torch.nn.HuberLoss(delta = 0.1))
        
                r2_train = np.round(r2_score(model(dataset['train_input'])[:,0].detach().numpy(),
                         dataset['train_label'][:,0].detach().numpy()), 6)
        
                r2_test = np.round(r2_score(model(dataset['test_input'])[:,0].detach().numpy(),
                         dataset['test_label'][:,0].detach().numpy()), 6)
        
                results_kan_hp.append({
                                            'grid': g,
                                            'k': ks,
                                            'width': w,
                                            'R2_train': r2_train,
                                            'R2_test': r2_test,
                                        })

df_kan_hp = pd.DataFrame(results_kan_hp).sort_values('R2_test')
print(df_kan_hp)
df_kan_hp.to_excel("df_kan_hp.xlsx", sheet_name='Sheet_name_1')

'''





