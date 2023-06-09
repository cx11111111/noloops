import math

from globals import *

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def show_evaluation(net, subsequences, scaler, debug=True):
    ''' Evaluates performance of the RNN on the entire
        dataset, and shows the prediction as well as
        target values.

    Arguments:
        net (nn.Module): RNN to evaluate
        dataset (numpy.ndarray): dataset
        target (numpy.ndarray): target values for prediction,
                                original (unscaled)
        scaler (MinMaxScaler): used for denormalization
        debug (bool): should we calculate/display eval.
                      MSE/MAE
    '''
    net.eval()
    TRAIN_SPLIT=int(config.train_ratio*len(subsequences))
    VAL_SPLIT=TRAIN_SPLIT+int(config.val_ratio*len(subsequences))
    COL_NUM=int((subsequences.shape[-1]-1)/config.lag)
    # Prediction on the entire dataset
    test_set=torch.Tensor(subsequences[VAL_SPLIT:,:-1]).view(-1,config.lag,COL_NUM).to(device)
    prediction = net(test_set).unsqueeze(-1).cpu().data.numpy()
    scaling_temp=np.concatenate([subsequences[VAL_SPLIT:,-10:-1],prediction],axis=1)
    prediction = scaler.inverse_transform(scaling_temp)
    prediction=prediction[:,-1]

    test_real = torch.Tensor(subsequences[VAL_SPLIT:,-1]).to(device)
    test_real=test_real.cpu().data.numpy().reshape(-1,1)
    scaling_temp1 = np.concatenate([subsequences[VAL_SPLIT:, -10:-1],test_real], axis=1)
    test_real = scaler.inverse_transform(scaling_temp1)[:,-1]


    #prediction1=[]
    #for i in range(len(test_set)):
        #output = net(test_set[i:i+config.lag-1,:])
        #prediction1.append(output)
    #prediction1=np.array(prediction1)
    #print(prediction1.shape)
    #print(prediction1)
    #prediction1 = scaler.inverse_transform(prediction1.cpu().data.numpy())


    # Plotting the original sequence vs. predicted
    plt.plot(test_real,label='real')
    plt.plot(prediction,label='predict')
    #plt.plot(range(config.lag, len(test_set)), prediction1,label='predict1')
    plt.title('Multivariate Time-Series Forecast')
    plt.ylabel("Patv")
    plt.legend()
    plt.show()

    if debug:
        # Calculating test MSE & MAE
        test_mse = mean_squared_error(prediction, test_real)
        test_rmse=math.sqrt(test_mse)
        test_mae = mean_absolute_error(prediction, test_real)
        test_r2=r2_score(prediction,test_real)

        # Calculating test MSE & MAE
        #test_mse1= (np.square(prediction1 - test_set['Y'])).mean()
        #test_mae1 = (np.abs(prediction1 - test_set['Y'])).mean()

        print(f"Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f} | Test R2: {test_r2:.4f}")
        #print(f"Test MSE:  {test_mse1:.4f}   |  Test MAE:  {test_mae1:.4f}")


def show_loss(history):
    ''' Display train and evaluation loss

    Arguments:
        history(dict): Contains train and test loss logs
    '''
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Val loss')

    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSELoss')
    plt.legend()
    plt.show()


def display_dataset(dataset):
    ''' Displays the loaded data

    Arguments:
        dataset(numpy.ndarray): loaded data
        xlabels(numpy.ndarray): strings representing
                                 according dates
    '''
    #global x_ticks
    #global tick_positions

    # Remove information about hours (only for plotting purposes)
    #xlabels = [x[:10] for x in xlabels]
    # We can't show every date in the dataset
    # on the x axis because we couldn't see
    # any label clearly. So we extract every
    # n-th label/tick
    #segment = int(len(dataset) / 6)

    #for i, date in enumerate(xlabels):
        #if i > 0 and (i + 1) % segment == 0:
            #x_ticks.append(date)
            #tick_positions.append(i)
        #elif i == 0:
            #x_ticks.append(date)
            #tick_positions.append(i)

    # Display loaded data
    plt.plot(dataset)
    plt.title('TurbData')
    #plt.xlabel('Time')
    plt.ylabel("Patv")
    #plt.xticks(tick_positions, x_ticks, size='small')
    plt.legend()
    plt.show()
