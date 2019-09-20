# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:19:28 2019

@author: sirro
"""

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
from metrics import accuracy
from LogisticRegression import LogisticRegression
from os import path
  
def main():
    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "..", "..", "xor.csv"))
    data = pd.read_csv(filepath, header=0)
    
    #create third dimension and a bais to prevent division by 0
    data['x3'] = (data['x1'] * data['x2'])
    
    #Divide data into variables and target
    X = data.drop(['y'], axis=1).to_numpy()
    y = data['y'].to_numpy()
    
    color= ['red' if l == 0 else 'green' for l in y]
    
    #Plot the variables and target
    plt.figure()
    plt.scatter(X[:,0],X[:,1], c=color, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('XOR 2D Plot')
    plt.show()
    
    plt.figure()
    plt.axes(projection='3d')
    plt.scatter(X[:,0], X[:,1], X[:,2], c=color, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('XOR 3D Plot')
    plt.show()
    
    log_reg = LogisticRegression()
    log_reg.fit(X, y, eta = 1, show_curve = True)
    y_hat = log_reg.predict(X)
    
    print(f"Training Accuracy: {accuracy(y, y_hat):0.4f}")
    
    x1 = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 1000)
    x2 = -(log_reg.b + x1*log_reg.w[0])/(log_reg.w[1]+ x1*log_reg.w[2])
    
    x2 = np.ma.masked_less(x2, -0.2) 
    x2 = np.ma.masked_greater(x2, 0.2)
 
    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=color, alpha = 0.5)
    plt.plot(x1, x2, color = "#000000", linewidth = 2)
    plt.plot(x2, x1, color = "#000000", linewidth = 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('XOR 2D Plot')
    plt.show()
    
    
if __name__ == "__main__":
    main()

