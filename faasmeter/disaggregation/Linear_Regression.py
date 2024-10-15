import numpy as np 

from sklearn import svm

import warnings

class LinearModel:
    
    def __init__(self, name, history_size):
        self.name = name
        self.hsize = history_size
        self.gen()
        self.hx = None
        self.hz = None
            
    def gen(self):
        if self.name == 'SVR':
            self.m = svm.SVR(kernel='linear', C=37.0, epsilon=2.0, max_iter=4000)
            
    def train(self, x, y):                    
        warnings.filterwarnings('ignore')
        self.m.fit(x, y)
        warnings.filterwarnings('default')
    
    def train_on_history(self):
        warnings.filterwarnings('ignore')
        self.train(self.hx, self.hy)
        warnings.filterwarnings('default')
        
    def add_history( self, x, y):
        
        if self.hx is None:
            self.hx = x
            self.hy = y   
        else:
            self.hx = np.concatenate([self.hx,[x]], axis=0)
            self.hx = self.hx[1:]
            self.hy = np.concatenate([self.hy,[y]], axis=0)
            self.hy = self.hy[1:]
        
    def predict(self, x):
        warnings.filterwarnings('ignore')
        y = self.m.predict(x)
        warnings.filterwarnings('default')
        return y

