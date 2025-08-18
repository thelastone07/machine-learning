import numpy as np

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - x * x 

class VanillaRNN:
    '''
    just a single cell of RNN block
    '''
    def __init__(self,seq_len,batch_size,input_size,hidden_size,output_size,lr=0.01,clip=5.0):
        self.len = seq_len
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr 
        self.clip = clip 
        # hidden layer do depend on batch_size

        self.h0 = np.zeros((self.batch_size,self.hidden_size))

        self.W_xh = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.W_hy = np.random.randn(self.hidden_size,self.output_size) * 0.1

        self.b_h = np.zeros((self.hidden_size,))
        self.b_y = np.zeros((self.output_size,))

        self.zero_grads()

    def zero_grads(self):
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_h = np.zeros_like(self.b_h)
        self.db_y = np.zeros_like(self.b_y)

    def forward(self, X):
        '''
        X shape : (seq_len, batch_size, input_size)
        Y : (seq_len, B, O)
        '''
        T, B, I = X.shape 
        hs = np.zeros((T+1,B,self.hidden_size))
        hs[0] = self.h0
        ys = np.zeros((T,B,self.output_size))

        for t in range(T):
            h_prev = hs[t]
            x_t = X[t]
            h_t = tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)
            y_t = h_t @ self.W_hy + self.b_y
            hs[t+1] = h_t 
            ys[t] = y_t

        cache = {"X" : X, "hs" : hs, "ys" : ys}
        self.h0 = hs[-1] 
        return ys, cache 
    
    def compute_loss(self, Y, targets):
        '''
        MSE
        '''
        diff = Y - targets 
        loss = np.mean(diff ** 2) 
        dy = (2.0 /np.prod(targets.shape)) * diff 
        return loss, dy 
    
    def backward(self, cache, dy):
        '''
        backprop through time 
        '''
        X, hs, ys = cache["X"], cache['hs'], cache['ys']
        T, B, I = X.shape 

        self.zero_grads() 

        dh_next = np.zeros((B, self.hidden_size))

        for t in reversed(range(T)):
            h_t = hs[t+1]
            h_prev = hs[t]
            x_t = X[t] 

            self.dW_hy += h_t.T @ dy[t]
            self.db_y += dy[t].sum(axis=0)

            dh = dy[t] @ self.W_hy.T + dh_next 
            # just carry the gradient across time 

            dhtanh = dh * dtanh(h_t)

            self.dW_xh += x_t.T @ dhtanh 
            self.dW_hh += h_prev.T @ dhtanh 
            self.db_h += dhtanh.sum(axis=0) 

            dh_next = dhtanh @ self.W_hh.T 

        for g in [self.dW_xh, self.dW_hh, self.dW_hy, self.db_h, self.db_y]:
            np.clip(g, -self.clip, self.clip, out=g)  

    
    def step(self):
        self.W_xh -= self.lr * self.dW_xh
        self.W_hh -= self.lr * self.dW_hh
        self.W_hy -= self.lr * self.dW_hy
        self.b_h  -= self.lr * self.db_h
        self.b_y  -= self.lr * self.db_y