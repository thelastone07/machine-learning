import numpy as np 
from numpy.lib.stride_tricks import sliding_window_view


class AvgPool:
    def forward(self,X,kh,kw,stride=1):
        self.x_shape = X.shape 
        self.stride = stride 
        self.kh = kh 
        self.kw = kw 
        X_patches = sliding_window_view(X,(kh,kw),axis=(2,3)) # (N, C, H_out, W_out, kh, kw)
        X_patches = X_patches[:,:,::self.stride,::self.stride,:,:]
        X_patches = np.mean(X_patches,axis=(4,5))
        return X_patches 


    def backward(self, dout): 
        #need to convert (N,C,H_out,W_out) to (N,C,H,W)
        _,_, h_out, w_out = dout.shape 
        dx = np.zeros(self.x_shape)
        for i in range(h_out):
            for j in range(w_out):
                h_start = i*self.stride 
                w_start = j*self.stride 
                h_end = h_start + self.kh 
                w_end = w_start + self.kw 

                dx[:,:,h_start:h_end,w_start:w_end] += (dout[:,:,i:i+1,j:j+1])/ (self.kh * self.kw)

        return dx

class MaxPool:
    def forward(self,X,kh,kw,stride=1):
        self.x_shape = X.shape 
        self.kh = kh
        self.kw = kw
        self.stride = stride  
        X_patches = sliding_window_view(X,(kh,kw),axis=(2,3))
        X_patches = X_patches[:,:,::self.stride,::self.stride,:,:]
        self.mask = (X_patches == X_patches.max(axis=(4,5),keepdims=True))

        out = X_patches.max(axis=(4, 5))
        return out

    def backward(self,dout):
        _,_,h_out, w_out = dout.shape 
        dx = np.zeros(self.x_shape)

        for i in range(h_out):
            for j in range(w_out):
                h_start = i*self.stride 
                w_start = j*self.stride 
                h_end = h_start + self.kh 
                w_end = w_start + self.kw 

                mask = self.mask[:,:,i:i+1,j:j+1,:,:] #(N,C,1,1,kh,kw)

                grad = dout[:,:,i:i+1,j:j+1][:,:,:,:,None,None] #(N,C,1,1,1,1)

                dx[:,:,h_start:h_end,w_start:w_end] += (mask * grad).squeeze(2).squeeze(2)
        return dx 