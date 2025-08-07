import numpy as np 
from numpy.lib.stride_tricks import sliding_window_view

class Conv2D:
    def __init__(self, c , h, w, kernels, size:tuple, stride=1 ,n=1 ):
        self.channels = c 
        self.batch_size = n 
        self.height = h 
        self.width = w 
        self.kernels = kernels 
        self.kh, self.kw = size 
        self.stride = stride 

        self.W =  np.random.randn(self.kernels,self.channels,self.kh, self.kw)
        self.dw = np.zeros_like(self.W)

        self.b = np.zeros((self.kernels,))
        self.db = np.zeros_like(self.b)

    def im2col(self,X):
        N, C, H, W = X.shape 
        out_h = (H - self.kh) // self.stride + 1
        out_w = (W - self.kw) // self.stride + 1 

        X_patches = sliding_window_view(X,(C, self.kh, self.kw))[..., ::self.stride, ::self.stride]
        X_patches = X_patches.reshape(N, C, self.kh, self.kw, out_h, out_w)
        X_patches = X_patches.transpose(0,4,5,1,2,3) # (N, out_h, out_w, C, kh, kw)
        X_col = X_patches.reshape(N*out_h*out_w, -1).T #(C*kh*kw, N*out_h*out_w)
        return X_col, out_h, out_w
    
    def forward(self, X):
        self.X = X 
        N, C_in, H, W = X.shape
        C_out, _ , KH, KW = self.W.shape 
        #incase of im2col use this
        # W_col = self.W.reshape(self.kernels, -1) # (F, C*kw*kh)
        # X_col, out_h, out_w = self.im2col(X) 

        # Y_col = W_col @ X_col + self.b # (F, N*out_h*out_w)
        # Y = Y_col.reshape(self.kernels, N, out_h, out_w).transpose(1,0,2,3)

        X_patches = sliding_window_view(X,(KH,KW),axis=(2,3))
        H_out = (H - KH) // self.stride + 1
        W_out = (W - KW) // self.stride + 1

        self.X_patches = X_patches[:,:, ::self.stride, ::self.stride, : , : ] #(N, C_int, H_out, W_out, KH, KW)

        out = np.einsum("nchwkl,ockl->nohw",self.X_patches,self.W)
        out += self.b[None,:,None,None]
        return out 


    def backward(self, dout):
        N, C_in, H, W = self.X.shape 
        C_out, _, KH, KW = self.W.shape 
        _, _, H_out, W_out = dout.shape 

        #self.dW = self.x.T @ dout

        self.dW = np.einsum("nchwkl,nohw->ockl",self.X_patches,dout)

        #self.db = np.sum(dout, axis=0, keepdims=True)
        
        self.db = np.sum(dout,axis=(0,2,3)) # (C_out,)

        # dx = dout @ self.W.T 
        # need to reshape, first find dX_patches 

        dX_patches = np.einsum("nohw,ockl->nchwkl",dout,self.W) #(N,C_in,C_out,H-out,W_out,KH,KW)

        dx = np.zeros_like(self.X)

        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride 
                w_start = j * self.stride
                dx[:,:,h_start:h_start+KH,w_start:w_start+KW] += dX_patches[:,:,i,j,:,:]

        return dx 

         

    def get_params_and_grads(self):
        return [(self.W, self.dw),(self.b, self.db)]