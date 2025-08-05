import numpy as np 

class MSELoss:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred 
        self.y_true = y_true 
        self.batch = y_pred.shape[0] 
        loss = np.mean((y_pred - y_true)**2)
        return loss 
    
    def backward(self):
        return (2 / self.batch) * (self.y_pred - self.y_true)

class CrossEntropyLoss:
    def forward(self, logits, y_true):
        self.logits = logits 
        self.y_true = y_true 
        self.batch_size = logits.shape[0] 

        exp_logits = np.exp(logits - np.max(logits, axis = 1, keepdims = True))
        self.probs = exp_logits / np.sum(exp_logits, axis =1 , keepdims = True) 

        log_probs = -np.log(self.probs[range(self.batch_size), y_true])
        loss = np.mean(log_probs)
        return loss

    def backward(self):
        grad = self.probs.copy() 
        grad[range(self.batch_size), self.y_true] -= 1 
        return grad / self.batch_size