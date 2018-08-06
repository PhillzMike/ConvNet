# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 01:34:46 2018

@author: Fasipe Timilehin
"""

import numpy as np
import math
class conv_net:
    """A class for implementing any convolutional neural network of any size"""
    def __init__(self, conv_layers, pool_layers, fully_connected_layers,order):
        """parameters
        ----------------
        conv_layers : list of tuples
            it is a list containing tuples that tries to model the conv layers present in the 
            network, the tuples contain four integers,
            1. The filter size for the particular conv layer
            2. The stride for the particular conv layer
            3. The padding for the particular conv layer
            4. The number of filters
            
            The order of the tuples in the list is the order the conv layers take in the network
            
        pool_layers: list of tuples
            it is a list containing tuples that tries to model the pooling layers present in the 
            network, the tuples contain three integers,
            1. The pooling size for the particular pool layer
            2. The stride for the particular pool layer
            3. The padding for the particular pool layer
            
            The order of the tuples in the list is the order the pool layers take in the network
        fully_connected_layers: list of tuples
            1. The height of the wieght matrix
            2. The width of the weight matrix
        order: A list of string
            it is a list  of strings that contain 'Conv', 'Pool', The order of this is the order of the conv layers
            and pooling layers are combined in the nework
            
        """
        self.conv_layers = conv_layers
        self.pool_layers = pool_layers
        self.fully = fully_connected_layers
        self.order = order
        
    def get_batch(self,X,Y,n):
        num = X.shape[0]
        arr = np.arange(num)
        np.random.shuffle(arr)
        for i in range(int(math.ceil(num/n))):
            yield X[arr[n*i:n*(i+1)],:,:,:],Y[arr[n*i:n*(i+1)]]
         
    def get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        assert (H + 2 * padding - field_height) % stride == 0
        assert (W + 2 * padding - field_height) % stride == 0
        out_height = (H + 2 * padding - field_height) / stride + 1
        out_width = (W + 2 * padding - field_width) / stride + 1
        out_height = int(out_height)
        out_width = int(out_width)
        
        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    
        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    
        return (k, i, j)


    def im2col_indices(self, x, field_height, field_width, padding=1, stride=1):
        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    
        k, i, j = self.get_im2col_indices(x.shape, field_height, field_width, padding,
                                     stride)
    
        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols
            
    def pad(self,X,pad,value=0):
        return np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values = (value,))
    
    def max_pool(self,X,f,stride=1,pad = 0):
        n = X.shape[2]  #width of each image
        h = X.shape[1] #hieght of each image
        outWidth = (n-f + 2* pad)/stride + 1
        outHieght = (h-f+ 2 * pad)/stride + 1
        numOfImages = X.shape[0]           
        if(not outWidth.is_integer() or not outHieght.is_integer()): #change this method of checking if a number is an integer
            raise ValueError("A valid output won't be produced")
        
        if(pad > 0):
            X = self.pad(X,pad) 
        outWidth = int(outWidth)
        outHieght = int(outHieght)
        
        
        actReg = np.zeros((numOfImages,outHieght,outWidth,X.shape[3]))
        startRow = 0
        endRow = f
        filterDepth = X.shape[3] #since the depth of the filer should be the same as the depth of the image
        indices = np.zeros((outHieght,outWidth,filterDepth,2,numOfImages),dtype=int)
        for i in range(outHieght):
            startCol = 0
            endCol = f
            for j in range(outWidth):
                actReg[:,i,j,:] = np.max(X[:,startRow:endRow,startCol:endCol,:],axis=(1,2))
                for k in range(filterDepth):
                    p = X[:,startRow:endRow,startCol:endCol,k]
                    indices[i,j,k,:,:] = np.unravel_index(np.argmax(p.reshape(numOfImages,f*f),axis=1),(f,f))
                startCol += stride
                endCol += stride
            startRow += stride
            endRow += stride
        
        return actReg,indices
  

    def conv(self,X,f,stride=1,pad=0):
        w = X.shape[3]  
        h = X.shape[2]
        n_filters,d_filter,h_filter,w_filter = f.shape
        w_out = int((w-w_filter + 2 * pad)/stride + 1)
        h_out = int((h-h_filter + 2 * pad)/stride + 1)
        X_col = self.im2col_indices(X,h_filter,w_filter,padding=pad,stride=stride)
        W_col = f.reshape(n_filters,-1)

        out = W_col @ X_col
        
        # Reshape back from 20x500 to 5x20x10x10
        # i.e. for each of our 5 images, we have 20 results with size of 10x10
        out = out.reshape(n_filters, h_out, w_out, X.shape[0])
        return out.transpose(3, 1, 2, 0)
    
    np.transpose
      
    def d_convolve(self, upstream_grad, actReg, f,b, stride = 1, pad = 0):
        d_actReg = np.zeros(actReg.shape)
        df = np.zeros(f.shape)
        #db = np.zeros(b.shape)
        for h in range(upstream_grad.shape[1]):
            for w in range(upstream_grad.shape[2]):
                x = np.reshape(np.matmul(np.reshape(f,(f.shape[0]**2*f.shape[2],f.shape[3])),upstream_grad[:,h,w,:].T).T,(upstream_grad.shape[0],f.shape[0],f.shape[1],f.shape[2]))
                d_actReg[:,h*stride:h*stride + f.shape[0],w*stride: w*stride + f.shape[0],:] += x
                df += np.reshape(np.matmul(np.reshape(actReg[:,h*stride:h*stride + f.shape[0],w*stride: w*stride + f.shape[0],:],(actReg.shape[0],f.shape[0]**2*actReg.shape[3])).T,upstream_grad[:,h,w,:]),df.shape)
        
        db = np.sum(np.sum(upstream_grad,axis=(1,2)),axis=0)
        return (d_actReg,df,db)
    
    
    def d_up_sample(self, upstream_grad, actReg_shape, numOfInputs, positionalIndex, stride):
        d_actReg = np.zeros(actReg_shape)
        for i in range(upstream_grad.shape[1]):
            for j in range(upstream_grad.shape[2]):
                for k in range(upstream_grad.shape[3]):
                    d_actReg[range(numOfInputs),positionalIndex[i,j,k,0,:] + (i*stride) ,positionalIndex[i,j,k,1,:] + (j*stride),k:k+1] = upstream_grad[:,i,j,k:k+1]
                    
        return d_actReg
    
    def get_output(self, act_size,filter_size,stride,pad):
        out_size = (act_size - filter_size + 2 *pad)/stride + 1
        #TODO Remember to make this check for also the height, since stride for width and hieght might nbe different
        if(not out_size.is_integer()):
            raise ValueError("filter of size", filter_size, "cannot be used to convolve ",act_size," in a valid way")
        return out_size
    
    
    def fully_connected(self, X, W,b):
        return np.maximum(0,np.dot(X,W) + b)
    
    def calculate_loss(self, scores,Y,n):
        expScores = np.exp(scores)
        prob = expScores/np.sum(expScores, axis=1,keepdims=True)
        
        lossForEachImage = -np.log(prob[range(n),Y])
        return lossForEachImage
        
    def train(self,data,labels,batch_size,no_of_epochs,no_of_labels):
        conv_net = []
        params = []
        conv_count = 0
        pool_count = 0
        act_reg_size = data.shape[1]
        act_reg_channel = data.shape[3]
        w_fully = []
        b_fully = []
        for layer in self.order:
            if(layer == "Conv"):
                
                filter_size,stride,pad,no_filters = self.conv_layers[conv_count]
                #Throws an error when act_reg_size is not valid
                act_reg_size = self.get_output(act_reg_size,filter_size,stride,pad)
                params.append((np.random.randn(filter_size,filter_size,act_reg_channel,no_filters)/np.sqrt(filter_size**2*act_reg_channel/2),stride,pad))
                act_reg_channel = no_filters
                conv_count += 1
                
                conv_net.append(self.conv)
            elif(layer == "Pool"):
                
                layer.append(self.maxPool)
                filter_size, stride, pad = self.pool_layers[pool_count]
                params.append(self.pool_layers[pool_count])
                act_reg_size = self.get_output(act_reg_size, filter_size, stride, pad)
                pool_count += 1
                conv_net.append(self.max_pool)
                
            else:
                raise ValueError("Wrong type")
                
        for fully_connected in self.fully:
            w_fully.append(np.random.randn(fully_connected[0],fully_connected[1])/(np.sqrt(fully_connected[0]*fully_connected/2)))
            b_fully.append(np.zeros(1,fully_connected[1]))
            
            
        for x,y in self.getBatch(data,labels,batch_size):
            act_reg = []
            act_reg.append(x)
            layer_count = 0
            no_element_in_batch = x.shape[0]
            #basically forwward pass without the fully connected region
            for layers in conv_net:
                act_reg.append(layers(x,params[layer_count][0],params[layer_count][1],params[layer_count][2]))
            
            act_reg[len(act_reg) - 1] = np.reshape(act_reg[len(act_reg)-1],(-1,w_fully[0].shape[0]))
            act_reg.append(self.fully_connected(act_reg[len(act_reg) - 1],w_fully[0],b_fully[0]))
            
            for i in range(1,len(self.fully)):
                act_reg.append(self.fully_connected(act_reg[len(act_reg) - 1],w_fully[i],b_fully[i]))
                
            
            loss = self.calculate_loss(act_reg[len(act_reg) - 1],y,no_element_in_batch)    
            
            print(loss)   
        
            
    