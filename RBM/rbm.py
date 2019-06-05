# Boltzmann Machine

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data

# Importing the dataset
movies = pd.read_csv("ml-1m/movies.dat", header=None, sep='::', engine='python', encoding='latin-1')
users = pd.read_csv("ml-1m/users.dat", header=None, sep='::', engine='python', encoding='latin-1')
ratings = pd.read_csv("ml-1m/ratings.dat", header=None, sep='::', engine='python', encoding='latin-1')

# Preparing the training set and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in rows and movies in columns
# Torch expects the data to be in list of list
def convert(data):
    new_data = []
    for id_users in range(1, nb_users+1):
        id_movies = data[:, 1][data[:,0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(ratings)
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
# Tensors are simply arrays/multi-dimensional matrices that contain elements of single datatype
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings, 1(Liked) or 0(Not Liked)
# Ratings of the output should be equal to ratings of input in RBM
training_set[training_set==0] = -1
training_set[training_set==1] = 0
training_set[training_set==1] = 0                               # or operator doesnt work in pytorch
training_set[training_set>=3] = 1
test_set[test_set==0] = -1
test_set[test_set==1] = 0
test_set[test_set==1] = 0 
test_set[test_set>=3] = 1

# Creating the architecture of Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)   #100*1682              # weight matrix of p_v_given_h
        self.a = torch.randn(1, nh)    #1*100                 # bias of visible nodes
        self.b = torch.randn(1, nv)    #1*1682                # bias of hidden nodes
    def sample_h(self, x):                                    # sample the hidden nodes
        wx = torch.mm(x, self.W.t())    #100*1682 * 1682*100 = 100*100
        activation = wx + self.a.expand_as(wx)                  # adding to every line
        p_h_given_v = torch.sigmoid(activation)     #100*100
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):                                      # sample the visible nodes
        wy = torch.mm(y, self.W)    #100*100 * 100*1682 = 100*1682
        activation = wy + self.b.expand_as(wy)                  
        p_v_given_h = torch.sigmoid(activation)     #100*1682
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):                          # implementing contrastive divergence
        self.W += (torch.mm(v0.t(), ph0)- torch.mm(vk.t(), phk)).t()   
                                                    #1682*100 * 100*100 - 1682*100 * 100*100 = 1682*100.t
        self.b += torch.sum((v0-vk), 0)                             # trick used to keep proper format
        self.a += torch.sum((ph0-phk), 0)    
 
# nv = nb_movies
nv = len(training_set[0])
nh = 100
# batch_size = 1, => onilne learning
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epochs = 10
for epoch in range(1, nb_epochs+1):
    train_loss = 0
    s = 0.                                                                     # counter in float
    for id_user in range(0, nb_users - batch_size, batch_size):                # (start, end, step)
         vk = training_set[id_user:id_user+batch_size]
         v0 = training_set[id_user:id_user+batch_size]
         ph0,_ = rbm.sample_h(v0)
         for k in range(10):                                                   #contrastive divergence                  
             _,hk = rbm.sample_h(vk)
             _,vk = rbm.sample_v(hk)
             vk[v0<0] = v0[v0<0]                               # freeze visible nodes with values = -1
         phk,_ = rbm.sample_h(vk)
         rbm.train(v0, vk, ph0, phk)
         train_loss += torch.mean(torch.abs(v0[v0>=0]-vk[v0>=0]))
         s+=1.
    print('epoch: ' +str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.                                                                     
for id_user in range(nb_users):                
     v = training_set[id_user:id_user+1]            #used to activate the neurons of rbm
     vt = test_set[id_user:id_user+1]     
     if len(vt[vt>=0]) > 0:                                                          
         _,h = rbm.sample_h(v)                        #Blind walk in MCMC, Markov Chain Monte Carlo
         _,v = rbm.sample_v(h)                         
         test_loss += torch.mean(torch.abs(vt[vt>=0]-v[vt>=0]))
         s+=1.
print('Test loss: '+str(test_loss/s))
    
