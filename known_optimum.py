import numpy as np
import GPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randrange, uniform
from scipy.stats import norm
import math


def Objective_Function2(variance=10,lengthscale=1.2,M=40, seed=1):

  np.random.seed(seed) 
  temp = float(np.random.uniform(low=-2, high=2, size=1))

  X_total=[]


  for n in range(M*M):
    i = int(n/M)
    j = n%M
    X_total.append([i*0.15,j*0.15])


  X_total = np. array(X_total)


  X_sample=np.array([[5.4,3.8]])
  Y_sample=np.array([[temp]])

  kernel = GPy.kern.RBF(input_dim=2,variance=variance,lengthscale=lengthscale)
  m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
  m.Gaussian_noise.variance.fix(0.0)

  Y_total = m.posterior_samples_f(X_total,size=1).reshape(-1,1)
  Y_total = Y_total.reshape(M,M)

  return X_total, Y_total

def turn(ind):
  i = int(ind/40)
  j = int(ind%40)

  return i,j

def init(X_total, Y_total,seed=1):
  
  np.random.seed(seed)

  X_sample = []
  Y_sample = []

  x1_index_holder = [3,5,10,12,18,20,24,28, 30,35,35]
  x2_index_holder = [38,20,16,4,39,28,30,35,29,15]

  for i in range(10):
    x1_index = x1_index_holder[i]           #int(np.random.randint(40, size=1))
    x2_index =  x2_index_holder[i]           #int(np.random.randint(40, size=1))


    X_temp = X_total[x1_index*40+x2_index]
    Y_temp = Y_total[x1_index,x2_index]
    X_sample.append(X_temp)
    Y_sample.append(Y_temp)

  return np.array(X_sample),np.array(Y_sample)


def EI(mean,var,y_max):

  z = (mean - y_max)/np.sqrt(var)        
  out=(mean - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

  return out 

def findmax(mean,var,fstar):
   pdf_fmax = 1/(np.sqrt(2*np.pi*var))*np.exp(-(fstar-mean)**2/(2*var))

   return pdf_fmax
 

def MSE(mean,var,fstar):
    
  gamma = (fstar-mean)/np.sqrt(var)  
  
  out = (gamma*norm.pdf(gamma))/(2*norm.cdf(gamma))-np.log(norm.cdf(gamma))

  return out 

def over_max(mean,var,fstar):
  z = (fstar - mean)/np.sqrt(var)   
  below_max = norm.cdf(z)
  out = 1 - below_max
  
  return out 
    
  


def normal_BO(X_total,Y_total,acq,fstar, seed = 1): #the shape of X_total is (2500,2), the shape of Y_total is (50,50)

  fstar_true = np.max(Y_total)
  
  total_round = 10
  Y_max_holder = []

  X_sample, Y_sample = init(X_total, Y_total,seed)

  Y_max= np.max(Y_sample)
  Y_max_holder.append(Y_max)

  for n in range(total_round):
    #train the GP model for X and centrailised Y
    kernel = GPy.kern.RBF(input_dim=2,variance=10,lengthscale=1.2)
    m = GPy.models.GPRegression(X_sample,Y_sample.reshape(-1,1),kernel)
    m.Gaussian_noise.variance.fix(0.0)
    #m.optimize()

    #find the X that can maximize the acqusition function:
    mean,var = m.predict(X_total,include_likelihood=False)

    #print(mean[mean>fstar])

    if acq == 'ei':
      acq_value = EI(mean,var,Y_max_holder[-1])
    
    elif acq == 'mse':
      acq_value = MSE(mean,var,fstar)
      
    elif acq == 'over_max':
      acq_value = over_max(mean,var,fstar)
      
    elif acq == 'new ei':
      part1 = EI(mean,var,Y_max_holder[-1])
      part2 = EI(mean,var,fstar)
      acq_value = part1-part2

    index = np.argmax(acq_value)
    X_chosen = X_total[index]
    i_index,j_index = turn(index)
    Y_chosen = Y_total[i_index,j_index]

    X_sample = np.concatenate((X_sample, X_chosen.reshape(-1,2)), axis=0)
    Y_sample = np.concatenate((Y_sample, np.array([Y_chosen])), axis=0)

    Y_max= np.max(Y_sample)
    Y_max_holder.append(Y_max)


  Y_max_holder = np.array(Y_max_holder)
  regret_holder = fstar_true - Y_max_holder

  return regret_holder



def my_new_BO(X_total,Y_total,acq, fstar,seed = 1): #the shape of X_total is (2500,2), the shape of Y_total is (50,50)

  fstar_true = np.max(Y_total)
    
  total_round = 10
  Y_max_holder = []

  X_sample, Y_sample = init(X_total, Y_total,seed=seed)

  Y_max= np.max(Y_sample)
  Y_max_holder.append(Y_max)

  for n in range(total_round):
    #train the GP model for X and centrailised Y
    kernel = GPy.kern.RBF(input_dim=2,variance=10,lengthscale=1.2)
    m = GPy.models.GPRegression(X_sample,Y_sample.reshape(-1,1),kernel)
    m.Gaussian_noise.variance.fix(0.0)
    #m.optimize()

    #find the X that can maximize the acqusition function:
    mean,var = m.predict(X_total,include_likelihood=False)

    part1_total = findmax(mean,var,fstar)
    part1_total = part1_total.reshape(-1,)

    part2_total = np.zeros(X_total.shape[0])

    for i in range(X_total.shape[0]):
        X_sample_temp = np.concatenate((X_sample, np.array([X_total[i]])), axis=0)
        Y_sample_temp = np.concatenate((Y_sample, np.array([fstar])), axis=0)

        X_current_0 = X_total[i][0]
        X0_lower = max(0.0,X_current_0-10*0.15)
        X0_upper = min(5.85,X_current_0+10*0.15)
        X0_range = np.arange(X0_lower,X0_upper,0.15)

        X_current_1 = X_total[i][1]
        X1_lower = max(0.0,X_current_1-10*0.15)
        X1_upper = min(5.85,X_current_1+10*0.15)
        X1_range = np.arange(X1_lower,X1_upper,0.15)

        X_near = []

        for x0 in X0_range:
          for x1 in X1_range:
            X_near.append([x0,x1])
        X_near = np.array(X_near)



        kernel = GPy.kern.RBF(input_dim=2,variance=10,lengthscale=1.2)
        m_temp = GPy.models.GPRegression(X_sample_temp.reshape(-1,2),Y_sample_temp.reshape(-1,1),kernel)
        m_temp.Gaussian_noise.variance.fix(0.0)

        mean_temp,var_temp = m_temp.predict(X_near,include_likelihood=False)
        z = (fstar-mean_temp)/np.sqrt(var_temp)
        PnI = norm.cdf(z)
        part2 = np.min(PnI)
        part2_total[i] = part2

    acq_value = part1_total*part2_total
    index = np.argmax(acq_value)
    X_chosen = X_total[index]
    i_index,j_index = turn(index)
    Y_chosen = Y_total[i_index,j_index]

    X_sample = np.concatenate((X_sample, X_chosen.reshape(-1,2)), axis=0)
    Y_sample = np.concatenate((Y_sample, np.array([Y_chosen])), axis=0)

    Y_max= np.max(Y_sample)
    Y_max_holder.append(Y_max)


  Y_max_holder = np.array(Y_max_holder)
  regret_holder = fstar_true - Y_max_holder

  return regret_holder