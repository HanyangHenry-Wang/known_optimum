import known_optimum
from known_optimum import Objective_Function2
from known_optimum import normal_BO, my_new_BO, my_new_BO_full_search
import numpy as np

# regret_holder1 = []
# regret_holder2 = []
# regret_holder3 = []
# regret_holder4 = []
regret_holder5 = []

N = 200

for i in range(N):
    
  X_total, Y_total = Objective_Function2(variance=10,lengthscale=1.2,seed=i)
  fstar = np.max(Y_total)
  print('optimal: ',fstar)

  # regret_temp1 = normal_BO(X_total,Y_total,'ei',fstar,seed = i)
  # regret_holder1.append(regret_temp1)

  # regret_temp2 = normal_BO(X_total,Y_total,'new ei',fstar,seed = i)
  # regret_holder2.append(regret_temp2)

  # regret_temp3 = my_new_BO(X_total,Y_total,'ei',fstar,seed = i)
  # regret_holder3.append(regret_temp3)

  # regret_temp4 = normal_BO(X_total,Y_total,'mse',fstar,seed = i)
  # regret_holder4.append(regret_temp4)
  
  regret_temp5 = my_new_BO_full_search(X_total,Y_total,'find_max',fstar,seed = i)
  regret_holder5.append(regret_temp5)
  
  
# regret_holder1 = np.array(regret_holder1)
# regret_holder2 = np.array(regret_holder2)
# regret_holder3 = np.array(regret_holder3)
# regret_holder4 = np.array(regret_holder4)
regret_holder5 = np.array(regret_holder5)
  
  
# np.savetxt('GP2d_ei', regret_holder1, delimiter=',')
# np.savetxt('GP2d_tei', regret_holder2, delimiter=',')
# np.savetxt('GP2d_po', regret_holder3, delimiter=',')
# np.savetxt('GP2d_mes', regret_holder4, delimiter=',')
np.savetxt('GP2d_po_full', regret_holder5, delimiter=',')