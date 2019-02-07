import numpy as np



# Output the entropy
def infromation_entropy(label):
    H = 0
    clas = np.unique(label)
    Num = label.shape[0]
    for j in clas:
      Numj = np.count_nonzero(label == j)
      pj = float(Numj)/float(Num)
      H -= pj* np.log2(pj)  
    return H

# Calculate the sum of entropy
def left_right_entropy(left, right):
    H_left = infromation_entropy(np.array(left))	
    H_right = infromation_entropy(np.array(right))
    Num_left = float(len(left))
    Num_right = float(len(right))
    Num = Num_left + Num_right
    H_record = Num_left/Num*H_left + Num_right/Num*H_right
    return H_left, H_right, H_record

# How to define the label of the leaf
def majority(dataset): 
    Num = dataset.shape[1]	
    label = np.unique(dataset[:,Num-1])
    record_label = 0
    record_num = 0
    for i in label:	
      num_label = np.count_nonzero(dataset[:,Num-1]==i)
      if num_label > record_label:
        record_label = i
	record_num = num_label
    return record_label 

def decision_tree_learning(dataset, depth):
    diction = {'atri':None,'value':None,'l': None,'r': None,'leaf':None}
    NumS = dataset.shape[0]
    Num = dataset.shape[1]
    label = dataset[:,Num-1]
    data = dataset[:,0:Num-1] 
    clas = np.unique(label)
    Nclas = clas.shape[0]
    H = infromation_entropy(label)
    record_right = []
    record_left = [] 
    record_atri = 0
    record_num = 0	
    record = H
    ReR = 0
    ReL	= 0	

    for i in range(0,Num-1): 
      data_use = data[:,i] 
      dataunique = np.unique(data_use)      
      for j2 in dataunique:
          left = []
          right = [] 
          left_data = []
          right_data = []
	  H_left = 0
          H_right = 0
          H_record = 0

          for k in range(0,NumS):
	    if data_use[k] <= j2:
	       left.append(label[k])
	       left_data.append(dataset[k,:])		
	    else: 	
	       right.append(label[k])
	       right_data.append(dataset[k,:])	
	  H_left, H_right, H_record = left_right_entropy(left, right)

          if H_record <= record:
            record = H_record
            ReR =  H_right
            ReL =  H_left
            record_atri = i
            record_num = j2
    	    record_right = np.array(right_data)
            record_left = np.array(left_data) 	

    
    information_gain = H - record
    diction['atri'] = record_atri
    diction['value'] = record_num
    if record > 0.:
      if ReL > 0:		
        diction['leaf'] = -1
        L_depth = depth+1
        dicleft, L_depth=decision_tree_learning(record_left, L_depth)
        diction['l'] = dicleft  
      else:
        if len(record_left) != 0:
           diction['l'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':majority(record_left)}
           diction['leaf'] = -1
           L_depth = depth+1
      if ReR > 0:
        diction['leaf'] = -1	
        R_depth = depth+1	
        dicright, R_depth=decision_tree_learning(record_right, R_depth) 
        diction['r'] = dicright 	 
      else:	
        if len(record_right) != 0:
           diction['r'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':majority(record_right)}
           diction['leaf'] = -1
           R_depth = depth+1
    else:
      diction['leaf'] = -1
      diction['l'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':majority(record_left)}
      diction['r'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':majority(record_right)}
      R_depth = depth+1
      L_depth = depth+1
    return diction, max(L_depth,R_depth)

######################## STEP 2 #####################################
def ten_fold_cross_validation(data):
  N = 10
  total_len = data.shape[0]
  one_fold = int(total_len/10)
  total_error_rate = 0
  for i in range(N):        
    test_data = data[one_fold*i:one_fold*(i+1)]
    train_data = np.concatenate((data[0: one_fold*i], data[one_fold*(i+1):]), axis = 0)
    root_node, _ = decision_tree_learning(train_data, 0) 
    error = evaluate(root_node, test_data)
    error_rate = float(error) / one_fold
    
    total_error_rate = total_error_rate + error_rate
  error_rate = float(total_error_rate)/N
  
  return 1-error_rate

def evaluate(node, split_data): 
  d = np.array(split_data)
  atri_index = node['atri']
  split_value = node['value']
  left_node = node['l']
  right_node = node['r']
  leaf = node['leaf']
  if leaf == -1:  
    left_data, right_data = extract_data(split_data, atri_index, split_value)
    e1 = evaluate(left_node, left_data)
    e2 = evaluate(right_node, right_data)
    error = e1 + e2
  else:
    count = 0
    for i in range(split_data.shape[0]):
      if int(split_data[i,split_data.shape[1]-1]) != int(leaf):
        count = count + 1
    error = count
  return error

def predict(node, data):
  predict = np.zeros((data.shape[0], 1))
  for i in range(data.shape[0]):
    predict[i] = p(node, data[i])
  return predict
'''
def p(node, d):
  atri = node['atri']
  value = node['value']
  l = node['l']
  r = node['r']
  leaf = node['leaf']
  if leaf == -1:
      if d[atri] <= value:
          return p(l,d)
      else:
          return p(r,d)
  else:
    return leaf
'''
def extract_data(data, atri_index, split_value):
  data = np.array(data)
  right_data = data[np.where(data[:,atri_index]>split_value)]
  left_data = data[np.where(data[:,atri_index]<=split_value)]
  return left_data, right_data

########################## 3 PRUNING ####################################
def pruing_check(root):

  stack = [root]
  flag = True

  while len(stack) != 0 and flag == True:
    node = stack.pop()
    if node['leaf'] == -1:      
      l = node['l']
      r = node['r']
    
      if l['leaf'] != -1 and r['leaf'] != -1:
        flag = False # need prune
      else:
        if r != None: stack.append(r)
        if l != None: stack.append(l)
  
  return flag



def dfs(node, training_data, validation_data):
  l = node['l']
  r = node['r']
  atri_index = node['atri']
  split_value = node['value']
  leaf = node['leaf']

  #new_node = None
  
  if leaf != -1:
    #new_node = node
    return node
  else:
    training_data_l, training_data_r = extract_data(training_data, atri_index, split_value)
    validation_data_l, validation_data_r = extract_data(validation_data, atri_index, split_value)

    # if find the pattern to prune
    l_leaf = l['leaf']
    r_leaf = r['leaf']
    if l_leaf != -1 and r_leaf != -1:

      error1 = error(validation_data_l, l_leaf)
      error2 = error(validation_data_r, r_leaf)

      new_leaf = majority(training_data)
      error3 = error(validation_data, new_leaf)

      if error3 < error1 + error2:
        # prune      
        new_node = {'atri':None,'value':None,'l': None,'r': None,'leaf':new_leaf}
        print('----- pruing once', new_node == node)
        return new_node
        
      else:
        # do not prune
        return node
    else:
      # combine two branch and check again
      new_l = dfs(l, training_data_l, validation_data_l)
      new_r = dfs(r, training_data_r, validation_data_r)
      new_node = {'atri':atri_index, 'value':split_value, 'l': new_l, 'r': new_r, 'leaf': leaf}
      

      #------------------------------------------------------------------------------------
      l_leaf = new_l['leaf']
      r_leaf = new_r['leaf']
      if l_leaf != -1 and r_leaf != -1:        
    
        error1 = error(validation_data_l, l_leaf)
        error2 = error(validation_data_r, r_leaf)

        new_leaf = majority(training_data)
        error3 = error(validation_data, new_leaf)

        if error3 < error1 + error2:
          # prune again    
          print('----- pruing on pruned node')
          new_node = {'atri':None, 'value':None, 'l': None, 'r': None, 'leaf': new_leaf}
        
        else:
          pass
      #------------------------------------------------------------------------------------    
      return new_node


def error(data, leaf):
  e = 0
  for i in data:
    if i[7] != leaf:
      e+= 1
  return e


#============================================================================
def tfcv(data):
  N = 10
  total_len = data.shape[0]
  one_fold = int(total_len/10)
  total_error_rate = 0
  
  low = 1000
  node = None
  
  acc = []
  p_acc = []
  node1 = []
  node2 = []
  
  for i in range(N):        
    test_data = data[one_fold*i:one_fold*(i+1)]
    train_data = np.concatenate((data[0: one_fold*i], data[one_fold*(i+1):]), axis = 0)
    root_node, _ = decision_tree_learning(train_data, 0) 
    
    error = evaluate(root_node, test_data)
    error_rate = float(error) / one_fold
    
    node1.append(root_node)
    acc.append(1 - error_rate)      
    root2= dfs(root_node, training_data, test_data)
    node2.append(root2)
    error2 = evaluate(root2, test_data)
    error2_rate = float(error2) / one_fold
    p_acc.append(1 - error2_rate)
    
    if error_rate < low:
      low = error_rate
      node = root_node
    
    total_error_rate = total_error_rate + error_rate
  error_rate = float(total_error_rate)/N
  
  return 1-error_rate, low, acc, p_acc, node1, node2



######################### test
'''
data = np.loadtxt('wifi_db/noisy_dataset.txt')
#np.random.seed(2)
idx = np.random.permutation(data.shape[0])
data = data[idx]


l8 = int(8*(data.shape[0]/10.0))
l9 = int(9*(data.shape[0]/10.0))
training_data = data[0:l8]
validation_data = data[l8:l9]
test_data = data[l9:]

tfcv(data)
'''
#dt= decision_tree_learning(data, 0)
'''
d1 = 0
root1, d = decision_tree_learning(training_data, d1)
cr1 = evaluate(root1, test_data)/float(len(test_data))
print('----before pruing', 1-cr1, 'depth: ', d)

d2 = 0
root2= dfs(root1, training_data, validation_data)
cr2 = evaluate(root2, test_data)/float(len(test_data))
print('----after pruing:', 1-cr2, 'depth: ', d2)

print(root1 == root2)
'''

'''
cr, low, acc, p_acc, node1, node2 = tfcv(data[0:l9])
print(cr, low)
print(acc)
print(p_acc)
high = 0
idx = 0
for i in range(len(acc)):
  if p_acc[i]-acc[i]> high:
    high = p_acc[i]-acc[i]
    idx = i
print(idx, ' : ', high)
'''


'''
root = root1
cr = 0
while not pruing_check(root):
  root = dfs(root, training_data, validation_data)
  cr_new = 1-evaluate(root, test_data)/float(len(test_data))
  print(cr_new)
  if cr_new > cr:
    cr = cr_new
  else:
    break

## check if the tree is fully pruned
print('pruned ? ',pruing_check(root2))
'''
#print(predict(root1, test_data))
