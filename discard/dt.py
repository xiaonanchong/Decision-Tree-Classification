import numpy as np
import cw2
data = np.loadtxt('WIFI_db/clean_dataset.txt')
#print(data) # [2000, 8]

## return dictionary object {'atri', 'value', 'l', 'r', leaf}
#atri [0,6]
#value >, <=
    
def same_label(data):
  label = data[:, 7]
  label1 = label[0]
  for i in range(label.shape[0]):
    if label[i] != label1:
      return False
      break


######################################################################

def decision_tree_learning(data, depth):
  return {'atri': 0,
          'value': -55,
          'l':{'atri': 'None', 'value':'None', 'l':'None', 'r': 'None', 'leaf': 1}, 
          'r':{'atri': 'None', 'value':'None', 'l':'None', 'r': 'None', 'leaf': 2},
          'leaf': -1}
  '''
  return {'atri': 0,
          'value': -60,
          'l':{None, None, None, None, 'leaf': 1}, 
          'r':{None, None, None, None, 'leaf': 2},
          'leaf': -1}
  '''

#######################################################################
## Step 3: evaluation

N = 10

def ten_fold_cross_validation(data):
  total_len = data.shape[0]
  one_fold = int(total_len/10)
  #print('one_fold', one_fold)

  total_error = 0
  for i in range(N):
        
    test_data = data[one_fold*i : one_fold*(i+1)]
    train_data = np.concatenate((data[0: one_fold*i], data[one_fold*(i+1):]), axis = 0)
    root_node = cw2.decision_tree_learning(train_data, 0) #### training function
    print(root_node)
    error = evaluate(root_node, test_data)
    #print(i,'th error', error )
    total_error = total_error + error

  error_rate = float(total_error)/(one_fold*N)
  #print('total error:', total_error, 'total sample:', one_fold*N)
  return 1-error_rate


def evaluate(node, split_data): 

  d = np.array(split_data)
  #print('lenght of split data:', d.shape[0])

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
    for i in split_data:
      if int(i[7]) != leaf:
        count = count + 1
    error = count
    #print('#', error) 
  return error
  

def error(leaf, split_data): ## not used!

  count = 0
  for i in split_data:
    if int(i[7]) == leaf:
      count = count + 1
  return count


def extract_data(data, atri_index, split_value):
  right_data = [[0,0,0,0,0,0,0,0]]
  left_data =[[0,0,0,0,0,0,0,0]]
  print('data')
  print(data)
  print()
  for i in range(np.array(data).shape[0]):
    if data[i, atri_index] > split_value:
      right_data = np.concatenate((right_data, [data[i, :]]), axis = 0)
    else:
      left_data = np.concatenate((left_data, [data[i, :]]), axis = 0) 
  return left_data[1:], right_data[1:]

##############################################
#step 4 pruning
'''
def pruing(root_node, validate_data):
  
def depth_first_tree(root):
  if root

def leaf(node):
  if  == ln
''' 

##############################################
dt = decision_tree_learning(data, 0)
cr = ten_fold_cross_validation(data)
print('----', cr)

