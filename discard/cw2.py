import numpy as np

data = np.loadtxt('WIFI_db/clean_dataset.txt')
num_class = 4

def infromation_entropy(label):
    H = 0
    clas = np.unique(label)
    Num = label.shape[0]
    for j in clas:
      Numj = np.count_nonzero(label == j)
      pj = float(Numj)/float(Num)
      H -= pj* np.log2(pj)  
    return H

def majority(label):
    clas = np.unique(label)
    Num = label.shape[0]
    leaf = -1
    for j in clas:
      Numj = np.count_nonzero(label == j)
      pj = float(Numj)/float(Num)
      if pj > leaf:
        leaf = j     
    
    return leaf

def decision_tree_learning(dataset, depth):
    diction = {'atri':None,'value':None,'l': None,'r': None,'leaf':None}
    NumS = dataset.shape[0]
    Num = dataset.shape[1]
    label = dataset[:,Num-1]
    data = dataset[:,0:Num-1] 
    clas = np.unique(label)
    Nclas = clas.shape[0]

    H = infromation_entropy(label)

    record_right = []  ### wrong
    record_left = []  ### wrong

    record_atri = 0
    record_num = 0
	
    record = H

    ReR = 0
    ReL	= 0	

    flag = False    

    for i in range(0,Num-1): 
      data_use = data[:,i] 
      #print('data_use:', data_use)
      dataunique = np.unique(data_use)      
      for j2 in dataunique:
          left = []
          right = [] 
          left_data = []
          right_data = []
          for k in range(0,NumS):
	    if data_use[k] <= j2:
	       left.append(label[k])
	       left_data.append(dataset[k,:])		
	    else: 	
	       right.append(label[k])
	       right_data.append(dataset[k,:])

          
	
          # print('--------------------------left', left_data)
	  H_left = infromation_entropy(np.array(left))		
	  H_right = infromation_entropy(np.array(right))	
	  H_record = H_left + H_right
          # print('----------------------HHRECORD', H_record)
          
          if H_record < record: ### never satisfied 
            flag = True
            #print('check empty:', len(right_data), len(left_data))
            #if right_data==[] or left_data == []:
               #print('***********************************************')
            record = H_record
            ReR =  H_right
            ReL =  H_left
            record_atri = i
            record_num = j2
    	    record_right = np.array(right_data)
            record_left = np.array(left_data) 	
    #print('flag:', flag)
    if flag == False:
        diction['leaf'] = majority(label)

    
               
    information_gain = H - record
    
    print('--------------------H', H)
    print('--------HRECORD',ReR)
    print('--------LRECORD', ReL)
    print('----------------------RECORD', record)
    print('-------------------------GAIN', information_gain)
    print('----------------------------------------DEP', depth)
    
    diction['atri'] = record_atri
    diction['value'] = record_num
    
    	    
    if record != 0.:
      depth += 1 
      # print('-----------------------RIGHT', record_right.shape)
      	
      if ReL != 0:		
        diction['leaf'] = -1
        dicleft=decision_tree_learning(record_left, depth)
        #print(record_left.shape[0])
        diction['l'] = dicleft  
      else:
        if len(record_left) != 0:
           diction['l'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':record_left[0,7]}
           diction['leaf'] = -1
        #else:
           #print('#######L')
           #print(record_left)
           #print(record_right)

      # print('-----------------------LEFT', record_left.shape)
      if ReR != 0:
        diction['leaf'] = -1		
        dicright=decision_tree_learning(record_right, depth) 
        diction['r'] = dicright 	 
      else:	
        if len(record_right) != 0:
           diction['r'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':record_right[0,7]}
           diction['leaf'] = -1
        #else:
           #print('#######R')
           #print(record_right)
           #print(record_left)

    else:
      diction['leaf'] = -1
      diction['l'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':record_left[0,7]}
      diction['r'] = {'atri':None,'value':None,'l': None,'r': None,'leaf':record_right[0,7]}
    return diction

########################################################
#print(decision_tree_learning(data,0))

'''
{'atri': 4, 'r': {'atri': 0, 'r': {'atri': None, 'r': None, 'leaf': 3.0, 'l': None, 'value': None}, 'leaf': -1, 'l': {'atri': None, 'r': None, 'leaf': 4.0, 'l': None, 'value': None}, 'value': -52.0}, 'leaf': -1, 'l': {'atri': 0, 'r': {'atri': 0, 'r': {'atri': 3, 'r': {'atri': 0, 'r': None, 'leaf': 2.0, 'l': None, 'value': 0}, 'leaf': -1, 'l': {'atri': None, 'r': None, 'leaf': 3.0, 'l': None, 'value': None}, 'value': -52.0}, 'leaf': -1, 'l': {'atri': 3, 'r': {'atri': None, 'r': None, 'leaf': 2.0, 'l': None, 'value': None}, 'leaf': -1, 'l': {'atri': 4, 'r': {'atri': 3, 'r': {'atri': 5, 'r': {'atri': None, 'r': None, 'leaf': 2.0, 'l': None, 'value': None}, 'leaf': -1, 'l': {'atri': 6, 'r': {'atri': None, 'r': None, 'leaf': 2.0, 'l': None, 'value': None}, 'leaf': -1, 'l': {'atri': 0, 'r': None, 'leaf': 2.0, 'l': None, 'value': 0}, 'value': -73.0}, 'value': -73.0}, 'leaf': -1, 'l': {'atri': None, 'r': None, 'leaf': 1.0, 'l': None, 'value': None}, 'value': -67.0}, 'leaf': -1, 'l': {'atri': None, 'r': None, 'leaf': 2.0, 'l': None, 'value': None}, 'value': -78.0}, 'value': -40.0}, 'value': -45.0}, 'leaf': -1, 'l': {'atri': 4, 'r': {'atri': None, 'r': None, 'leaf': 4.0, 'l': None, 'value': None}, 'leaf': -1, 'l': {'atri': 3, 'r': {'atri': None, 'r': None, 'leaf': 3.0, 'l': None, 'value': None}, 'leaf': -1, 'l': {'atri': 2, 'r': {'atri': None, 'r': None, 'leaf': 4.0, 'l': None, 'value': None}, 'leaf': -1, 'l': {'atri': 2, 'r': {'atri': None, 'r': None, 'leaf': 3.0, 'l': None, 'value': None}, 'leaf': -1, 'l': {'atri': 4, 'r': {'atri': None, 'r': None, 'leaf': 3.0, 'l': None, 'value': None}, 'leaf': -1, 'l': {'atri': 4, 'r': {'atri': None, 'r': None, 'leaf': 4.0, 'l': None, 'value': None}, 'leaf': -1, 'l': {'atri': 0, 'r': None, 'leaf': 1.0, 'l': None, 'value': 0}, 'value': -59.0}, 'value': -58.0}, 'value': -50.0}, 'value': -49.0}, 'value': -54.0}, 'value': -57.0}, 'value': -56.0}, 'value': -56.0}


'''


