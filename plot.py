import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
#--------------------------------------------------------
import predict
data = np.loadtxt('wifi_db/noisy_dataset.txt')
np.random.seed(2)
idx = np.random.permutation(data.shape[0])
data = data[idx]

l8 = int(8*(data.shape[0]/10.0))
l9 = int(9*(data.shape[0]/10.0))
training_data = data[0:l8]
validation_data = data[l8:l9]
test_data = data[l9:]

d = 0
root, depth = predict.decision_tree_learning(training_data, d)
root2= predict.dfs(root, training_data, validation_data)
#---------------------------------------------------------

'''
def label(xy, text):
    x = xy[0] 
    y = xy[1]   # shift y-value for label so that it's below the artist
    plt.text(x+0.005, y, text, ha="center", family='sans-serif', size=8)

fig, ax = plt.subplots()
N = 9j
n = 9
grid = np.mgrid[0.2:0.8:N, 0.2:0.8:100j].reshape(2, -1).T

patches = []
for i in range(2):
  rect = mpatches.Rectangle(grid[i], 0.02, 0.01, ec="none")
  patches.append(rect)
  label(grid[i], str(i))

collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
ax.add_collection(collection)

plt.axis('equal')
plt.axis('off')
plt.tight_layout()

plt.show()

'''
#-------------------------------------------------------------
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(18, 10))

from collections import deque

gap = 1.0/depth

def bfs(root, xmin, xmax, ymin, ymax):
  queue = deque([(root, xmin, xmax, ymin, ymax)])
  while len(queue) > 0:
    #(node, xmin, xmax, ymin, ymax) = queue.popleft()
    e = queue.popleft()
    node = e[0]
    xmin = e[1]
    xmax = e[2]
    ymin = e[3]
    ymax = e[4]
    atri = node['atri']
    val = node['value']
    text = '['+str(atri)+']:'+str(val)
    #---------------------
    center = xmin+(xmax-xmin)/2.0
    d = (center-xmin)/2.0
    
    
    #---------------------
    if node['l'] != None:
      queue.append((node['l'], xmin, center, ymin, ymax-gap))
      ax.annotate(text, xy=(center-d, ymax-gap), xytext=(center, ymax),
            arrowprops=dict(facecolor='grey', shrink=10),
            )
    if node['r'] != None:
      queue.append((node['r'], center, xmax, ymin, ymax-gap))
      ax.annotate(text, xy=(center+d, ymax-gap), xytext=(center, ymax),
            arrowprops=dict(facecolor='grey', shrink=10),
            )
    if node['leaf'] != -1:
    #---------------------
      an1 = ax.annotate(node['leaf'], xy=(center, ymax), xycoords="data",
                  va="bottom", ha="center",
                  bbox=dict(boxstyle="round", fc="w"))
    #---------------------

bfs(root2, 0.0, 1.0, 0.0, 1.0)

fig.subplots_adjust(top=0.83)
plt.show()
