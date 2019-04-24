
# coding: utf-8

# In[599]:


import numpy as np

def binarize(x):
    probs = 1 / (1 + np.exp(-x))
    return [1 if np.random.random() < p else 0 for p in probs]

class RBM(object):
    '''
    Restricted Boltzmann Machine class holds the function used to learn a dataset and simulation.
    Class attributes are:
    
    '''
    def __init__(self, connections, activations=None):
        '''
        Initialize RBM attributes
        - connections: list-like structure containing the connection matrices
                       between layers, starting from the visible layer
                       (that is, the firs matrix maps the input layer to the first
                        hidden layer, and it has as many rows as neurons in the
                        input layer and as many columns as neurons in the first
                        hidden layer)
        
        
        '''
        for i in range(len(connections)-1):
            if connections[i].shape[1] != connections[i+1].shape[0]:
                raise ValueError('Incompatible dimension for connection matrices')
        self.connections = np.array(connections)
        
        self.layer_state = np.array([[0.]*w.shape[0]
                                     for w in connections] +
                                    [[0.]*connections[-1][2]])
        self.biases = self.layer_state.copy()
        
        if activations is None:
            self.activations = [lambda x: x] * len(self.layer_state)
        elif len(connections) != len(activations):
            raise ValueError('Incompatibledimensions for connections and activations')
        else:
            self.activations = activations
        
    def _energy(self, level):
        '''
        Computes the energy of one layer
        Input
        - level: number of layer (from 0, which is visible layer to num_layers - 1)
        '''       
        ##TODO add check on level value
        return -np.dot(np.dot(self.layer_state[level], self.connections[level]),
                       np.transpose(self.layer_state[level+1])) \
               - np.dot(self.layer_state[level], self.biases[level])
   
    def energy(self):
        return sum([self._energy(level) for level in range(len(self.connections))])                - np.dot(self.layer_state[-1], self.biases[-1])
               # this last contribution is for the bias of last hidden layer
    
    def run_from_visible(self, v, verbose=False, starting_layer=1):
        '''encodes a visible vector
           - v: vector to be encoded
           - starting_layer: layer from which the propagation is started
                             (default: 1, which means visible layer; 2 would
                              mean first hidden layer and so on)
        '''
        if len(v) != len(self.layer_state[0]):
            raise ValueError('Specified visible layer is incoherent')
        self.layer_state[0] = np.array(v)
        for level in range(starting_layer, len(self.layer_state)):
            if verbose:
                print('  from visible now in level {}'.format(level-1))
                #print('matrix is {}x{}, vector is {}'.format(self.connections[level-1].shape[0],
                #                                             self.connections[level-1].shape[1],
                #                                             len(self.layer_state[level-1])))
            self.layer_state[level] = self.activations[level](np.dot(self.connections[level-1].T, self.layer_state[level-1])                                                               + self.biases[level])
        return self.layer_state[-1]
    
    def run_from_hidden(self, h, verbose=False, starting_layer=1):
        '''reconstruct from an hidden vector
           - h: hidden vector
           - starting_layer: layer from which the propagation is started
                             (default: 1, which means last hidden layer; 2 would
                              mean second-last hidden layer and so on)
        '''
        if len(h) != len(self.layer_state[-1]):
            raise ValueError('Specified hidden layer is incoherent')
        self.layer_state[-1] = np.array(h)
        for level in range(len(self.layer_state)-starting_layer)[::-1]:
            if verbose:
                print('  from hidden now in level {}'.format(level+1))
                #print('matrix is {}x{}, vector is {}'.format(self.connections[level].shape[0],
                #                                             self.connections[level].shape[1],
                #                                             len(self.layer_state[level+1])))
            self.layer_state[level] = self.activations[level](np.dot(self.connections[level], self.layer_state[level+1])                                                               + self.biases[level])
        
        self.layer_state[0] = binarize(self.layer_state[0])    
        return self.layer_state[0]


# In[601]:


class TransductiveRBM(object):
    
    def __init__(self, w, activations=None):
        m = w.shape[1]
        i = np.eye(m)
        self.rbm = RBM([w, i], activations)
        # self.rbm.biases[1] = [0] * m #as already done in RBM constructor
    
    def grad_h_middle(self, v, h_0):
        v_1 = self.rbm.run_from_hidden(h_0, starting_layer=2)
        h_1 = self.rbm.run_from_visible(v_1)
        return (np.dot(self.rbm.connections[0].T, v) -                 np.dot(self.rbm.connections[0].T, v_1)) +                (h_0 - h_1)
    
    def grad_bias_visible(self, v, h_0):
        v_1 = self.rbm.run_from_hidden(h_0)
        return v - v_1
    
    def grad_bias_hidden(self, h_0):
        v_1 = self.rbm.run_from_hidden(h_0, starting_layer=2)
        h_1 = self.rbm.run_from_visible(v_1)
        return h_0 - h_1
    
    def fit(self, v, use_bias=False, learning_rate=0.1, learning_rate_decay=0.998,
            max_iterations=3000, delta_max=10**-3,
            verbose=False):
        num_iter = 0
        old_h_middle = self.rbm.layer_state[1]
        curr_delta = delta_max * 1.1 # to ensure next loop is always entered
        energy_vals = []
        while num_iter < max_iterations and curr_delta > delta_max: 
            h_0 = self.rbm.run_from_visible(v, starting_layer=2)
            #print('next learning iteration-----')
            #print('grad h middle:')
            self.rbm.layer_state[1] -= learning_rate * self.grad_h_middle(v, h_0)
            #print('variation: {}'.format(learning_rate * self.grad_h_middle(v, h_0)))
            if use_bias:
                #print('grad bias visible:')
                self.rbm.biases[0] -= learning_rate * self.grad_bias_visible(v, h_0)
                #print('grad bias hidden:')
                self.rbm.biases[-1] -= learning_rate * self.grad_bias_hidden(h_0)
            curr_delta = np.max(np.abs(self.rbm.layer_state[1] - old_h_middle))
            old_h_middle = self.rbm.layer_state[1]
            num_iter += 1
            learning_rate *= learning_rate_decay
            if verbose:
                #print('iteration {}, delta={}'.format(num_iter, curr_delta))
                #print('----->h={}'.format(str(self.rbm.layer_state[1])))
                pass
            energy_vals.append(self.rbm.energy())
        
        return energy_vals
        


# In[557]:


n = 25
m = 10
w = np.random.random((n, m))


trbm = TransductiveRBM(w)


# In[558]:


energy = trbm.fit(np.random.randint(2, size=n), learning_rate=1, use_bias=True, delta_max=10**-3, verbose=True)


# In[559]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

plt.plot(range(len(energy)), energy)
plt.show()


# In[525]:


str(trbm.rbm.layer_state[1])


# In[381]:


trbm.rbm.biases


# In[357]:


trbm.rbm.layer_state


# In[382]:


a = np.array((9, 3, -1))


# ## Usiamo i dati veri

# In[561]:


def load_data(filename, separator=' '):
    '''
    Read file with data set, one object per line expressed as space separated integer values
    
    Input
    - filename: string containing the pathname of the file to be read
    - separator: character dividing the columns of the input dataset; default being whitespace
    
    Returns an array of lists
    '''
    with open(filename) as f:
        input_values = [line.strip("\n\r").split(separator) for line in f.readlines()]
    return np.array(input_values).astype(int)

def load_weights(filename, separator=' '):
    '''
    Read file with data set, one object per line expressed as space separated integer values
    
    Input
    - filename: string containing the pathname of the file to be read
    - separator: character dividing the columns of the input dataset; default being whitespace
    
    Returns an array of lists containing the weights saved in the file
    '''

    with open(filename) as f:
        input_values = [line.strip("\n\r").split(separator) for line in f.readlines()]
    return np.array(input_values).astype(float)


# In[575]:


data_path = '/home/malchiodi/oldocs/ricerca/rbm/rbm_tesi/src/options/new-dataset/yeast.GOA.ann.9.may.17.onto.CC.tsv'
weight_path = '/home/malchiodi/oldocs/ricerca/rbm/rbm_tesi/src/options/new-dataset/yeast.STRING.v10.5.net_no_labels.tsv'

labels = load_data(data_path, '\t')
weights = load_weights(weight_path, '\t')


# In[606]:


from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

def extract_matrix(m, rows, cols):
    row_idx = np.array(rows)
    col_idx = np.array(cols)
    return m[row_idx[:, None], col_idx]

num_folds = 3 
kf = StratifiedKFold(n_splits=num_folds)
avg_roc = np.array([0.]*labels.T.shape[0])
avg_prc = np.array([0.]*labels.T.shape[0])


for _class in range(labels.T.shape[0]):
    auroc_values = []
    prc_values = []
    num_fold = 1

    expected_all = labels.T[_class]
    for indices_train, indices_test in kf.split(expected_all, expected_all):
        #print("%s %s" % (train, test))
        labels_train = labels[indices_train]
        labels_test = labels[indices_test]
        weights_fold = extract_matrix(weights, indices_train, indices_test)
        trbm = TransductiveRBM(weights_fold)      
    
        trbm.fit(labels_train.T[_class], learning_rate=10, delta_max=5*10**-2)
        pred = trbm.rbm.layer_state[-1]
        expected = labels_test.T[_class]
        auroc = metrics.roc_auc_score(expected, pred)
        auroc_values.append(auroc)
        
        precision, recall, _ = metrics.precision_recall_curve(expected, pred)
        prc = metrics.auc(recall, precision)
        prc_values.append(prc)
        print('    class {}, fold {}: avg roc={}, avg prc={}'.format(_class,
                                                                     num_fold,
                                                                     auroc,
                                                                     prc))
        
        num_fold += 1
    avg_roc[_class] += np.average(auroc_values)
    avg_prc[_class] += np.average(prc_values)
    print('class {}: avg roc={}, avg prc={}'.format(_class, sum_roc[_class], sum_prc[_class]))


# In[594]:


from sklearn.model_selection import StratifiedKFold
X = [0]*7 + [1]*3

num_folds = 3 
kf = StratifiedKFold(n_splits=num_folds)

for indices_train, indices_test in kf.split(X, X):
    print("%s %s" % (indices_train, indices_test))

