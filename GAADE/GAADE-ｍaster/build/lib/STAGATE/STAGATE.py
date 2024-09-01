# import tensorflow.compat.v1 as tf
import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()
import scipy.sparse as sp
import numpy as np
from model import GATE
from tqdm import tqdm

#ADD
import pandas as pd
import numpy as np
import sklearn.neighbors

class STAGATE():

    def __init__(self, hidden_dims, alpha, n_epochs=500, lr=0.0001, 
                 gradient_clipping=5, nonlinear=True, weight_decay=0.0001, 
                 verbose=True, random_seed=2020):
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        self.loss_list = []
        self.lr = lr
        self.n_epochs = n_epochs
        self.gradient_clipping = gradient_clipping
        self.build_placeholders()
        self.verbose = verbose
        self.alpha = alpha
        self.gate = GATE(hidden_dims, alpha, nonlinear, weight_decay)
        self.loss, self.H, self.C, self.ReX = self.gate(self.A, self.prune_A, self.X)
        self.optimize(self.loss)
        self.build_session()

    def build_placeholders(self):
        self.A = tf.sparse_placeholder(dtype=tf.float32)
        self.prune_A = tf.sparse_placeholder(dtype=tf.float32)
        self.X = tf.placeholder(dtype=tf.float32)

    def build_session(self, gpu= True):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if gpu == False:
            config.intra_op_parallelism_threads = 0
            config.inter_op_parallelism_threads = 0
        self.session = tf.Session(config=config)
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    def optimize(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

    def __call__(self, A, prune_A, X):
        for epoch in tqdm(range(self.n_epochs)):
            self.run_epoch(epoch, A, prune_A, X)

    def run_epoch(self, epoch, A, prune_A, X):

        loss, _ = self.session.run([self.loss, self.train_op],
                                         feed_dict={self.A: A,
                                                    self.prune_A: prune_A,
                                                    self.X: X})
        self.loss_list.append(loss)
        #if self.verbose:
        #    print("Epoch: %s, Loss: %.4f" % (epoch, loss))
        return loss

    def infer(self, A, prune_A, X):
        H, C, ReX = self.session.run([self.H, self.C, self.ReX],
                           feed_dict={self.A: A,
                                      self.prune_A: prune_A,
                                      self.X: X})

        return H, self.Conbine_Atten_l(C), self.loss_list, ReX

    def Conbine_Atten_l(self, input):
        if self.alpha == 0:
            return [sp.coo_matrix((input[layer][1], (input[layer][0][:, 0], input[layer][0][:, 1])), shape=(input[layer][2][0], input[layer][2][1])) for layer in input]
        else:
            Att_C = [sp.coo_matrix((input['C'][layer][1], (input['C'][layer][0][:, 0], input['C'][layer][0][:, 1])), shape=(input['C'][layer][2][0], input['C'][layer][2][1])) for layer in input['C']]
            Att_pruneC = [sp.coo_matrix((input['prune_C'][layer][1], (input['prune_C'][layer][0][:, 0], input['prune_C'][layer][0][:, 1])), shape=(input['prune_C'][layer][2][0], input['prune_C'][layer][2][1])) for layer in input['prune_C']]
            return [self.alpha*Att_pruneC[layer] + (1-self.alpha)*Att_C[layer] for layer in input['C']]
   
   
    def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
        """\
        Construct the spatial neighbor networks.

        Parameters
        ----------
        adata
            AnnData object of scanpy package.
        rad_cutoff
            radius cutoff when model='Radius'
        k_cutoff
            The number of nearest neighbors when model='KNN'
        model
            The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
        Returns
        -------
        The spatial networks are saved in adata.uns['Spatial_Net']
        """

        assert(model in ['Radius', 'KNN'])
        if verbose:
            print('------Calculating spatial graph...')
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index
        coor.columns = ['imagerow', 'imagecol']

        if model == 'Radius':
            nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
            distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
        if model == 'KNN':
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
            distances, indices = nbrs.kneighbors(coor)
            KNN_list = []
            for it in range(indices.shape[0]):
                KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

        KNN_df = pd.concat(KNN_list)
        KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

        Spatial_Net = KNN_df.copy()
        Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
        id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
        Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
        Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
        if verbose:
            print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
            print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

        adata.uns['Spatial_Net'] = Spatial_Net