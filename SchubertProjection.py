import numpy as np
import matplotlib.pyplot as plt
import time
from helper_functions import dUU
import os

class SchubertProjection:
    X: np.ndarray
    Omega: np.ndarray
    r: int
    lamb: float
    step_size: float
    g_threshold: float
    bound_zero: float
    singular_value_bound: float
    g_column_norm_bound: float
    U_manifold_bound: float
    shape:list

    m:int
    n:int

    U_array: np.ndarray
    X0:list
    
    def save_model(self, path):
        np.savez_compressed(path, X=self.X,
                                Omega = self.Omega,
                                r = self.r,
                                lamb = self.lamb,
                                g_threshold = self.g_threshold,
                                bound_zero = self.bound_zero,
                                singular_value_bound = self.singular_value_bound,
                                g_column_norm_bound = self.g_column_norm_bound,
                                U_manifold_bound = self.U_manifold_bound,
                                U_array = self.U_array)
                                
        print('Successfully save to: ' + path )
        return True

    def load_model(path):
        #rebuild the object
        data = np.load(path)
        SP = SchubertProjection(X = data['X'],
                Omega = data['Omega'],
                r = data['r'],
                lamb = data['lamb'],
                g_threshold = data['g_threshold'],
                bound_zero = data['bound_zero'],
                singular_value_bound = data['singular_value_bound'],
                g_column_norm_bound = data['g_column_norm_bound'],
                U_manifold_bound = data['U_manifold_bound'],
                U_array = data['U_array'])
        
        print('Successfully loaded the model!')
        return SP


    #U_array load version
    def __init__(self, X, Omega, r, lamb,
                g_threshold = 0.15,
                bound_zero = 1e-10,
                singular_value_bound = 1e-2,
                g_column_norm_bound = 1e-5,
                U_manifold_bound = 1e-2,
                **optional_params):
           
        print('\n########### GrassmannianFusion Initialization Start ###########')
        self.X = X
        self.Omega = Omega
        self.r = r
        self.lamb = lamb
        self.g_threshold = g_threshold
        self.bound_zero = bound_zero
        self.singular_value_bound = singular_value_bound
        self.g_column_norm_bound = g_column_norm_bound
        self.U_manifold_bound = U_manifold_bound

        self.m = X.shape[0]
        self.n = X.shape[1]

        self.shape = [self.m, self.n, self.r]
        
        if 'U_array' in optional_params:
            self.load_U_array(optional_params['U_array'])
            print('U_array Loaded successfully!')
        else:
            self.initialize_U_array()
            print('U_array initialized successfully!')

        self.construct_X0()
        
        print('########### GrassmannianFusion Initialization END ###########\n')

    def get_U_array(self):
        return self.U_array.copy()


    def load_U_array(self, U_array):
        self.U_array = U_array.copy()
        
    def change_lambda(lamb: float):
        self.lamb = lamb

    def initialize_U_array(self):
        U_array = np.array([np.random.randn(self.m, self.r) for i in range(self.n)])
        for i in range(self.n):
            U_array[i,:,0] = self.X[:,i] / np.linalg.norm(self.X[:,i])
            q_i,r_i = np.linalg.qr(U_array[i])
            U_array[i,:,:] = q_i * r_i[0,0]

            #print(U_array[i].shape)
            #make sure the first col is x_i
            assert np.linalg.norm(U_array[i,:,0] - self.X[:,i] / np.linalg.norm(self.X[:,i])) < self.bound_zero
            #make sure its orthogonal
            assert np.linalg.norm(U_array[i].T @ U_array[i] - np.identity(self.r)) < self.bound_zero
            #make sure its normal
            assert  np.linalg.norm( np.linalg.norm(U_array[i], axis = 0) - np.ones(self.r) )  < self.bound_zero

        self.load_U_array(U_array)

    def construct_X0(self):
        #construct X^0_i
        Omega_i = [np.sort( self.Omega[ self.Omega % self.n == self.i]) // self.n for self.i in range(self.n)]
        #find the compliment of Omega_i
        Omega_i_compliment = [sorted(list(set([i for i in range(self.m)]) - set(list(o_i)))) for o_i in Omega_i]
        #calculate length of U
        len_Omega = [o.shape[0] for o in Omega_i]

        #init X^0
        self.X0 = [np.zeros((self.m, self.m - len_Omega[i] + 1)) for i in range(self.n)]
        for i in range(self.n):
            #fill in the first row with normalized column
            self.X0[i][:,0] = self.X[:,i] / np.linalg.norm(self.X[:,i])
            for col_index,row_index in enumerate(Omega_i_compliment[i]):
                #fill in the "identity matrix"
                self.X0[i][row_index, col_index+1] = 1

    def project(self, xi, Uj):
        #projects U_j onto the Schubert variety corresponding to X_i
        #A = self.X0[xi] # for a given matrix, first index is row, second is column
        print('shape X0: ', self.X0.shape)
        print('U_array: ', self.U_array.shape)
        #B = self.U_array[Uj]
        print("Shape A: ", A.shape)
        print("Shape B: ", B.shape)
        #U, s, Vt = np.linalg.svd(A.T @ B)
        #print("Shape U: ", U.shape)
        #print("Shape Vt: ", Vt.shape)
        #P = A@U
        #print("Shape P: ", P.shape)
        #Q = B@Vt.T
        #print("Shape Q: ", Q.shape)
        #new_U = Q
        #new_U[:,0] = P[:,0]
        #self.U_array[Uj] = new_U
        
    def testProjection(self, xi, Uj):
        self.project(xi, Uj)
        A = self.X0[xi] @ self.X0[xi].T @ self.U_array[Uj]
        U_A, s_A, VT_A = np.linalg.svd(A) #SVD of X_iX_i^T U_j
        u = U_A[:,0]
        vt = VT_A[0,:] ##### verify that these are taken properly #####

        if s_A[0] > 1 and s_A[0] - 1 < self.singular_value_bound:
            s_A[0] = 1
        elif s_A[0] > 1:
            raise Exception('Chordal, s_A[0] = ', s_A[0])
        print(s_A[0])
    
    def train(self, max_iter:int, step_size:float):
        obj_record = []
        gradient_record = []
        start_time = time.time()
        
        print('\n################ Training Process Begin ################')
        #main algo
        for iter in range(max_iter):

            new_U_array, end, gradient_norm = self.Armijo_step(alpha = step_size,
                                                        beta = 0.5,
                                                        sigma = 1e-5)
            
            new_np_U_array = np.empty((self.n, self.m, self.r))
            if iter % 1 == 0: ## projects back to the grassmannian after (1) iterations
                for i in range(self.n):
                    u,s,vt = np.linalg.svd(new_U_array[i], full_matrices= False)
                    new_np_U_array[i,:,:] = u@vt

            assert np.linalg.norm(new_np_U_array[i].T @ new_np_U_array[i] - np.identity(new_np_U_array[i].shape[1])) < self.U_manifold_bound
            self.load_U_array(new_np_U_array)

            #record
            obj = self.cal_obj(self.U_array)
            obj_record.append(obj)
            gradient_record.append(gradient_norm)

            #print log
            if iter % 10 == 0:
                print('iter', iter)
                print('Obj value:', obj)
                print('gradient:', gradient_norm)
                print('Time Cost(min): ', (time.time() - start_time)/60 )
                print()

            if end:
                print('Final iter', iter)
                print('Final Obj value:', obj)
                break
                
        print('################ Training Process END ################\n')


    def cal_obj(self, U_array):

        w = np.zeros((self.n, self.n))
        chordal_dist = np.zeros((self.n, self.n))

        obj = 0

        for i in range(self.n):
            for j in range(self.n):
                A = self.X0[i] @ self.X0[i].T @ U_array[j]
                U_A, s_A, VT_A = np.linalg.svd(A) #SVD of X_iX_i^T U_j
                u = U_A[:,0]
                vt = VT_A[0,:] ##### verify that these are taken properly #####

                chordal_dist[i][j] = 1 - s_A[0]**2 # gives d_c^2(xi, Uj)

        for i in range(self.n):
            for j in range(self.n):
                w[i][j] = 1/(1 + np.exp(self.weight_factor * ( chordal_dist[i][j] - self.weight_offset ))) # (old weight) np.exp(self.weight_factor * -0.5 * (chordal_dist[i][j]))

        geodesic_distances = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(self.n):
                u_j, s_j, vt_j = np.linalg.svd(U_array[j] @ U_array[j].T @ U_array[i]) # gives components of distance d_g(Uj, Ui)

                for r_index in range(self.r):
                    if s_j[r_index] > 1 and s_j[r_index] - 1 < self.singular_value_bound:
                        s_j[r_index] = 1
                    elif s_j[r_index] > 1:
                        raise Exception('Ui^T Uj, s[0] = ', s_j[r_index])

                geodesic_distances[i][j] = sum([np.arccos(s_j[i_r])**2 for i_r in range(self.r)]) # gives d_g^2(Ui, Uj)

        for i in range(self.n):
            obj += chordal_dist[i][i]
            for j in range(self.n):
                obj += self.lamb / 2 * w[i][j] * geodesic_distances[i][j]

        return obj


    def Armijo_step(self, alpha = 1, beta = 0.5, sigma = 0.9):
        L = R = 0

        w = np.zeros((self.n,self.n))
        chordal_dist = np.zeros((self.n,self.n))
        chordal_gradients = np.empty((self.n,self.n), dtype=np.ndarray) ######### issue with type, check where needed
        w_gradients = np.empty((self.n,self.n), dtype=np.ndarray)
        #print("test", type(w_gradients))
        #print("test", type(chordal_gradients[1][1]))

        for i in range(self.n):
            for j in range(self.n):
                A = self.X0[i] @ self.X0[i].T @ self.U_array[j]
                U_A, s_A, VT_A = np.linalg.svd(A) #SVD of X_iX_i^T U_j
                u = U_A[:,0]
                vt = VT_A[0,:] ##### verify that these are taken properly #####

                if s_A[0] > 1 and s_A[0] - 1 < self.singular_value_bound:
                    s_A[0] = 1
                elif s_A[0] > 1:
                    raise Exception('Chordal, s_A[0] = ', s_A[0])

                chordal_dist[i][j] = 1 - s_A[0]**2 # gives d_c^2(xi, Uj)
                chordal_gradients[i][j] = -2 * s_A[0] * np.outer(u,vt) # gives the gradient d_c^2(x_i,U_j) w.r.t. U_j

                #print("test grad", type(chordal_gradients[1][1]))


        for i in range(self.n):
            for j in range(self.n):
                w[i][j] = 1/(1 + np.exp(self.weight_factor * ( chordal_dist[i][j] - self.weight_offset ))) #old weight np.exp(self.weight_factor * -0.5 * (chordal_dist[i][j]))
                w_gradients[i][j] = -self.weight_factor * np.exp(self.weight_factor * (chordal_dist[i][j] - self.weight_offset)) * w[i][j]**2 * chordal_gradients[i][j] #old w[i][j] * self.weight_factor * (-0.5) * chordal_gradients[i][j] # gives the gradient of w_ij w.r.t. U_j

        geodesic_distances = np.zeros((self.n,self.n))
        geodesic_gradients = np.empty((self.n,self.n), dtype=np.ndarray) ########## issue with type, check where needed

        for i in range(self.n):
            for j in range(self.n):
                u_j, s_j, vt_j = np.linalg.svd(self.U_array[j] @ self.U_array[j].T @ self.U_array[i]) # gives components of distance d_g(Uj, Ui)

                for r_index in range(self.r):
                    if s_j[r_index] > 1 and s_j[r_index] - 1 < self.singular_value_bound:
                        s_j[r_index] = 1
                    elif s_j[r_index] > 1:
                        raise Exception('Ui^T Uj, s[0] = ', s_j[r_index])

                geodesic_distances[i][j] = sum([np.arccos(s_j[i_r])**2 for i_r in range(self.r)]) # gives d_g^2(Ui, Uj)
                geodesic_gradients[i][j] = np.zeros((self.m, self.r))

                for r_index in range(self.r): # if/ else to account for computational errors when s=1, equivalent forms in the limit
                    if s_j[r_index] < 1:
                        geodesic_gradients[i][j] += -2 * np.arccos(s_j[r_index]) / np.sqrt(1 - s_j[r_index]**2) * np.outer(u_j[:, r_index] , vt_j[r_index, :])
                    else:
                        geodesic_gradients[i][j] += -2 * np.outer(u_j[:, r_index] , vt_j[r_index, :]) # gradient w.r.t U_i


        #calculate the true gradient
        grad_f_array = []
        for i in range(self.n):
            grad_f_i = chordal_gradients[i][i] #takes chordal gradient
            for j in range(self.n):

            #cap singular values
            #for r_index in range(r):
            # if (s_j[r_index] - 1 > 0):
            #  s_j[r_index] = 1

                dg_UU = w[i][j] * geodesic_gradients[i][j] + (w_gradients[j][i] * geodesic_distances[j][i] + w[j][i] * geodesic_gradients[i][j])
                grad_f_i += self.lamb / 2 * dg_UU

            grad_f_i = (np.identity(self.m) - self.U_array[i] @ self.U_array[i].T) @ grad_f_i
            grad_f_array.append(grad_f_i) #appends true projected gradient w.r.t. U_i of overall function

        gradient_norm = 0
        for i in range(self.n):
            gradient_norm += np.trace(grad_f_array[i].T @ grad_f_array[i])
        gradient_norm = np.sqrt(gradient_norm) #norm of overall gradient for iter

        #avoid using m
        arm_m = 0
        while True:
            #print('Testing Step size:' , (beta**arm_m) * alpha)
            new_U_array = [np.zeros((self.m,self.r)) for i in range(self.n)]
            ###############
            L = 0
            for i in range(self.n):
                L += chordal_dist[i][i]
                for j in range(self.n):
                    L += self.lamb / 2 * w[i][j] * geodesic_distances[i][j]
            #L = cal_obj(shape, X0, U_array, lamb,singular_value_bound) #calculate overall objective using previous U_array
            ##### already computed SVDs above, don't do again.

            for i in range(self.n):
                Gamma_i, Del_i, ET_i = np.linalg.svd( -1 * (beta**arm_m) * alpha * grad_f_array[i], full_matrices= False)
                first_term = np.concatenate((self.U_array[i]@ET_i.T, Gamma_i), axis = 1)
                second_term = np.concatenate((np.diag(np.cos( Del_i)), np.sin(np.diag(Del_i))), axis = 0)

                new_U_array[i] = first_term @ second_term @ ET_i #new array using armijo step and computed gradient

            ###############
            L -= self.cal_obj(new_U_array) #objective using new array
            R =  -1 * sigma
            inner_sum = 0
            for i in range(self.n):
                inner_sum += np.trace(grad_f_array[i].T @ grad_f_array[i] * (beta**arm_m) * alpha * -1) #scaled norm of new array

            R = R * inner_sum

            #print('L:', L)
            #print('R:', R)

            if L >= R: #steps if successful
                #print('Step: ', (beta**arm_m) * alpha
                return new_U_array, False, gradient_norm #returns new_U_array
            else:

                if (beta**arm_m) * alpha < 1e-10:
                #print('No Step')
                    return self.U_array, True, gradient_norm

            arm_m += 1 #otherwise, increase arm_m


    def distance_matrix(self):
        #calculate the distance
        d_matrix = []
        for i in range(self.n):
            d_matrix_row = []
            for j in range(self.n):
                if i == j:
                    d_matrix_row.append(0)
                    continue
                d_matrix_row.append(dUU(self.U_array[i], self.U_array[j], self.r))

            d_matrix.append(d_matrix_row)

        d_matrix = np.array(d_matrix)

        return d_matrix
    
    def chordal_distance_matrix(self):
        #calculate the distance matrix with respect to the chordal distance
        chordal_matrix = []
        for i in range(self.n):
            chordal_matrix_row = []
            for j in range(self.n):
                A = self.X0[i] @ self.X0[i].T @ self.U_array[j]
                U_A, s_A, VT_A = np.linalg.svd(A) #SVD of X_iX_i^T U_j
                u = U_A[:,0]
                vt = VT_A[0,:] ##### verify that these are taken properly #####

                if s_A[0] > 1 and s_A[0] - 1 < self.singular_value_bound:
                    s_A[0] = 1
                elif s_A[0] > 1:
                    raise Exception('Chordal, s_A[0] = ', s_A[0])
                
                chordal_matrix_row.append(1 - s_A[0]**2) # gives d_c^2(xi, Uj)
            
            chordal_matrix.append(chordal_matrix_row)
            
        chordal_matrix = np.array(chordal_matrix)
        
        return chordal_matrix
        
