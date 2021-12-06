
from math import nan
from utils import *
from preprocessDN import *

# import torchvision

class GraphLossDN():
    def __init__(self,
        pLambda,
        pAlpha,
        pCigma_int,
        pCigma_spa,
        pB,
        pK,
        pQ
    ) -> None:
        
        '''
        B,Q must be odd number !
        '''
        self._pLambda = pLambda
        self._pAlpha = pAlpha
        self._pCigma_int = pCigma_int
        self._pCigma_spa = pCigma_spa
        self._pB = pB
        self._pK = pK
        self._pQ = pQ
        
        self._cut_Q = int(pQ / 2.0)
        self._cut_B = int(pB / 2.0)
        self._cut = self._cut_Q + self._cut_B
        
    '''Auxiliary functions'''
    def get_coord_jmn(self, mat_d_upleftcoord, j):
        '''
        Input related j and up left coordinate in mat_d, return mj,nj in mat_d
        '''
        m_related = j // ( 2 * self._cut_B + 1 )
        n_related = j % ( 2*self._cut_B + 1 )
        
        mj = mat_d_upleftcoord[0] + m_related
        nj = mat_d_upleftcoord[1] + n_related

        return mj,nj
        
    def get_coord_imn(self, mat_aux_imn):
        '''
        Input mi,ni in auxiliary matrix, return mi,ni in mat_d
        '''
        mi = mat_aux_imn[0] + self._cut
        ni = mat_aux_imn[1] + self._cut
        return mi,ni

    def loss_omega_mat(self,mat_d):

        cut = self._cut
        cq = self._cut_Q    
         
        mat_omega_all = torch.ones((mat_d.shape[0]-2*cut, mat_d.shape[1]-2*cut, self._pB*self._pB),dtype=torch.float32) 
        
        ## Get mat_omega_all
        '''
            for each grid(mi,ni) in mat_omega_all:
                mi,ni = get_coord_imn()
                up left corner is (m1,n1) : m1=mi-cut_B , n1=ni-cut_B
                for each channel(j) in grid:
                    get (mj,nj)
                    
                    if (mj,nj) is (m1,n1) :
                        continue
                        
                    Qij = torch.norm( mat_d[] - mat_d[]  )
                    wij_1 = torch.exp( ... )
                    wij_2 = torch.exp( ... (mi,ni)..(mj,nj) ) 
                    
                    mat_omega_all[aux_mi,aux_ni,j] = wij_1 * wij_2
        '''
        
        
        '''TODO: (test)'''
        test_counter = 0
        
        
        for aux_mi in range(mat_omega_all.shape[0]):
            for aux_ni in range(mat_omega_all.shape[1]):

                
                '''TODO: (test)'''
                test_counter += 1
                if(test_counter % ( int( (mat_d.shape[0]*mat_d.shape[1]) / 10 ) ) == 0):
                    print(f'test_counter in loss_omega_mat(): {test_counter}')
                
                
                mi,ni = self.get_coord_imn([aux_mi,aux_ni])
                imn = torch.tensor([mi,ni])              
                ul_coord = [mi-self._cut_B, ni-self._cut_B]                
                  
                for j in range(mat_omega_all.shape[2]):
                    mj,nj = self.get_coord_jmn(ul_coord, j)
                    jmn = torch.tensor([mj,nj]) 
                                       
                    if (mj == mi and nj==ni):
                        continue
                    
                    Qij = torch.norm( mat_d[mi-cq:mi+cq+1, ni-cq:ni+cq+1] - mat_d[mj-cq:mj+cq+1, nj-cq:nj+cq+1] )**2
                    wij_1 = torch.exp( -Qij / (2*(self._pCigma_int**2)) )
                    wij_2 = torch.exp( torch.sum(-torch.pow( imn-jmn, 2)) / (2*(self._pCigma_spa**2)) ) 
                    wij = wij_1*wij_2
                    mat_omega_all[aux_mi,aux_ni,j] = wij
                    
                    '''TODO: (test) check whether wij is nan'''
                    if(wij == nan):
                        print("wij is nan !  wij_1: {wij_1}, wij_2: {wij_2}")
                    
        

        ## Retreve top K channel for each grid from mat_omega_all
        
        '''
            use torch.topk to get top k+1 values and indices matrix
            remember that the first elem is 1 which corresponds to the center element itself, and should be abandon
            can use indices to recover (mj,nj) as above
            
        '''
        mat_omega_topk,mat_indices = torch.topk(mat_omega_all, self._pK+1)  # +1: Remember the center elem itself
        mat_omega_topk = mat_omega_topk[:,:,1:]
        mat_indices = mat_indices[:,:,1:]
        
        # mat_omega_topk.requires_grad = True
        # mat_omega_all.requires_grad = True
        return mat_omega_topk, mat_indices
    
        
    def loss_duij_mat(self,mat_d,mat_u,mat_indice):
        '''
            Use mat_indice from omega topk to recover (mj,nj) and count duij matrix
            mat_indice : (n1,n2,n3)          
        '''
        mat_duij = torch.zeros_like(mat_indice,dtype=torch.float32)
        
        '''
            for each grid(i,j) in mat_duij:
                mi,ni = get_coord_imn()
                upper left coordinate m_ul=mi-cut_B, n_ul=ni-cut_B
                for each channel c :
                    j = mat_indice[mi,ni,c]
                    mj,nj = get_coord_jmn()
                    duij = torch.pow( d[mj,nj] - d[mi,ni] - (u[mi,ni] * torch.tensor([mj-mi,nj-ni])) , 2)
                    mat_duij[aux_mi,aux_ni,c] = duij
        '''
        for aux_mi in range(mat_duij.shape[0]):
            for aux_ni in range(mat_duij.shape[1]):

                mi,ni = self.get_coord_imn([aux_mi,aux_ni])
                ul_coord = [mi-self._cut_B, ni-self._cut_B]
                
                for c in range(mat_duij.shape[2]):
                    j = mat_indice[aux_mi,aux_ni,c]
                    mj,nj = self.get_coord_jmn(ul_coord,j)
                    vec_ij = torch.tensor([mj-mi,nj-ni],dtype=torch.float32)
                    duij = torch.pow( mat_d[mj,nj] - mat_d[mi,ni] - torch.dot(mat_u[mi,ni],vec_ij) , 2 )
                    mat_duij[aux_mi,aux_ni,c] = duij
        
        return mat_duij
    
    def loss_uij_mat(self,mat_u,mat_indice):
        '''
            Use mat_indice from omega topk to recover (mj,nj) and count uij matrix
        '''
        
        mat_uij = torch.zeros_like(mat_indice, dtype=torch.float32)

        
        '''
            for each grid(aux_mi,aux_ni) in mat_uij:
                mi,ni = get_coord_imn()
                ul = [mi-cut_B, ni-cut_B]
                for each c in mat_uij:
                    j = mat_indice[mi,ni,c]
                    mj,nj = get_coord_jmn()
                    uij = torch.norm( u[mj,nj] - u[mi,ni] )
                    mat_uij[aux_mi,aux_ni,c] = uij
                    
        '''
        for aux_mi in range(mat_uij.shape[0]):
            for aux_ni in range(mat_uij.shape[1]):
                
                mi,ni = self.get_coord_imn([aux_mi,aux_ni])            
                ul = [mi-self._cut_B, ni-self._cut_B]
                
                for c in range(mat_uij.shape[2]):
                    j = mat_indice[aux_mi,aux_ni,c]
                    mj,nj = self.get_coord_jmn(ul,j)
                    uij = torch.norm( mat_u[mj,nj] - mat_u[mi,ni])
                    mat_uij[aux_mi,aux_ni,c] = uij

        return mat_uij
     
     
     
    '''Core functions'''
    def fidalityTerm(self,mat_d_orig,mat_d,mat_c):
        loss_fidality = torch.abs(mat_d - mat_d_orig) * mat_c
        return loss_fidality

    def regularizTerm(self,mat_d,mat_u):
        '''
        Multiply the aformentioned matrix in the third channel to form the loss
        '''
        mat_omega_topk, mat_indices = self.loss_omega_mat(mat_d)
        mat_duij = self.loss_duij_mat(mat_d, mat_u, mat_indices)
        mat_uij = self.loss_uij_mat(mat_u, mat_indices)
        
        
        
        
        '''TODO: (test) exame these element'''
        print(f'mat_omega_topk: {mat_omega_topk} \n')
        print(f'mat_duij: {mat_duij} \n')
        print(f'mat_uij: {mat_uij} \n')        
        
        
        
        
        '''TODO: the backpropagation interupt here !!! '''
        mat_loss_1 = torch.sqrt( torch.sum(torch.pow(mat_omega_topk,2) * mat_duij , -1) )
        mat_loss_2 = self._pAlpha * torch.sum(mat_omega_topk * mat_uij , -1)
        
        loss_regularization = torch.sum( mat_loss_1 + mat_loss_2 )
        
        return loss_regularization
    

    def lossfunc(self,mat_d_orig,mat_d,mat_u,mat_c):
        
        loss1 = self.fidalityTerm(mat_d_orig,mat_d,mat_c).sum()
        loss2 = self.regularizTerm(mat_d,mat_u).sum()
        loss = loss1 + self._pLambda*loss2
        
        '''TODO: (debug)(test)'''        
        print(f'loss = loss1 + lambda*loss2')
        print(f'loss1:{loss1} loss2:{loss2}')
        
        return loss




if __name__ == '__main__':
    
    '''
    These are demos
    '''
    
    sys.path.append(DIR_NAME)    
    
    '''
    GraphLoss (original paper): 

        cigma_int = 0.07
        cigma_spa = 3
        B = 9
        Q = 3
        K = 20
        
        (grid search)
        lambda = 15, 25        
        alpha = 3.5 ?
    '''
    preprocess_dn = PreprocessDN()
    graphlossdn = GraphLossDN(pLambda=10, pAlpha=3.5, pCigma_int=0.07, pCigma_spa=3, pB=7, pK=10, pQ=3) 
    
    
         
    '''TODO: '''
    # torch.autograd.set_detect_anomaly(True)
    
    
  
    '''read image , copy from preprocessDN __main__'''
    cv_MatI = cv.imread('./images/d_image_2_000000.png',cv.IMREAD_GRAYSCALE)
    
    rows = cv_MatI.shape[0] # y    
    cols = cv_MatI.shape[1] # x
    
    # cv_MatI = cv.resize(cv_MatI, (int(0.1*orig_rows),int(0.1*orig_cols)))
    rowmin = int(0.40*rows)
    rowmax = int(0.55*rows)
    colmin = int(0.35*cols)
    colmax = int(0.45*cols)
    
    cv_MatI = np.array(np.uint8(cv_MatI))
    # print(cv_MatI.shape)
    cv_MatI = cv_MatI[rowmin:rowmax , colmin:colmax]     
    # cv.imshow("cutted matI",cv_MatI)
    # cv.waitKey(0)
    
    
      
      
      
    '''eliminate singular elements'''
    MatI = torch.Tensor(cv_MatI) + 1e-10    # In case it appear zero element
    MatI_lt_one = MatI < 1.0
    MatI[MatI_lt_one] = 1.0


    '''preprocess'''
    MatD,MatU = preprocess_dn.compute(MatI)
    
    
    '''eliminate singular elements'''
    MatD += 1e-10
    MatU += 1e-10 
    
    
    
    MatC = torch.zeros_like(MatD)
        
    MatD_orig = MatD.clone()    # original di
    
    '''TODO: (test)'''
    print("debug:")
    print(f'MatD.shape: {MatD.shape} \n MatD: {MatD}')
    print(f'MatU.shape: {MatU.shape} \n MatUx: {MatU[:,:,0]}   \n\n\n\n')
    
    MatD.requires_grad = True
    MatU.requires_grad = True

    
    '''GraphLoss and Optimizer construction'''
    n_iter = 20
    lr = 0.001   # default: 0.001
    betas = (0.9,0.999)
    eps = 1e-08
    optimizer = optim.Adam([MatD,MatU], lr=lr, betas=betas)
    
    
    for t in range(n_iter):
        optimizer.zero_grad(set_to_none=True)

        loss = graphlossdn.lossfunc(MatD_orig, MatD, MatU, MatC)
        # loss.requires_grad = True
            
        '''TODO: (test)'''
        print(f't:{t}, Finish constructing loss')                
        
        
        '''TODO:'''
        # with torch.autograd.detect_anomaly():
        #     loss.backward()
        loss.backward()        
        
        '''TODO: (test)'''
        print(f'require grad ? {MatD.requires_grad} {MatU.requires_grad} {MatD_orig.requires_grad}')
        print(f'MatD grad and MatUx grad: {MatD.grad.shape} {MatU.grad.shape} \n')
        print(f'MatD.grad: {MatD.grad} \n')
        print(f'MatU.grad[0:10,0:10]: {MatU.grad[0:10,0:10]} \n')
        print(f'MatU.grad[10:20,10:20]: {MatU.grad[10:20,10:20]} \n')
        print(f'MatU.grad[20:30,20:30]:  {MatU.grad[20:30,20:30]} \n')
        print(f't:{t}, loss.item(): {loss.item()}')   
             
        # if( t % (n_iter / 100) == 0 ):
        #     print(t, loss.item())
        
        
        optimizer.step()
        
        
        '''TODO: (test)'''
        print(f'Optimized result :\n MatD: {MatD} \nMatUx: {MatU[:,:,0]}  \n\n\n\n\n ')
        
        # temp_str = DIR_NAME + './temp_result/iter_' + str(t) + '.png'
        # cv.imwrite(temp_str, np.uint8(MatI))
        
        
        
        
    
    
    
    
