

from torch.autograd.grad_mode import F
from utils import *



class PreprocessDN():
    '''
    params : none
    input : depth img MatI
    output : inverse depth img MatD , gradient on inversed MatU
    '''
    def __init__(self) -> None:
        pass

    def _computeInverseDepth(self, mat_i:torch.Tensor) -> torch.Tensor:
        mat_d = 1/mat_i
        return mat_d
        

    def _computeInverseGrad(self, mat_d:torch.Tensor) -> torch.Tensor:
        '''
        mat_d : torch.Tensor, shape: (n1,n2)
        '''
        raws = mat_d.shape[0]
        cols = mat_d.shape[1]
        temp = mat_d.clone()    
           
        '''Ux'''
        mat_ux = temp[:,1:] - mat_d[:,:-1]
        mat_ux = torch.cat((mat_ux,mat_ux[:,-1].reshape((raws,1))),dim=1)
        
        '''Uy'''
        mat_uy = temp[1:,:] - mat_d[:-1,:]
        mat_uy = torch.cat((mat_uy,mat_uy[-1,:].reshape((1,cols))),dim=0)
        
        mat_u = torch.zeros((raws,cols,2)) 
        mat_u[:,:,0] = mat_ux
        mat_u[:,:,1] = mat_uy
        return mat_u
        

    def compute(self, MatI:torch.Tensor):
        '''
        Warning: if there's zero element in input MatI ? The these problem should have been fixed in the 
        input tensor. Each element in input matrix > 1.0.
        '''
        
        MatD = self._computeInverseDepth(MatI)
        MatU = self._computeInverseGrad(MatD)
        return MatD,MatU


if __name__ == '__main__':
    
    '''
    These are demos
    '''
    
    sys.path.append(DIR_NAME)
    preprocess_dn = PreprocessDN()     
    

    # WARNING: For temporary. This is in the case that the depth image is RGB, which is the wrong one.
    cv_MatI = cv.imread('./images/depth.png',cv.IMREAD_GRAYSCALE)
    # cv.imshow("original mat",cv_MatI)
    
    # orig_rows = cv_MatI.shape[1]
    # orig_cols = cv_MatI.shape[0]
    # cv_MatI = cv.resize(cv_MatI, (int(0.1*orig_rows),int(0.1*orig_cols)))
    cv.imshow("test_resized",cv_MatI)
    cv.waitKey(0)

 
    MatI = torch.Tensor(cv_MatI) + 1e-10    # In case it appear zero element
    # print(MatI)
    
    MatI_lt_one = MatI < 1.0
    MatI[MatI_lt_one] = 1.0
    print(MatI)
    
    
    # cv.imshow("tensor to ndarray",np.uint8(MatI))
    # cv.waitKey(0)
    
    temp_str = DIR_NAME + './temp_result/test.png'
    cv.imwrite(temp_str, np.uint8(MatI))
    
    print("Compute:")
    MatD,MatU = preprocess_dn.compute(MatI)
    print(MatD.shape)
    print(MatU.shape)    
    print(MatD)


    
   
    
    # print(f'Shape of MatD:{MatD.shape}, shape of MatU:{MatU.shape}')
    # print(f'MatI:{MatI}')    
    # print(f'MatD:{MatD}')
    # print(f'MatU:{MatU}')


    
    
