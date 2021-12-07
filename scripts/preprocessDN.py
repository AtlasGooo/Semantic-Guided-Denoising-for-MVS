

import numpy
from torch.autograd.grad_mode import F
from utils import *



class PreprocessDN(nn.Module):
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
    
    # sys.path.append(DIR_NAME)
    preprocess_dn = PreprocessDN()     
    

    cv_MatI = cv.imread('./images/d_image_2_000000.png',cv.IMREAD_GRAYSCALE)
    print(f'origin cv_MatI: {cv_MatI}')
    # cv.imshow("original mat",cv_MatI)
    
    cols = cv_MatI.shape[1] # x
    rows = cv_MatI.shape[0] # y
    print(f'rows:{rows} cols:{cols} \n')
    
    # cv_MatI = cv.resize(cv_MatI, (int(0.1*orig_rows),int(0.1*orig_cols)))
    rowmin = int(0.40*rows)
    rowmax = int(0.60*rows)
    colmin = int(0.40*cols)
    colmax = int(0.60*cols)
    

    cv_MatI = np.array(np.uint8(cv_MatI))
    print(cv_MatI.shape)
    cv_MatI = cv_MatI[rowmin:rowmax , colmin:colmax] 
    print(f'numpy cv_MatI[]: {cv_MatI} \n')
    
    cv.imshow("show middle",cv_MatI)    
    cv.waitKey(0)
    
    torch_MatI = torch.Tensor(cv_MatI)
    
    torch_to_cv = np.uint8(torch_MatI)
    cv.imshow("torch to cv",torch_to_cv)
    cv.waitKey(0)
    
    
    
    
    
    
    # cv.imshow("test_resized",cv_MatI)
    # cv.waitKey(0)

 
    # MatI = torch.Tensor(cv_MatI) + 1e-10    # In case it appear zero element
    # # print(MatI)
    
    # MatI_lt_one = MatI < 1.0
    # MatI[MatI_lt_one] = 1.0
    # print(f'MatI after preprocess: {MatI} \n')
    
    
    # # cv.imshow("tensor to ndarray",np.uint8(MatI))
    # # cv.waitKey(0)
    
    # temp_str = DIR_NAME + './temp_result/test.png'
    # cv.imwrite(temp_str, np.uint8(MatI))
    
    # print("Compute:")
    # MatD,MatU = preprocess_dn.compute(MatI)
    # print(f'MatD.shape: {MatD.shape} \n')
    # print(f'MatU.shape: {MatU.shape} \n')    
    # print(f'MatD: {MatD} \n')
    # print(f'MatUx: {MatU[:,:,0]} \n')


    
   
    
    # print(f'Shape of MatD:{MatD.shape}, shape of MatU:{MatU.shape}')
    # print(f'MatI:{MatI}')    
    # print(f'MatD:{MatD}')
    # print(f'MatU:{MatU}')


    
    
