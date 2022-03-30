"""
TVMax implementation. Mostly copy-paste from:
https://github.com/deep-spin/TVmax/blob/main/core/model/tv2d_numba.py
"""
import numpy as np
import warnings
from numba import njit, jit

import torch
import torch.nn as nn
from torch.autograd import Function

@njit
def _prox_tv1d(step_size, input, output):
    """low level function call, no checks are performed"""
    width = input.size + 1
    index_low = np.zeros(width, dtype=np.int32)
    slope_low = np.zeros(width, dtype=input.dtype)
    index_up  = np.zeros(width, dtype=np.int32)
    slope_up  = np.zeros(width, dtype=input.dtype)
    index     = np.zeros(width, dtype=np.int32)
    z         = np.zeros(width, dtype=input.dtype)
    y_low     = np.empty(width, dtype=input.dtype)
    y_up      = np.empty(width, dtype=input.dtype)
    s_low, c_low, s_up, c_up, c = 0, 0, 0, 0, 0
    y_low[0] = y_up[0] = 0
    y_low[1] = input[0] - step_size
    y_up[1] = input[0] + step_size
    incr = 1

    for i in range(2, width):
        y_low[i] = y_low[i-1] + input[(i - 1) * incr]
        y_up[i] = y_up[i-1] + input[(i - 1) * incr]

    y_low[width-1] += step_size
    y_up[width-1] -= step_size
    slope_low[0] = np.inf
    slope_up[0] = -np.inf
    z[0] = y_low[0]

    for i in range(1, width):
        c_low += 1
        c_up += 1
        index_low[c_low] = index_up[c_up] = i
        slope_low[c_low] = y_low[i]-y_low[i-1]
        while (c_low > s_low+1) and (slope_low[max(s_low, c_low-1)] <= slope_low[c_low]):
            c_low -= 1
            index_low[c_low] = i
            if c_low > s_low+1:
                slope_low[c_low] = (y_low[i]-y_low[index_low[c_low-1]]) / (i-index_low[c_low-1])
            else:
                slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])

        slope_up[c_up] = y_up[i]-y_up[i-1]
        while (c_up > s_up+1) and (slope_up[max(c_up-1, s_up)] >= slope_up[c_up]):
            c_up -= 1
            index_up[c_up] = i
            if c_up > s_up + 1:
                slope_up[c_up] = (y_up[i]-y_up[index_up[c_up-1]]) / (i-index_up[c_up-1])
            else:
                slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

        while (c_low == s_low+1) and (c_up > s_up+1) and (slope_low[c_low] >= slope_up[s_up+1]):
            c += 1
            s_up += 1
            index[c] = index_up[s_up]
            z[c] = y_up[index[c]]
            index_low[s_low] = index[c]
            slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])
        while (c_up == s_up+1) and (c_low>s_low+1) and (slope_up[c_up]<=slope_low[s_low+1]):
            c += 1
            s_low += 1
            index[c] = index_low[s_low]
            z[c] = y_low[index[c]]
            index_up[s_up] = index[c]
            slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])

    for i in range(1, c_low - s_low + 1):
        index[c+i] = index_low[s_low+i]
        z[c+i] = y_low[index[c+i]]
    c = c + c_low-s_low
    j, i = 0, 1
    while i <= c:
        a = (z[i]-z[i-1]) / (index[i]-index[i-1])
        while j < index[i]:
            output[j * incr] = a
            output[j * incr] = a
            j += 1
        i += 1
    return


@njit
def prox_tv1d_cols(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along columns of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_cols):
        _prox_tv1d(stepsize, A[:, i], out[:, i])
    return out.ravel()


@njit
def prox_tv1d_rows(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along rows of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_rows):
        _prox_tv1d(stepsize, A[i, :], out[i, :])
    return out.ravel()


def c_prox_tv2d(step_size, x, n_rows, n_cols, max_iter, tol):
    """
    Douglas-Rachford to minimize a 2-dimensional total variation.
    Reference: https://arxiv.org/abs/1411.0589
    """
    n_features = n_rows * n_cols
    p = np.zeros(n_features)
    q = np.zeros(n_features)

    for it in range(max_iter):
        y = x + p
        y = prox_tv1d_cols(step_size, y, n_rows, n_cols)
        p += x - y
        x = y + q
        x = prox_tv1d_rows(step_size, x, n_rows, n_cols)
        q += y - x

        # check convergence
        accuracy = np.max(np.abs(y - x))
        if accuracy < tol:
            break
    else:
        warnings.warn("prox_tv2d did not converged to desired accuracy\n" +
                      "Accuracy reached: %s" % accuracy)
    return x


def prox_tv2d(w, step_size, n_rows, n_cols, max_iter=500, tol=1e-2):
    """
    Computes the proximal operator of the 2-dimensional total variation operator.
    This solves a problem of the form
         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2
    where TV(x) is the two-dimensional total variation. It does so using the
    Douglas-Rachford algorithm [Barbero and Sra, 2014].
    Parameters
    ----------
    w: array
        vector of coefficients
    step_size: float
        step size (often denoted gamma) in proximal objective function
    max_iter: int
    tol: float
    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)
    Barbero, Alvaro, and Suvrit Sra. "Modular proximal optimization for
    multidimensional total-variation regularization." arXiv preprint
    arXiv:1411.0589 (2014).
    """

    x = w.copy().astype(np.float64)
    return c_prox_tv2d(step_size, x, n_rows, n_cols, max_iter, tol)


@jit(nopython=True)
def isin(x, l):
    for i in l:
        if x==i:
            return True
    return False

@jit(nopython=True)        
def back(Y, dX, dY, size):
    neigbhours=list([(1,1)])
    del neigbhours[-1] 
    group=[(0,0)]
    del group[-1]
    n=0
    idx_grouped = [(size,size) for x in range(size)]
    count=0
    value=0
    s=0

    while True:
        if len(neigbhours)!=0:
            while len(neigbhours)!=0:
                if Y[neigbhours[0][0],neigbhours[0][1]] == value:
                    a = neigbhours[0][0]
                    b = neigbhours[0][1]
                    del neigbhours[0]
                    count+=1
                    s+=dY[a,b]
                    group.append((a,b))
                    idx_grouped[n]=(a,b)
                    n+=1
                    if b<dX.shape[1]-1 and isin((a,b+1), idx_grouped)==False and isin((a,b+1), neigbhours)==False:
                        neigbhours.append((a,b+1))
                    if a<dX.shape[0]-1 and isin((a+1,b), idx_grouped)==False and isin((a+1,b), neigbhours)==False:
                        neigbhours.append((a+1,b)) 
                    if b>0 and isin((a,b-1), idx_grouped)==False and isin((a,b-1), neigbhours)==False:
                        neigbhours.append((a,b-1)) 
                    if a>0 and isin((a-1,b), idx_grouped)==False and isin((a-1,b), neigbhours)==False:
                        neigbhours.append((a-1,b)) 
                else:
                    del neigbhours[0]
        else:
            if len(group)>0:
                o=s/count
                count=0
                for x in group:
                    dX[x[0],x[1]]=o
                group=[(0,0)]
                del group[0]
            
            if n>=size:
                break
            B=False
            for i in range(dX.shape[0]):
                for j in range(dX.shape[1]):
                    if isin((i,j), idx_grouped)==False:
                        value = Y[i,j]
                        s = dY[i,j]
                        count+=1
                        group.append((i, j))
                        idx_grouped[n] = (i, j)
                        n+=1
                        if j<dX.shape[1]-1 and isin((i,j+1), idx_grouped)==False and isin((i,j+1), neigbhours)==False:
                            neigbhours.append((i,j+1))
                        if i<dX.shape[0]-1 and isin((i+1,j), idx_grouped)==False and isin((i+1,j), neigbhours)==False:
                            neigbhours.append((i+1,j)) 
                        if j>0 and isin((i,j-1), idx_grouped)==False and isin((i,j-1), neigbhours)==False:
                            neigbhours.append((i,j-1)) 
                        if i>0 and isin((i-1,j), idx_grouped)==False and isin((i-1,j), neigbhours)==False:
                            neigbhours.append((i-1,j)) 
                        B=True
                        break
                if B:
                    break
    return dX

@jit(nopython=True)      
def back_generic(Y, A, dX, dY):
    neigbhours=list([0])
    del neigbhours[0]
    group=list([0])
    del group[0]
    idx_grouped=list([0])
    del idx_grouped[0]
    value=0
    s=0
    while True:
        if len(neigbhours)!=0:
            while len(neigbhours)!=0:
                if Y[neigbhours[0]] == value:
                    a = neigbhours[0]
                    del neigbhours[0]
                    s+=dY[a]
                    group.append(a)
                    idx_grouped.append(a)
                    for b in range(a):
                        if A[a,b]==1 and isin(b, idx_grouped)==False and isin(b, neigbhours)==False:
                            neigbhours.append(b)
                else:
                    del neigbhours[0]
        else:
            if len(group)>0:
                o=s/len(group)
                for x in group:
                    dX[x]=o
                group=list([0])
                del group[0]
            
            if len(idx_grouped)>=len(Y):
                break
            B=False
            for i in range(len(Y)-1,-1,-1):

                if isin(i, idx_grouped)==False:
                    value = Y[i]
                    s = dY[i]    
                    group.append(i)
                    idx_grouped.append(i)
                    for j in range(i):
                        if A[i,j]==1 and isin(j, idx_grouped)==False and isin(j, neigbhours)==False:
                            neigbhours.append(j)
                    B=True
                    break
                if B:
                    break
    return dX


class TV2DFunction(Function):

    @staticmethod
    def forward(ctx, X, alpha=0.1, max_iter=35, tol=1e-2):
        torch.set_num_threads(8)
        ctx.digits_tol = int(-np.log10(tol)) // 2

        X_np = X.detach().cpu().numpy()
        n_rows, n_cols = X_np.shape
        Y_np = prox_tv2d(X_np.ravel(),
                         step_size=alpha / 2,
                         n_rows=n_rows,
                         n_cols=n_cols,
                         max_iter=max_iter,
                         tol=tol)
        

        Y_np = Y_np.reshape(n_rows, n_cols)
        Y = torch.from_numpy(Y_np)  # double-precision
        Y = torch.as_tensor(Y, dtype=X.dtype, device=X.device)
        ctx.save_for_backward(Y.detach()) 

        return Y

    @staticmethod
    def backward(ctx, dY):
        torch.set_num_threads(8)
        Y, = ctx.saved_tensors

        Y_np = np.array(Y.cpu()).round(ctx.digits_tol)
        dY_np = np.array(dY.cpu())
        dX = np.zeros((dY.size(0),dY.size(1)))

        dX = back(Y_np, dX, dY_np,dX.shape[0]*dX.shape[1])
        dX = torch.as_tensor(dX, dtype=dY.dtype, device=dY.device)

        return dX, None 


_tv2d = TV2DFunction.apply


class TV2D(nn.Module):
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-12) -> None:
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    
    def forward(self, X):
        return _tv2d(X, self.alpha, self.max_iter, self.tol)
