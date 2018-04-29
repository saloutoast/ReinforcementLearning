import autograd.numpy as np
from autograd import value_and_grad 
from autograd.misc.flatten import flatten_func
from timeit import default_timer as timer
import time

# gradient descent
def gradient_descent(g,w,a_train,s_train,alpha,max_its,verbose): 
    '''
    A basic gradient descent module (full batch) for system identification training.  
    Inputs to gradient_descent function:
    
    g - function to minimize
    w - initial weights
    a_train - training action sequence
    s_train - training state sequence
    alpha - steplength / learning rate
    max_its - number of iterations to perform
    verbose - print out update each step if verbose = True
    '''
    
    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w)
    grad = value_and_grad(g_flat)

    # record history
    num_val = y_val.size
    w_hist = [unflatten(w)]
    train_hist = [g_flat(w,a_train,s_train)]
        
    # over the line
    alpha_choice = 0
    for k in range(1,max_its+1): 
        # take a single descent step
        start = timer()

        # plug in value into func and derivative
        cost_eval,grad_eval = grad(w,a_train,s_train)
        grad_eval.shape = np.shape(w)
    
        # take descent step with momentum
        w = w - alpha*grad_eval
        
        end = timer()
        
        # update training and validation cost
        train_cost = g_flat(w,a_train,s_train)
        val_cost = np.nan

        # record weight update, train cost
        w_hist.append(unflatten(w))
        train_hist.append(train_cost)

        if verbose == True:
            print ('step ' + str(k+1) + ' done in ' + str(np.round(end - start,1)) + ' secs, train cost = ' + str(np.round(train_hist[-1],4)[0]) ) 

    if verbose == True:
        print ('finished all ' + str(max_its) + ' steps')
    return w_hist,train_hist,val_hist
