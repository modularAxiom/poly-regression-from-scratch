# hw1.py: Implements and runs functions for linear/polynomial regression,
# regularization, cross validation, and graphing
# Date: 2024/03/05

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.model_selection import KFold


# Function to graph training or test data
def graph_data(x, y, title):
    style.use('ggplot')
    plt.scatter(x, y, c="red")
    plt.xlabel('X - axis')
    plt.ylabel('Y - axis')
    plt.title(title)
    plt.autoscale()
    plt.show()
    plt.clf()


# Computes the minimum weight vector of parameters from training data
# feature matrix - used to create a regression model
def compute_weights(x, y):
    w = np.matmul(np.linalg.inv(np.matmul(x.T, x)), np.matmul(x.T, y))
    return w


# Computes the predicted output y from input matrix x and weights w
def predicted_y(x, w):
    y = np.matmul(w.T, x.T)
    return y


# Takes in a feature matrix with 2 columns (the vector x and a column of 1s),
# and outputs its order-d polynomial feature matrix w.r.t. x
def poly_transform(d, x, x_mat):
    xd_mat = np.copy(x_mat)
    for k in range(2, d + 1):
        xd = x**k
        xd_mat = np.column_stack((xd_mat, xd))
    return xd_mat


# Function graphs both data and a regression model of desired degree d
# constructing the model visualization within function using d and w
def graph_model(d, x, y, w, title):
    x_min = np.min(x) - 0.1
    x_max = np.max(x) + 0.1
    model_x = np.linspace(x_min, x_max, 1000)
    model_x1 = np.column_stack((np.ones_like(model_x), model_x))
    if d > 1:
        model_x1 = poly_transform(d, model_x, model_x1)
    model_y = predicted_y(model_x1, w)
    style.use('ggplot')
    plt.plot(model_x, model_y, c="blue")
    plt.scatter(x, y, c="red")
    plt.xlabel('X - axis')
    plt.ylabel('Y - axis')
    plt.title(title)
    plt.autoscale()
    # plt.savefig('graph_' + str(d + 4) + '.pdf')
    plt.show()
    plt.clf()


# Computes the average error of a regression model's predictions for a given dataset
def average_error(prd_y, y):
    errs = prd_y - y
    sqr_errs = errs ** 2
    sum_sqr = np.sum(sqr_errs)
    m = np.size(y)
    err = sum_sqr / m
    return err


# Implements regression of selected degree d, for both training and test data, including
# computing weights, reporting average errors, and graphing both data and models
def regression(d, xtr, ytr, xte, yte):
    # Add column vector of 1s to xtr and xte
    xtr_1 = np.column_stack((np.ones_like(xtr), xtr))
    xte_1 = np.column_stack((np.ones_like(xte), xte))
    # If d = 1, do linear regression
    if d == 1:
        # Compute weights, get dataset predictions, compute average errors
        w = compute_weights(xtr_1, ytr)
        prd_ytr = predicted_y(xtr_1, w)
        err_tr = average_error(prd_ytr, ytr)
        prd_yte = predicted_y(xte_1, w)
        err_te = average_error(prd_yte, yte)
        # Report average errors
        print('LINEAR REGRESSION:', end='\n')
        print('Training Error: ' + str(round(err_tr, 4)), end='\n')
        print('Test Error:     ' + str(round(err_te, 4)), end='\n\n')
        # Graph data and models with rounded average errors in the title
        graph_model(d, xtr, ytr, w, 'Training Data w/ Linear Model (err = ' + str(round(err_tr, 4)) + ')')
        graph_model(d, xte, yte, w, 'Test Data w/ Linear Model (err = ' + str(round(err_te, 4)) + ')')
    # If d is greater than 1, do polynomial regression of degree d
    elif d > 1:
        # Transform 2-D feature matrices to degree d feature matrices, compute weights
        xtr_d = poly_transform(d, xtr, xtr_1)
        xte_d = poly_transform(d, xte, xte_1)
        w = compute_weights(xtr_d, ytr)
        # Get dataset predictions, compute average errors
        prd_ytr = predicted_y(xtr_d, w)
        err_tr = average_error(prd_ytr, ytr)
        prd_yte = predicted_y(xte_d, w)
        err_te = average_error(prd_yte, yte)
        # Report average errors
        print('ORDER-' + str(d) + ' POLYNOMIAL REGRESSION:', end='\n')
        print('Training Error: ' + str(round(err_tr, 4)), end='\n')
        print('Test Error:     ' + str(round(err_te, 4)), end='\n\n')
        # Graph data and models with rounded average errors in the title
        graph_model(d, xtr, ytr, w, 'Training Data w/ Order-' + str(d) + ' Poly. Model '
                                                                         '(err = ' + str(round(err_tr, 4)) + ')')
        graph_model(d, xte, yte, w, 'Test Data w/ Order-' + str(d) + ' Poly. Model ' 
                                                                     '(err = ' + str(round(err_te, 4)) + ')')
    else:
        print('Error: The value d = ' + str(d) + ' is an invalid degree for feature transformation')


# Computes the l2-regularized vector of weight parameters from training
# data feature matrix and the value inputted for lambda
def weights_l2reg(x, y, lamb):
    i_hat = np.identity(np.shape(x)[1], like=x)
    i_hat[0][0] = 0
    w = np.matmul(np.linalg.inv(np.matmul(x.T, x) + lamb * i_hat), np.matmul(x.T, y))
    return w


# Graphs the l2-regularized average errors of both test and training data as
# a function of lambda, using the inputted set of lambda values lams
def graph_errslam(xtr_d, xte_d, ytr, yte, lams, title):
    errs_tr = np.empty_like(lams)
    errs_te = np.empty_like(lams)
    # Compute the average error for the test and training set for each value of lambda
    for i in range(np.size(lams)):
        w = weights_l2reg(xtr_d, ytr, lams[i])
        prd_ytr = predicted_y(xtr_d, w)
        errs_tr[i] = average_error(prd_ytr, ytr)
        prd_yte = predicted_y(xte_d, w)
        errs_te[i] = average_error(prd_yte, yte)
        print('Lambda = ' + str(lams[i]), end='\n')
        print('Training Error: ' + str(round(errs_tr[i], 4)), end='\n')
        print('Test Error:     ' + str(round(errs_te[i], 4)), end='\n\n')
    # Graph the result
    style.use('ggplot')
    plt.plot(lams, errs_tr, c="blue", label='Training')
    plt.plot(lams, errs_te, c="red", label='Test')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.autoscale()
    # plt.savefig('graph_5.pdf')
    plt.show()
    plt.clf()


# Graphs l2-regularized weight parameters as a function of lambda
# for an order-d polynomial feature matrix
def graph_wlam(xtr_d, ytr, lams, title):
    style.use('ggplot')
    wlam = np.empty(shape=(np.size(lams), np.shape(xtr_d)[1]))
    # Fill rows of matrix wlam with weight vectors for each value of lambda
    for i in range(np.size(lams)):
        w = weights_l2reg(xtr_d, ytr, lams[i])
        wlam[i] = w
    # Graph each weight parameter as a function of lambda using column of wlam
    for j in range(np.shape(wlam)[1]):
        lab = 'w' + str(j)
        plt.plot(lams, wlam[:, j], label=lab)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Weight Parameters')
    plt.title(title)
    plt.legend()
    plt.autoscale()
    # plt.savefig('graph_6.pdf')
    plt.show()
    plt.clf()

# def graph_errslam_cv():



if __name__ == '__main__':
    # Training data is loaded into memory and graphed
    x_tr = np.loadtxt('hw1xtr.dat')
    y_tr = np.loadtxt('hw1ytr.dat')
    graph_data(x_tr, y_tr, 'Training Data')

    # Test data is loaded into memory and graphed
    x_te = np.loadtxt('hw1xte.dat')
    y_te = np.loadtxt('hw1yte.dat')
    graph_data(x_te, y_te, 'Test Data')

    # Regression models from linear to degree-4 polynomial are computed and graphed
    for deg in range(1, 5):
        regression(deg, x_tr, y_tr, x_te, y_te)

    # Create array of lambda values, add column vector of 1s to x_tr and x_te
    lambs = np.array([0.01, 0.1, 1, 10, 100, 1000])
    x_tr1 = np.column_stack((np.ones_like(x_tr), x_tr))
    x_te1 = np.column_stack((np.ones_like(x_te), x_te))

    # Transform x_tr1 and x_te1 to order-4 polynomial feature matrices
    x_tr4 = poly_transform(4, x_tr, x_tr1)
    x_te4 = poly_transform(4, x_te, x_te1)

    # Graph l2-regularized 4th order polynomial regression training and test errors as a function of lambda
    # and report errors per value of lambda
    print("L2-REGULARIZED ORDER-4 POlYNOMIAL REGRESSION:")
    graph_errslam(x_tr4, x_te4, y_tr, y_te, lambs, 'Errors of Ord-4 Poly. Model for Values of Lambda')
    # Graph l2-regularized 4th order polnomial regression weight parameters as a function of lambda
    graph_wlam(x_tr4, y_tr, lambs, 'W. Params. of Ord-4 Poly. Mod. for Vals. of Lambda')

    # Cross Validation on the training set with l2-regularized 4th order polynomial model for each value of lambda
    print("CROSS VALIDATED L2-REGULARIZED ORDER-4 POlYNOMIAL REGRESSION:")
    errslamb_cv = np.empty_like(lambs)
    for l in range(np.size(lambs)):
        total_err = 0
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(x_tr4, y_tr):
            x_trn, y_trn = x_tr4[train_index], y_tr[train_index]
            x_tes, y_tes = x_tr4[test_index], y_tr[test_index]
            # Compute 4th order regularized weights and fold error on validation set
            # then add it to total validation error
            w = weights_l2reg(x_trn, y_trn, lambs[l])
            prd_y = predicted_y(x_tes, w)
            fold_err = average_error(prd_y, y_tes)
            total_err += fold_err
        # Compute and store average validation error for each value l of lambda
        errslamb_cv[l] = total_err / kf.get_n_splits()

    # Get best value of lambda and the index of the corresponding min error
    min_err = np.min(errslamb_cv)
    best_cv_lamb = np.argmin(errslamb_cv)
    print('Best Lambda = ' + str(lambs[best_cv_lamb]), end='\n')
    print('Average Validation Error = ' + str(round(min_err, 4)), end='\n\n')

    # Graph average l2-reg 4th order polynomial validation error for values of lambda
    style.use('ggplot')
    plt.plot(lambs, errslamb_cv, c="red")
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('Validation Error')
    plt.title('5-Fold C.V. Errors of L2-reg Ord-4 Poly.\n Model for Vals. of Lambda')
    plt.autoscale()
    # plt.savefig('graph_7.pdf')
    plt.show()
    plt.clf()

    # As seen from the previous l2-reg errors on the test set, the best value for lambda on the test set
    # is still when lambda = 0.1, as the error is lower than it is for lambda = 0.01 on the test set
    # Thus, the test data and that model are graphed below
    best_lamb = 0.1
    w = weights_l2reg(x_te4, y_te, best_lamb)
    graph_model(4, x_te, y_te, w, 'Test Data w/ L2-reg. Ord-4 Poly. '
                                  '\n Model s.t. Lambda = 0.1 (err = ' + str(0.057) + ')')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
