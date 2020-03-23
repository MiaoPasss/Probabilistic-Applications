import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)

    Inputs:
    ------
    beta: hyperparameter in the proir distribution

    Outputs: None
    -----
    """
    mean = np.array([0, 0])
    cov = np.array([[beta, 0], [0, beta]])

    h = np.linspace(-1, 1, 1000)
    v = np.linspace(-1, 1, 1000)

    H, V = np.meshgrid(h, v)
    prob = []

    for i in range(1000):
        data_point = np.concatenate((h.reshape(1000,1), V[i].reshape(1000,1)), 1)
        prob.append(util.density_Gaussian(mean, cov, data_point))


    plt.figure()
    plt.grid()
    plt.title("Prior Distribution")
    plt.plot([-0.1], [-0.5], marker='o', markersize=5, color='red')
    plt.contour(H, V, prob)
    plt.xlabel('$a_{0}$')
    plt.ylabel('$a_{1}$')
    plt.savefig("prior.pdf")
    plt.close()

    return

def posteriorDistribution(x,z,beta,sigma2,ns=1):
    """
    Plot the contours of the posterior distribution p(a|x,z)

    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise

    Outputs:
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    N = len(x)
    X = np.concatenate((np.ones((N, 1)), x.reshape(N, 1)), 1)

    mu = np.linalg.inv(X.T @ X + sigma2 * np.identity(2)/beta) @ X.T @ z.reshape(N, 1)
    Cov = sigma2 * np.linalg.inv(X.T @ X + sigma2 * np.identity(2)/beta)

    h = np.linspace(-1, 1, 1000)
    v = np.linspace(-1, 1, 1000)

    H, V = np.meshgrid(h, v)
    prob = []

    for i in range(1000):
        data_point = np.concatenate((h.reshape(1000,1), V[i].reshape(1000,1)), 1)
        prob.append(util.density_Gaussian(np.squeeze(mu), Cov, data_point))


    plt.figure()
    plt.grid()
    plt.title("Posterior Distribution with sample size {}".format(ns))
    plt.plot([-0.1], [-0.5], marker='o', markersize=5, color='red')
    plt.contour(H, V, prob)
    plt.xlabel('$a_{0}$')
    plt.ylabel('$a_{1}$')
    plt.savefig("posterior{}.pdf".format(ns))
    plt.close()

    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train, ns=1):
    """
    Make predictions for the inputs in x, and plot the predicted results

    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot

    Outputs: None
    -----
    """
    prediction = []
    yerr = []

    for i in x:
        x_new = np.array([1, i]).reshape(2,1)
        prediction.append(mu.T @ x_new)
        yerr.append(x_new.T @ Cov @ x_new + sigma2)

    plt.figure()
    plt.grid()
    plt.title("Prediction with training sample size {}".format(ns))
    plt.scatter(x_train,z_train, color='red', s=10)
    plt.errorbar(x, np.squeeze(np.array(prediction)), yerr=np.squeeze(np.array(yerr)), color='blue')
    plt.legend(['training samples', 'predictions'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("predict{}.pdf".format(ns))
    plt.close()

    return

if __name__ == '__main__':

    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction
    x_test = [x for x in np.arange(-4,4.01,0.2)]

    # known parameters
    sigma2 = 0.1
    beta = 1

    # number of training samples used to compute posterior

    # used samples
    x = x_train[0:1]
    z = z_train[0:1]

    x5 = x_train[0:5]
    z5 = z_train[0:5]

    x100 = x_train[0:100]
    z100 = z_train[0:100]

    # prior distribution p(a)
    priorDistribution(beta)

    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    mu5, Cov5 = posteriorDistribution(x5,z5,beta,sigma2, 5)
    mu100, Cov100 = posteriorDistribution(x100,z100,beta,sigma2, 100)

    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    predictionDistribution(x_test,beta,sigma2,mu5,Cov5,x5,z5, 5)
    predictionDistribution(x_test,beta,sigma2,mu100,Cov100,x100,z100, 100)
