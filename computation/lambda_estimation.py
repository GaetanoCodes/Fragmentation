import numpy as np
import computation.simulation_class as simulation
import computation.theory_class as theory
import scipy.special as sp
import matplotlib.pyplot as plt


class LambdaEstimation:
    def __init__(self, r, alpha, theoryVector, timeVector):
        self.theoryVector = theoryVector
        self.timeVector = timeVector
        self.r = r
        self.alpha = alpha

    def basicMatrixes(self, order):
        tK = np.power(
            np.repeat(self.timeVector[:, None], order + 1, axis=1),
            np.arange(order + 1),
        )
        kFact = sp.factorial(np.arange(order + 1))
        tFact = tK / kFact
        # to be changed with r derivative matrix
        return tFact

    def derivativeLambdaVector(self, lambda_, order):
        premierTerme = 1 - self.alpha * np.arange(order + 1)
        deuxiemeTerme = np.power(
            lambda_ * np.ones(order + 1), -self.alpha * np.arange(order + 1)
        )
        troisiemeTerme = (
            np.power(
                lambda_ * np.ones(order + 1), 1 - self.alpha * np.arange(order + 1)
            )
        ) - 1
        somme = premierTerme * deuxiemeTerme / troisiemeTerme
        # vecteur de la somme

        if np.isnan(somme).sum() > 0:
            zeroIndex = int(np.argwhere(np.isnan(somme))[0, 0])
            somme[zeroIndex:] = 0
        somme = somme[None, :]

        mask = np.tril(np.ones((order + 1, order + 1)))

        vectorAi = self.lambdaVector(lambda_, order)[:, None]
        finalVector = (mask * vectorAi * somme).sum(axis=1)
        return finalVector

    def lambdaVector(self, lamb, order):
        lambdaVect = np.cumprod(
            np.power(lamb * np.ones(order + 1), 1 - self.alpha * np.arange(order + 1))
            - 1
        )
        lambdaVect = np.append(np.ones(1), lambdaVect[:-1])
        return lambdaVect

    def lambdaEstimation(self, order):
        L = []
        l = []
        basicMatrix = self.basicMatrixes(order)
        for lamb_ in np.linspace(1.01, 10, 100):
            membreGauche = (
                self.derivativeLambdaVector(lamb_, order)
                .dot(basicMatrix.T)
                .dot(self.theoryVector)
            )
            membreDroit = (
                self.derivativeLambdaVector(lamb_, order)
                .dot(basicMatrix.T)
                .dot(basicMatrix)
                .dot(self.lambdaVector(lamb_, order))
            )
            l.append(lamb_)
            L.append(membreGauche - membreDroit)

        return l[np.argmin(np.abs(L))]

    def lambdaEstimationVectorOrder(self, orders):
        lambdaEstimatedList = []
        for order in orders:
            lambdaEstimated = self.lambdaEstimation(order)
            lambdaEstimatedList.append(lambdaEstimated)
        return lambdaEstimatedList

    def resultHandler(self, orders, mode="plot", path="", trueLambda=None):
        lambdaEstimatedList = self.lambdaEstimationVectorOrder(orders)
        title = rf"Lambda estimation for different orders and true lambda (line)"
        fontdict = {"size": 14}
        plt.figure(figsize=(10, 5))
        plt.scatter(orders, lambdaEstimatedList, marker=".", label="Lambda estimation")
        # params
        plt.title(title, fontdict=fontdict)
        plt.xlabel("Orders", fontdict=fontdict)
        plt.ylabel("Lambda Estimation", fontdict=fontdict)
        plt.xlim((0, orders[-1] + 10))

        if trueLambda is not None:
            plt.hlines(
                trueLambda,
                xmin=0,
                xmax=orders[-1] + 10,
                label="True Lambda",
                color="red",
            )
            plt.ylim((0, max(1.5 * trueLambda, max(lambdaEstimatedList))))
            plt.legend()
        if mode == "plot":
            plt.legend()
            plt.show()

        elif mode == "save":
            plt.legend()
            plt.savefig(path)
