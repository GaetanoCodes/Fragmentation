import numpy as np
import simulation_class as simulation
import theory_class as theory
import scipy.special as sp


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
