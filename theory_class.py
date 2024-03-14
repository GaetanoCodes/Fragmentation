import numpy as np
import sympy as syp
import scipy.special as sp
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression


class FragmentationTheory:
    def __init__(self, lambda_, alpha, step, r, t_max, derivativeOrder=150, sigma=1):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.step = step
        self.derivativeOrder = derivativeOrder
        self.r = r
        self.t_max = t_max
        self.sigma = sigma

    def timeInitialisation(self):
        self.timeVector = np.linspace(0, self.t_max, int(self.t_max / self.step) + 1)
        self.timeVectorDilated = self.lambda_ ** (-self.alpha) * self.timeVector
        print(
            f"(*) Time Vector from 0 to {self.t_max} with a {self.step} step has been built."
        )

    def pochhamerCoefficients(self):
        powers = np.arange(0, self.derivativeOrder)
        lambdaV = (self.lambda_ ** (-self.alpha)) * np.ones(self.derivativeOrder)
        lambdaV = np.power(lambdaV, powers)
        lambdaV = self.lambda_ ** (2 - self.sigma) * lambdaV - 1
        lambdaV = np.cumprod(lambdaV)
        lambdaV = np.hstack(([1], lambdaV))

        self.pochhamerVector = lambdaV
        print(f"(*) Pochhamer coefficients vector has been built.")

    def deathRateDerivative(self):
        # création du vecteur des dérivées
        t = syp.symbols("t")
        int_t = self.r * t
        expr = syp.exp(-int_t)
        derivees_0 = [1]
        for i in range(1, self.derivativeOrder + 1):
            d = syp.diff(expr, t, 1)
            f = syp.lambdify(t, d * syp.exp(int_t), "numpy")
            expr = d
            derivees_0.append(f(0))
        derivees_0 = np.array(derivees_0)  # derive i

        mat_derivees = np.repeat(derivees_0[None, :], self.derivativeOrder + 1, axis=0)

        for i in range(1, self.derivativeOrder + 1):
            mat_derivees[i, :] = np.roll(derivees_0[::-1], shift=1 + i)
        mat_derivees = np.tril(mat_derivees)

        self.deathRateMatrix = mat_derivees
        print(f"(*) Death rate matrix has been built.")

    def binomialMatrix(self):
        # création de la matrice des coefs binomiaux
        k = np.repeat(
            np.arange(self.derivativeOrder + 1)[None, :],
            self.derivativeOrder + 1,
            axis=0,
        )
        n = np.repeat(
            np.arange(self.derivativeOrder + 1)[:, None],
            self.derivativeOrder + 1,
            axis=1,
        )
        self.binomMatrix = sp.binom(n, k)
        print(f"(*) Binomial coefficients matrix has been built.")

    def expansion(self):
        serieExpansion = self.pochhamerVector * self.binomMatrix * self.deathRateMatrix
        powerTime = np.power(
            np.repeat(self.timeVector[:, None], self.derivativeOrder + 1, axis=1),
            np.arange(self.derivativeOrder + 1),
        )
        factorial = sp.factorial(np.arange(self.derivativeOrder + 1))
        powerTimeFactorial = powerTime / factorial
        self.theoryVector = np.einsum(
            "ab, bc -> acb", powerTimeFactorial, serieExpansion
        ).sum(axis=(2, 1))

        powerTime = np.power(
            np.repeat(
                self.timeVectorDilated[:, None], self.derivativeOrder + 1, axis=1
            ),
            np.arange(self.derivativeOrder + 1),
        )
        factorial = sp.factorial(np.arange(self.derivativeOrder + 1))
        powerTimeFactorial = powerTime / factorial
        self.theoryVectorDilated = np.einsum(
            "ab, bc -> acb", powerTimeFactorial, serieExpansion
        ).sum(axis=(2, 1))
        ##
        # self.ptsseries = np.einsum(
        #     "ab, bc -> acb", powerTimeFactorial, serieExpansion
        # ).sum(axis=2)[1000]
        # print(np.einsum('ab, bc -> acb', powerTimeFactorial, serieExpansion))
        print(f"(*) Theory vector has been built.")

    def maximum(self):
        self.maximum_ = self.timeVector[np.argmax(self.theoryVector)]

    def construction(self):
        print("####### Construction ######")
        self.timeInitialisation()
        self.pochhamerCoefficients()
        self.deathRateDerivative()
        self.binomialMatrix()
        self.expansion()
        self.maximum()
        print("#### END Construction #####")

    def plotResult(self):
        plt.plot(self.timeVector, self.theoryVector)
        plt.show()


if __name__ == "__main__":
    f = FragmentationTheory(
        lambda_=4, alpha=4, step=0.1, r=0.0, derivativeOrder=250, t_max=40, sigma=1.5
    )
    f.construction()

    # g = FragmentationTheory(
    #     lambda_=4, alpha=0.7, step=0.1, r=0.0, derivativeOrder=200, t_max=40, sigma=3
    # )
    # g.construction()
    # h = FragmentationTheory(
    #     lambda_=4, alpha=1.5, step=0.1, r=0.0, derivativeOrder=200, t_max=40, sigma=3
    # )
    # h.construction()
    eps = 0.5
    equiv_epssup = f.timeVector ** ((2 - f.sigma + eps) / f.alpha)
    equiv_epsinf = f.timeVector ** ((2 - f.sigma - eps) / f.alpha)
    equiv = f.timeVector ** ((2 - f.sigma) / f.alpha)
    plt.plot(f.timeVector, f.theoryVector / equiv_epssup, label="sup")
    plt.plot(f.timeVector, f.theoryVector / equiv_epsinf, label="inf")
    plt.plot(f.timeVector, f.theoryVector / equiv, label="True ratio")

    plt.grid()

    plt.legend()

    plt.show()
