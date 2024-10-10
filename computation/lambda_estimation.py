import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

import scienceplots

plt.style.use("science")


class LambdaEstimation:
    def __init__(self, r, alpha, theory_vector, time_vector):
        self.theory_vector = theory_vector
        self.time_vector = time_vector
        self.r = r
        self.alpha = alpha

    def basic_matrices(self, order):
        tK = np.power(
            np.repeat(self.time_vector[:, None], order + 1, axis=1),
            np.arange(order + 1),
        )
        k_fact = sp.factorial(np.arange(order + 1))
        t_fact = tK / k_fact
        # to be changed with r derivative matrix
        return t_fact

    def derivative_lambda_vector(self, lambda_, order):
        premier_terme = 1 - self.alpha * np.arange(order + 1)
        deuxieme_terme = np.power(
            lambda_ * np.ones(order + 1), -self.alpha * np.arange(order + 1)
        )
        troisieme_terme = (
            np.power(
                lambda_ * np.ones(order + 1), 1 - self.alpha * np.arange(order + 1)
            )
        ) - 1
        somme = premier_terme * deuxieme_terme / troisieme_terme
        # vecteur de la somme

        if np.isnan(somme).sum() > 0:
            zeroIndex = int(np.argwhere(np.isnan(somme))[0, 0])
            somme[zeroIndex:] = 0
        somme = somme[None, :]

        mask = np.tril(np.ones((order + 1, order + 1)))

        vector_ai = self.lambdaVector(lambda_, order)[:, None]
        final_vector = (mask * vector_ai * somme).sum(axis=1)
        return final_vector

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
        basicMatrix = self.basic_matrices(order)
        for lamb_ in np.linspace(1.01, 10, 100):
            membreGauche = (
                self.derivative_lambda_vector(lamb_, order)
                .dot(basicMatrix.T)
                .dot(self.theory_vector)
            )
            membreDroit = (
                self.derivative_lambda_vector(lamb_, order)
                .dot(basicMatrix.T)
                .dot(basicMatrix)
                .dot(self.lambdaVector(lamb_, order))
            )
            l.append(lamb_)
            L.append(membreGauche - membreDroit)

        return l[np.argmin(np.abs(L))]

    def lambda_estimation_vector_order(self, orders):
        lambda_estimated_list = []
        for order in orders:
            lambdaEstimated = self.lambdaEstimation(order)
            lambda_estimated_list.append(lambdaEstimated)
        return lambda_estimated_list

    def result_handler(self, orders, mode="plot", path="", true_lambda=None):
        lambda_estimated_list = self.lambda_estimation_vector_order(orders)
        title = rf"Lambda estimation for different orders and true lambda (line)"
        fontdict = {"size": 14}
        plt.figure(figsize=(10, 5))
        plt.scatter(
            orders, lambda_estimated_list, marker=".", label="Lambda estimation"
        )
        # params
        plt.title(title, fontdict=fontdict)
        plt.xlabel("Orders", fontdict=fontdict)
        plt.ylabel("Lambda Estimation", fontdict=fontdict)
        plt.xlim((0, orders[-1] + 10))
        plt.grid()
        if true_lambda is not None:
            plt.hlines(
                true_lambda,
                xmin=0,
                xmax=orders[-1] + 10,
                label="True Lambda",
                color="red",
            )
            plt.ylim((0, max(1.5 * true_lambda, max(lambda_estimated_list))))
            plt.legend()
        if mode == "plot":
            plt.legend()
            plt.show()

        elif mode == "save":
            plt.legend()
            plt.savefig(path)
