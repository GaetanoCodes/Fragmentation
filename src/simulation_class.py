import numpy as np
import matplotlib.pyplot as plt
from src.theory_class import FragmentationTheory
import tqdm

# import time
import scienceplots

plt.style.use("science")


class FragmentationSingleSimulation:
    def __init__(self, lambda_, alpha, r, t_max, step):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.t_max = t_max
        self.r = r
        self.step = step

    def timeInitialisation(self):
        self.time_vector = np.linspace(
            0, self.t_max, int(self.t_max / self.step) + 1)

    def simulation(self):
        massInit = np.array([1])
        mass = massInit
        demography = []
        for t in self.time_vector:
            demography.append(len(mass))
            random = np.random.random(size=mass.shape)
            deadParticules = random < self.step * self.r
            random = random[~deadParticules]
            random = np.random.random(
                int((~deadParticules).sum())
            )  # to be changed with new random of new sizes
            mass = mass[~deadParticules]
            fragMask = random <= self.step * (mass**self.alpha)
            massNoFrag = mass[~fragMask]
            newMass = np.repeat(mass[fragMask] / self.lambda_, self.lambda_)
            mass = np.hstack((massNoFrag, newMass))

        self.finalMass = mass
        self.demography = demography

    def plotResult(self):
        plt.plot(self.time_vector, self.demography)
        plt.show()


class Fragmentation_simulation:
    def __init__(self, lambda_, alpha, r, t_max, step, n_simul):
        self.paramsSingle = dict(
            lambda_=lambda_, alpha=alpha, r=r, t_max=t_max, step=step
        )
        self.n_simul = n_simul
        self.t_max = t_max
        self.step = step

    def timeInitialisation(self):
        self.time_vector = np.linspace(
            0, self.t_max, int(self.t_max / self.step) + 1)
        print(
            f"(*) Time Vector from 0 to {self.t_max} with a {self.step} step has been built."
        )

    def monteCarloSimulation(self):
        L = []
        for n_simul in tqdm.tqdm(range(self.n_simul)):
            f = FragmentationSingleSimulation(**self.paramsSingle)
            f.timeInitialisation()
            f.simulation()
            L.append(f.demography)
            # print(n_simul)

        self.simulations = np.array(L)

    def getStatistics(self):
        self.meanResult = np.mean(self.simulations, axis=0)
        self.stdEstimate = (
            (np.sum(self.simulations**2, axis=0) -
             self.n_simul * self.meanResult**2)
            ** (0.5)
        ) / self.n_simul
        self.quantile5 = self.meanResult - 1.95 * self.stdEstimate
        self.quantile95 = self.meanResult + 1.95 * self.stdEstimate
        fragmentationTh = FragmentationTheory(**self.paramsSingle)
        fragmentationTh.construction()
        self.theory_vector = fragmentationTh.theory_vector

    def result_handler(self, mode="plot", path=""):
        # plt.rcParams["text.usetex"] = True
        mainlabel = "Mean"
        if self.n_simul == 1:
            mainlabel = "Realisation"

        lambda_ = self.paramsSingle["lambda_"]
        alpha = self.paramsSingle["alpha"]
        fontdict = {"size": 14}
        title = rf"Monte-Carlo simulation with $n = {self.n_simul}, \lambda ={lambda_}, \alpha = {alpha}$"
        plt.figure(figsize=(10, 5))
        plt.plot(self.time_vector, self.meanResult, label=mainlabel)
        plt.plot(self.time_vector, self.quantile5, label="Quantile 5%")
        plt.plot(self.time_vector, self.quantile95, label="Quantile 95%")
        plt.fill_between(self.time_vector, self.quantile5,
                         self.quantile95, alpha=0.2)
        plt.plot(
            self.time_vector, self.theory_vector, marker=".", label="Serie Expansion"
        )
        # params
        plt.title(title, fontdict=fontdict)
        plt.xlabel("Time", fontdict=fontdict)
        plt.ylabel("Population", fontdict=fontdict)
        plt.legend()
        plt.grid()
        if mode == "plot":
            plt.show()

        elif mode == "save":
            plt.savefig(path)
