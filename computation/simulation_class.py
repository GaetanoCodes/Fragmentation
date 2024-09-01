import numpy as np
import matplotlib.pyplot as plt
from computation.theory_class import FragmentationTheory
import tqdm

# import time


class FragmentationSingleSimulation:
    def __init__(self, lambda_, alpha, r, t_max, step):
        self.lambda_ = lambda_
        self.alpha = alpha
        self.t_max = t_max
        self.r = r
        self.step = step

    def timeInitialisation(self):
        self.timeVector = np.linspace(0, self.t_max, int(self.t_max / self.step) + 1)

    def simulation(self):
        massInit = np.array([1])
        mass = massInit
        demography = []
        for t in self.timeVector:
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
        plt.plot(self.timeVector, self.demography)
        plt.show()


class FragmentationSimulation:
    def __init__(self, lambda_, alpha, r, t_max, step, nSimul):
        self.paramsSingle = dict(
            lambda_=lambda_, alpha=alpha, r=r, t_max=t_max, step=step
        )
        self.nSimul = nSimul
        self.t_max = t_max
        self.step = step

    def timeInitialisation(self):
        self.timeVector = np.linspace(0, self.t_max, int(self.t_max / self.step) + 1)
        print(
            f"(*) Time Vector from 0 to {self.t_max} with a {self.step} step has been built."
        )

    def monteCarloSimulation(self):
        L = []
        for nSimul in tqdm.tqdm(range(self.nSimul)):
            f = FragmentationSingleSimulation(**self.paramsSingle)
            f.timeInitialisation()
            f.simulation()
            L.append(f.demography)
            # print(nSimul)

        self.simulations = np.array(L)

    def getStatistics(self):
        self.meanResult = np.mean(self.simulations, axis=0)
        self.stdEstimate = (
            (np.sum(self.simulations**2, axis=0) - self.nSimul * self.meanResult**2)
            ** (0.5)
        ) / self.nSimul
        self.quantile5 = self.meanResult - 1.95 * self.stdEstimate
        self.quantile95 = self.meanResult + 1.95 * self.stdEstimate
        fragmentationTh = FragmentationTheory(**self.paramsSingle)
        fragmentationTh.construction()
        self.theoryVector = fragmentationTh.theoryVector

    def resultHandler(self, mode="plot", path=""):
        # plt.rcParams["text.usetex"] = True
        mainlabel = "Mean"
        if self.nSimul == 1:
            mainlabel = "Realisation"

        lambda_ = self.paramsSingle["lambda_"]
        alpha = self.paramsSingle["alpha"]
        fontdict = {"size": 14}
        title = rf"Monte-Carlo simulation with $n = {self.nSimul}, \lambda ={lambda_}, \alpha = {alpha}$"
        plt.figure(figsize=(10, 5))
        plt.plot(self.timeVector, self.meanResult, label=mainlabel)
        plt.plot(self.timeVector, self.quantile5, label="Quantile 5%")
        plt.plot(self.timeVector, self.quantile95, label="Quantile 95%")
        plt.fill_between(self.timeVector, self.quantile5, self.quantile95, alpha=0.2)
        plt.plot(
            self.timeVector, self.theoryVector, marker=".", label="Serie Expansion"
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


# if __name__ == "__main__":
#     params = dict(lambda_=5, alpha=0.5, r=0.1, t_max=30, step=0.2, nSimul=10000)
#     f = Fragmentation(**params)
#     f.timeInitialisation()
#     t0 = time.time()
#     f.monteCarloSimulation()
#     f.getStatistics()
#     print((time.time() - t0) / 100, "s")
#     f.plotResult(displayQuarter=True)
