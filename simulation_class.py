import numpy as np
import matplotlib.pyplot as plt
from theory_class import FragmentationTheory
import time


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
            demography.append(len(mass))

        self.finalMass = mass
        self.demography = demography

    def plotResult(self):
        plt.plot(self.timeVector, self.demography)
        plt.show()


class Fragmentation:
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
        for nSimul in range(self.nSimul):
            f = FragmentationSingleSimulation(**self.paramsSingle)
            f.timeInitialisation()
            f.simulation()
            L.append(f.demography)
            print(nSimul)

        self.simulations = np.array(L)

    def getStatistics(self):
        self.meanResult = np.mean(self.simulations, axis=0)
        self.firstQuarter = np.quantile(self.simulations, 0.25, axis=0)
        self.thirdQuarter = np.quantile(self.simulations, 0.75, axis=0)
        fragmentationTh = FragmentationTheory(**self.paramsSingle)
        fragmentationTh.construction()
        self.theoryVector = fragmentationTh.theoryVector

    def plotResult(self):
        plt.plot(self.timeVector, self.meanResult, label="Mean")
        plt.plot(self.timeVector, self.firstQuarter, label="1st quarter")
        plt.plot(self.timeVector, self.thirdQuarter, label="3rd quarter")
        plt.plot(self.timeVector, self.theoryVector, label="Serie Expansion")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    params = dict(lambda_=5, alpha=0.5, r=0.1, t_max=150, step=0.2, nSimul=200)
    f = Fragmentation(**params)
    f.timeInitialisation()
    t0 = time.time()
    f.monteCarloSimulation()
    f.getStatistics()
    print((time.time() - t0) / 100, "s")
    f.plotResult()
