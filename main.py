import numpy as np
import matplotlib.pyplot as plt
import simulation_class as simulation
import lambda_estimation as estimation


if __name__ == "__main__":
    params = dict(lambda_=5, alpha=0.5, r=0.1, t_max=150, step=0.2, nSimul=100)
    f = simulation.Fragmentation(**params)
    f.timeInitialisation()
    f.monteCarloSimulation()
    f.getStatistics()
    # f.plotResult()

    estimator = estimation.LambdaEstimation(0.1, 0.5, f.meanResult, f.timeVector)

    print(estimator.lambdaEstimation(100))
