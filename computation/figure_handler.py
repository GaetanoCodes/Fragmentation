import numpy as np
import matplotlib.pyplot as plt
import computation.simulation_class as simulation
import computation.lambda_estimation as estimation
import os


def finalFigureSave(lambda_, alpha, r, t_max, step, nSimulVector, orderVector):
    path = f"output/lambda = {lambda_}, alpha = {alpha}"
    os.mkdir(path)
    params = dict(lambda_=lambda_, alpha=alpha, r=r, t_max=t_max, step=step)

    for nSimul in nSimulVector:
        params["nSimul"] = nSimul
        f = simulation.FragmentationSimulation(**params)
        f.timeInitialisation()
        f.monteCarloSimulation()
        f.getStatistics()
        f.resultHandler(mode="save", path=f"{path}/POP_{nSimul}.png")
        estimator = estimation.LambdaEstimation(
            params["r"], params["alpha"], f.meanResult, f.timeVector
        )
        estimator.resultHandler(
            orderVector,
            trueLambda=params["lambda_"],
            mode="save",
            path=f"{path}/EST_{nSimul}.png",
        )
