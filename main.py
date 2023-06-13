import numpy as np
import matplotlib.pyplot as plt
import simulation_class as simulation
import lambda_estimation as estimation
import os 

def finalFigureSave(lambda_, alpha, r, t_max, step, nSimulVector, orderVector):
    path = f"lambda_{lambda_}_alpha_{alpha}"
    os.mkdir(path)
    params = dict(lambda_=lambda_, alpha=alpha, r=r, t_max=t_max, step=step)
    
    for nSimul in nSimulVector:
        params["nSimul"] = nSimul
        f = simulation.FragmentationSimulation(**params)
        f.timeInitialisation()
        f.monteCarloSimulation()
        f.getStatistics()
        f.resultHandler(mode = "save", path = f"{path}/POP_{nSimul}.png")
        estimator = estimation.LambdaEstimation(params["r"], params["alpha"], f.meanResult, f.timeVector)
        estimator.resultHandler(orderVector, trueLambda = params["lambda_"], mode = "save", path =f"{path}/EST_{nSimul}.png")



if __name__ == "__main__":
    params = dict(lambda_=7, alpha=0.6, r=0., t_max=10, step=0.1, nSimulVector = [10,50,100,500,1000], orderVector = np.arange(10,200,10))
    finalFigureSave(**params)