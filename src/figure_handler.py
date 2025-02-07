import matplotlib.pyplot as plt
import computation.simulation_class as simulation
import computation.lambda_estimation as estimation
import os

import scienceplots

plt.style.use("science")


def final_figure_save(lambda_, alpha, r, t_max, step, n_simul_range, order_vector):
    path = f"output/lambda = {lambda_}, alpha = {alpha}"
    os.mkdir(path)
    params = dict(lambda_=lambda_, alpha=alpha, r=r, t_max=t_max, step=step)

    for n_simul in n_simul_range:
        params["n_simul"] = n_simul
        f = simulation.Fragmentation_simulation(**params)
        f.timeInitialisation()
        f.monteCarloSimulation()
        f.getStatistics()
        f.result_handler(mode="save", path=f"{path}/POP_{n_simul}.png")
        estimator = estimation.LambdaEstimation(
            params["r"], params["alpha"], f.meanResult, f.time_vector
        )
        estimator.result_handler(
            order_vector,
            true_lambda=params["lambda_"],
            mode="save",
            path=f"{path}/EST_{n_simul}.png",
        )
