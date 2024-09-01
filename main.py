import numpy as np

from computation.figure_handler import finalFigureSave


def main():
    params = dict(
        lambda_=4,
        alpha=1.5,
        r=0.0,
        t_max=10,
        step=0.1,
        nSimulVector=[10, 50, 100, 500, 1000],
        orderVector=np.arange(10, 200, 10),
    )
    finalFigureSave(**params)
    return


if __name__ == "__main__":
    main()
