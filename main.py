import numpy as np

from computation.figure_handler import final_figure_save


def main():
    params = dict(
        lambda_=4,
        alpha=0.5,
        r=0.0,
        t_max=10,
        step=0.1,
        n_simul_range=[10, 50, 100, 500, 1000],
        order_vector=np.arange(10, 200, 10),
    )
    final_figure_save(**params)
    return


if __name__ == "__main__":
    main()
