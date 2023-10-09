import numpy as np
import matplotlib.pyplot as plt

from stanford_model_1 import RRAM

from waveform_generators import (
    generate_triangular_wave,
    custom_wave_generator,
)


def generate_butterfly_curve_current_sweep():
    rram = RRAM(gap_ini=2e-10)

    time_arr, current_arr = custom_wave_generator(
        generate_triangular_wave(level=-0.02, duration=8e-6, dt=1e-9),
        generate_triangular_wave(level=0.09, duration=8e-6, dt=1e-9),
    )

    history = rram.transient(time_arr, current_arr, input_nature="current")

    plt.figure()
    # plt.plot(history["Vtb"], np.log10(np.abs(current_arr)))
    plt.plot(history["Vtb"], current_arr)
    plt.show()


if __name__ == "__main__":
    generate_butterfly_curve_current_sweep()
