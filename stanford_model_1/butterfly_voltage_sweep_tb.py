import numpy as np
import matplotlib.pyplot as plt

from stanford_model_1 import RRAM

from waveform_generators import (
    generate_triangular_wave,
    custom_wave_generator,
)


def generate_butterfly_curve_voltage_sweep():
    rram = RRAM(gap_ini=2e-10)

    time_arr, voltage_arr = custom_wave_generator(
        generate_triangular_wave(level=-1.55, duration=8e-6, dt=1e-9),
        generate_triangular_wave(level=1.55, duration=8e-6, dt=1e-9),
    )

    history = rram.transient(time_arr, voltage_arr)

    plt.figure()
    plt.plot(voltage_arr, np.log10(np.abs(history["Itb"])))
    # plt.plot(voltage_arr, history["Itb"])
    plt.show()


if __name__ == "__main__":
    generate_butterfly_curve_voltage_sweep()
