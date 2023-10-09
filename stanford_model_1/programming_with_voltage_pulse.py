import matplotlib.pyplot as plt

from stanford_model_1 import RRAM

from waveform_generators import (
    generate_square_wave,
    custom_wave_generator,
)


def time_to_program():
    rram = RRAM(gap_ini=17e-10, model_switch=0)

    time_arr, voltage_arr = custom_wave_generator(
        generate_square_wave(
            level=1.4,
            delay=50e-9,
            transition_time=1e-9,
            pulse_duration=0.6e-6,
            period=0.65e-6,
            dt=1e-9,
            cycles=1,
        )
    )

    history = rram.transient(time_arr, voltage_arr)

    plt.figure()
    plt.plot(time_arr, history["gap"])
    plt.show()


if __name__ == "__main__":
    time_to_program()
