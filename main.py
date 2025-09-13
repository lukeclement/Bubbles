import filehandler as fh
import dataprocessor as processor
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import logging
import logging.config
import time

def log_filter(level):
    level = getattr(logging, level)

    def logging_filter(record):
        return record.levelno <= level

    return logging_filter

with open('LoggerConfig.json') as f:
    logging.config.dictConfig(json.load(f))



def main():
    simulations = glob.glob("Simulation_data/Simulation_data/*")
    series_size = 20
    sample_size = 20
    simulation_to_name = {
        "Example_Simulation": "A",
        "r0.54eps18.4": "B",
        "r0.54eps18.6": "C",
        "r0.54eps18.7": "D",
        "r0.54eps18.8": "E",
        "r0.54eps18.9": "F",
        "r0.54eps18.43": "G",
        "r0.54eps18.65": "H",
        "r0.54eps18.75": "I",
        "r0.54eps18.95": "J",
        "r0.54eps18.625": "K",
        "r0.54eps18.925": "L",
        "r0.54eps19.0": "M",
        "r0.54eps19.1": "N",
        "r0.54eps19.5": "O",
        "r0.54eps19.25": "P"
    }


    logging.info("Starting to process simulations...")
    for simulation in simulations[-1:]:
        simulation_name = simulation[32:]
        start = time.time()
        bubbles_data = fh.extract_data(simulation_name, series_size)
        end = time.time()

        #TODO: saving bubbles

        for i in range(500, 610, 10):
            investigate_bubble(bubbles_data, sample_size, series_size, i)

        y_average_final = np.average(bubbles_data[-1][1])
        time_taken = end - start
        time_per_frame = time_taken / len(bubbles_data)
        logging.info("Done! Final y average was {}, took {:.2f}s ({:.0f}ms per frame)"
                     .format(y_average_final, time_taken, time_per_frame * 1000))\



def investigate_bubble(bubbles_data, sample_size, series_size, bubble_number):
    logging.info("Looking into bubble {}".format(bubble_number))
    x = bubbles_data[bubble_number][0]
    y = bubbles_data[bubble_number][1]
    x_fit_coef = bubbles_data[bubble_number][2]
    y_fit_coef = bubbles_data[bubble_number][3]
    t = np.linspace(0, np.pi * 2, 500)
    # Fourier fit func expects something like:
    # [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] raw coefficent array
    # [ 0, 1, 2, 3, 4, 0, 1, 2, 3, 4] frequency of sine waves and coefficient refs
    # |      sin     |      cos     | which one the coefficients belong to
    x_fit = processor.fourier_fit_func(t, x_fit_coef[:sample_size], x_fit_coef[series_size:series_size + sample_size],
                                       sample_size)
    y_fit = processor.fourier_fit_func(t, y_fit_coef[:sample_size], y_fit_coef[series_size:series_size + sample_size],
                                       sample_size)
    plt.title(bubble_number)
    plt.scatter(x, y)
    plt.scatter(x_fit, y_fit, c=t)
    plt.show()
    plt.clf()
    for i in range(0, sample_size):
        logging.info("x += {:.3f} * sin ({} * t) + {:.3f} * cos ({} * t)"
                     .format(x_fit_coef[i], i, x_fit_coef[i + series_size], i))
        logging.info("y += {:.3f} * sin ({} * t) + {:.3f} * cos ({} * t)"
                     .format(y_fit_coef[i], i, y_fit_coef[i + series_size], i))


if __name__ == '__main__':
    main()