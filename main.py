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
    processed = glob.glob("Simulation_data/Simulation_data_processed/phase/*")
    processed_names = []
    for p in processed:
        processed_names.append(p[48:-4])
    series_size = 10

    logging.info("Starting to process simulations...")
    for simulation in simulations:
        simulation_name = simulation[32:]
        file = "Simulation_data/Simulation_data_processed/phase/" + simulation_name
        if simulation_name not in processed_names:

            start = time.time()
            bubbles_data_xy, bubbles_data_phase = fh.extract_data(simulation_name, series_size)
            end = time.time()

            y_average_final = np.average(bubbles_data_xy[-1][1])
            time_taken = end - start
            time_per_frame = time_taken / len(bubbles_data_xy)
            logging.info("Done! Final y average was {}, took {:.2f}s ({:.0f}ms per frame)"
                         .format(y_average_final, time_taken, time_per_frame * 1000))

            logging.info(np.shape(bubbles_data_phase))
            logging.debug("Saving " + file)
            np.save(file, np.asarray(bubbles_data_phase))
        else:
            logging.info("Reading from file " + file)
            bubbles_data_phase = np.fromfile(file + ".npy")




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