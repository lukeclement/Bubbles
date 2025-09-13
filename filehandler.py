import numpy as np
import glob
import dataprocessor as process
import logging
LOG = logging.getLogger(__name__)

def read_data(filename: str):
    """
    Reads in the .dat file specified
    :param filename:
    :return:
    """
    #TODO: Clean me! Can be made faster I'm sure (yeah, by moving to Java lmao)
    datapoints = []
    with open(filename) as f:
        passed = False
        for line in f:
            info = line.strip()
            if passed and info != "ZONE T=\"boundary5\"":
                datapoints.append([float(info.split(" ")[0]), float(info.split(" ")[1])])
            if info == "ZONE T=\"boundary4\"":
                passed = True

    a = np.asarray(datapoints)

    ind = np.lexsort((a[:,1],a[:,0]))
    datapoints = a[ind]

    sorted_datapoints = sort_datapoints(datapoints)

    x = sorted_datapoints[:, 0]
    y = sorted_datapoints[:, 1]
    return x, y


def sort_datapoints(datapoints):
    sorted_datapoints = []
    index = 0
    mask = np.zeros(len(datapoints))
    for i in range(0, len(datapoints)):
        point = datapoints[index]
        sorted_datapoints.append(point)
        mask[index] = 1
        vectors_to_point = datapoints[mask < 1] - point
        distances_to_point = np.sqrt(vectors_to_point[:, 0] ** 2 + vectors_to_point[:, 1] ** 2)
        if len(distances_to_point) > 1:
            smallest_nonzero_distance = np.min(distances_to_point)
        else:
            break
        all_vectors_to_point = datapoints - point
        all_distances_to_point = np.sqrt(all_vectors_to_point[:, 0] ** 2 + all_vectors_to_point[:, 1] ** 2)
        next_index_maybe = np.argwhere(all_distances_to_point == smallest_nonzero_distance)

        boop = 0
        for j in range(0, len(next_index_maybe)):
            if mask[next_index_maybe[j, 0]] == 0:
                boop = j
        next_index = next_index_maybe[boop, 0]
        # LOG.debug("Point {}, index {}".format(point, index))
        index = next_index
        # LOG.debug("Minimum distance {:.3f} (element {}, point {}) - or could be element {}".format(smallest_nonzero_distance, next_index, datapoints[next_index], next_index_maybe))
    sorted_datapoints.append(sorted_datapoints[-1])
    sorted_datapoints = np.asarray(sorted_datapoints)
    return sorted_datapoints


def extract_data(simulation_name: str, series_size: int):
    #TODO: ensure bubble is _always_ moving clockwise
    path = "Simulation_data/Simulation_data/{}".format(simulation_name)
    files = glob.glob("{}/boundaries_*.dat".format(path))
    frames = len(files)

    bubbles_data = []
    LOG.info("Extracting data for %s - %d frames", simulation_name, frames)
    for i in range(0, frames):
        LOG.debug("Reading frame {}/{}".format(i, frames))
        file = "{}/boundaries_{}.dat".format(path, i)
        x, y = read_data(file)
        if np.arctan2(x[10], y[10]) < np.arctan2(x[50], y[50]):
            x = x[::-1]
            y = y[::-1]
        LOG.debug("Fitting frame {}/{}".format(i, frames))
        popt_x, popt_y, _, _ = process.get_fit(x, y, series_size)
        bubbles_data.append([x, y, popt_x, popt_y])

    return bubbles_data

