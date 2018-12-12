import os
import pickle

from loop_environments import plot_episode_rewards, plot_episode_durations

RESULTS_FOLDER = "results"
FINAL_FOLDER = "final_images"

def load_files(folder, environment, result_type):
    """
    Loads all of the files in a folder from a given environment.
    """
    files = []

    for file in os.listdir(folder):
        if environment in file and result_type in file:
            files.append(os.path.join(folder, file))
    return files


def load_results(file_name):
    """
    Loads the saved results from a Pickle file.
    """
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    return data


def plot(env_name, result_type):
    """
    Plots the episode rewards or durations.
    """
    results = []
    mem_names = []

    files = load_files(RESULTS_FOLDER, env_name, result_type)

    for file in files:
        mem_name = file.split("_")[-2]
        mem_names.append(mem_name)
        results.append(load_results(file))

    if result_type == "durations":
        plot_episode_durations(results, mem_names, os.path.join(FINAL_FOLDER, f"{env_name}"))
    elif result_type == "rewards":
        plot_episode_rewards(results, mem_names, os.path.join(FINAL_FOLDER, f"{env_name}"))
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    if not os.path.exists(FINAL_FOLDER):
        os.mkdir(FINAL_FOLDER)

    plot("HugeGridWorld", "rewards")
