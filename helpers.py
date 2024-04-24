import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import scipy

from collections import defaultdict
import datetime
# from google.colab import files

import pickle
from pprint import pprint
from itertools import product


SEED = 12345
np.random.seed(SEED)


colors = tuple(reversed([
    '#cecece', 
    '#a559aa', 
    '#59a89c', 
    '#f0c571', 
    '#e02b35', 
    '#082a54',
]))


def process_constrained_swissroll(swissroll_constrained_dict, last_epochs_considered):
    swissroll_processed_dict = defaultdict(lambda: defaultdict(dict))

    # for (dataset_name, comparisons) in swissroll_processed_dict.items():
    for (comparison, examples) in swissroll_constrained_dict.items():
        for trait in examples:
            train_losses, train_successes, val_losses, val_successes = examples[trait]["results"]

            swissroll_processed_dict[comparison][trait] = {
              "val_avg": np.round(np.average( val_successes[-last_epochs_considered:] ) * 100, decimals=2),
              "val_std": np.round(np.std( val_successes[-last_epochs_considered:] ) * 100, decimals=2),

              "train_avg": np.round(np.average( train_successes[-last_epochs_considered:] ) * 100, decimals=2),
              "train_std": np.round(np.std( train_successes[-last_epochs_considered:] ) * 100, decimals=2),
            }
    
    return swissroll_processed_dict


def process_networks_comparisons(network_comparisons_dict, last_epochs_considered):
    networks_processed_dict = defaultdict(lambda: defaultdict(dict))

    for (dataset_name, comparisons) in network_comparisons_dict.items():
        for (comparison, examples) in comparisons.items():
            for trait in examples:
                train_losses, train_successes, val_losses, val_successes = examples[trait]["results"]

                networks_processed_dict[dataset_name][comparison][trait] = {
                  "val_avg": np.round(np.average( val_successes[-last_epochs_considered:] ) * 100, decimals=2),
                  "val_std": np.round(np.std( val_successes[-last_epochs_considered:] ) * 100, decimals=2),

                  "train_avg": np.round(np.average( train_successes[-last_epochs_considered:] ) * 100, decimals=2),
                  "train_std": np.round(np.std( train_successes[-last_epochs_considered:] ) * 100, decimals=2),
                }
        
    return networks_processed_dict


def process_grid_search_results(grid_search_dict, last_epochs_considered):
    processed_dict = defaultdict(lambda: defaultdict(dict))

    for dataset_name in grid_search_dict:
        for combination in grid_search_dict[dataset_name]:
            _, train_losses, train_successes, val_losses, val_successes = grid_search_dict[dataset_name][combination]

            processed_dict[dataset_name][combination] = {
                  "val_avg": np.round(np.average( val_successes[-last_epochs_considered:] ) * 100, decimals=2),
                  "val_std": np.round(np.std( val_successes[-last_epochs_considered:] ) * 100, decimals=2),

                  "train_avg": np.round(np.average( train_successes[-last_epochs_considered:] ) * 100, decimals=2),
                  "train_std": np.round(np.std( train_successes[-last_epochs_considered:] ) * 100, decimals=2),
              }
        
    return processed_dict


def comparison_string(a, b):
    return f'{a} (vs. {b}, {percent_change(a, b):+}%)'


def percent_change(compared, base):
    return np.round((compared - base)/base * 100, decimals=2)


def print_swissroll_constrained(swissroll_processed_dict, networks_processed_dict):
    print('=' * 100)
    print("SwissRoll")

    for (comparison, traits) in swissroll_processed_dict.items():
        print('-' * 100)
        print("\t" + comparison)
        print('-' * 100)

        format_string = "\t\t| " + "{:<25} | " * len(traits)

        print(format_string.format(*tuple(traits)))
        print('.' * 100)

        print(format_string.format(*tuple(
            comparison_string(
                results["val_avg"], 
                networks_processed_dict["SwissRoll"][comparison][trait]["val_avg"]
            )
            for (trait, results) in traits.items()))
        )


def print_networks_comparisons(networks_processed_dict):
    for (dataset_name, comparisons) in networks_processed_dict.items():
        print('=' * 100)
        print(dataset_name)

        for (comparison, traits) in comparisons.items():
            print('-' * 100)
            print("\t" + comparison)
            print('-' * 100)

            format_string = "\t\t| " + "{:<22} | " * len(traits)

            print(format_string.format(*tuple(traits)))
            print('.' * 100)

            print(format_string.format(*tuple(results["val_avg"] for (_, results) in traits.items())))


def print_grid_search_top_X(processed_dict, top_results_count, worst=False):
    print(f"Top {top_results_count} {'worst' if worst else 'best'} performing combination of parameters:")
    
    sorting_factor = -1 if worst else 1
    
    format_string = "{:<65} | " + "{:<20} | " * 4

    for dataset_name in processed_dict:
        print('=' * 160)
        print(dataset_name)
        print('-' * 160)
        print(format_string.format(
          "(epoch, minibatch, learning_rate, update_rate, update_factor)", 
          "val. success % avg.", "val. success % std",
          "train success % avg", "train success % std",
        ))
        print('-' * 160)
        for (combination, results) in sorted(processed_dict[dataset_name].items(), 
                                             key=lambda item: (sorting_factor * -item[1]["val_avg"], 
                                                               sorting_factor * item[1]["val_std"]))[:top_results_count]:
            print(format_string.format(f"{combination}", results["val_avg"], results["val_std"], results["train_avg"], results["train_std"]))
            
            
def plot_swissroll_constrained(swissroll_constrained_dict):
    print('=' * 100)
    print("SwissRoll")

    for (comparison, examples) in swissroll_constrained_dict.items():
        print('-' * 100)
        print("\t" + comparison)
        print('-' * 100)

        for trait in examples:
            train_losses, train_successes, val_losses, val_successes = examples[trait]["results"]

            plot_training_process(train_losses, val_losses, val_successes, train_successes, title=f"SwissRoll {trait}")


def plot_grid_search_top(processed_dict, grid_search_dict, worst=False):
    sorting_factor = -1 if worst else 1

    for dataset_name in processed_dict:
        print('='*80)
        print(dataset_name)
        print('-'*80)

        top_validation = min(
            processed_dict[dataset_name].items(), 
            key=lambda item: (sorting_factor * -item[1]["val_avg"],
                              sorting_factor * item[1]["val_std"])
        )

#         best_softmax_weights[dataset_name] = grid_search_dict[dataset_name][combination][0]

        combination, results = top_validation
        _, train_losses, train_successes, val_losses, val_successes = grid_search_dict[dataset_name][combination]

        plot_training_process(train_losses, val_losses, train_successes, val_successes,
            title=f"{dataset_name} best performing: result={results['val_avg']}% parameters={combination}"
        )
        
        
def plot_networks_comparisons(network_comparisons_dict):
    for (dataset_name, comparisons) in network_comparisons_dict.items():
        print('=' * 100)
        print(dataset_name)

        for (comparison, examples) in comparisons.items():
            print('-' * 100)
            print("\t" + comparison)
            print('-' * 100)

            for trait in examples:
                train_losses, train_successes, val_losses, val_successes = examples[trait]["results"]

                plot_training_process(train_losses, val_losses, val_successes, train_successes, title=f"{dataset_name} {trait}")
        
        
def get_top_grid_search_weights(processed_dict, grid_search_dict, worst=False):
    sorting_factor = -1 if worst else 1

    best_softmax_weights = dict()
    
    for dataset_name in processed_dict:
        top_validation = min(
            processed_dict[dataset_name].items(), 
            key=lambda item: (sorting_factor * -item[1]["val_avg"],
                              sorting_factor * item[1]["val_std"])
        )

        combination, results = top_validation
        _, train_losses, train_successes, val_losses, val_successes = grid_search_dict[dataset_name][combination]

        best_softmax_weights[dataset_name] = grid_search_dict[dataset_name][combination][0]
    
    return best_softmax_weights
            
            
def save_backup(content, file_name):
    with open(file_name, "wb") as file_obj:
        pickle.dump(dict(content), file_obj, pickle.HIGHEST_PROTOCOL)


def load_backup(file_name):
    with open(file_name, "rb") as file_obj:
        file_content = pickle.load(file_obj)
    return file_content


def plot_dataset(X, Y, title):
    if X.shape[0] == 2:
        plot_2d(X, Y, title)
    else:
        plot_3d(X, Y, title)


def plot_2d(X, Y, title):
    f = plt.figure()
    f.set_figwidth(6)
    f.set_figheight(6)
    
    for i in range(Y.shape[0]):
        points = X[:, Y[i] == 1]
        plt.scatter(points[0], points[1], color=colors[i], label=f"Class {i}",)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.show()
    

def plot_3d(X, Y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(5):
        points = X[:, Y[i] == 1]
        ax.scatter(points[0], points[1], points[2], color=colors[i], label=f"Class {i}", s=10,)
    
    ax.set_title(title)
    ax.legend()

    # Remove grid
    ax.grid(False)
    plt.show()
    
    
def show_given_datasets():
    _, _, X_SwissRoll, Y_SwissRoll = load_dataset('SwissRollData.mat')
    plot_dataset(X_SwissRoll, Y_SwissRoll, "Swiss Roll Dataset")
    
    _, _, X_Peaks, Y_Peaks =  load_dataset('PeaksData.mat')
    plot_dataset(X_Peaks, Y_Peaks, "Peaks Dataset")
    
    _, _, X_GMM, Y_GMM = load_dataset('GMMData.mat')
    plot_dataset(X_GMM, Y_GMM, "GMM Dataset")


def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2
    
    
def print_shapes(iterable):
    for w in iterable:
        print(w.shape)
    
    
def LS_random_data():
    DATAPOINTS_NUM = 1000
    TRUE_WEIGHTS = np.array([3.456, 10.987]).reshape(-1,1)

    X = np.random.uniform(low=-10, high=10, size=(1, DATAPOINTS_NUM))
    y_clean = TRUE_WEIGHTS.T @ add_ones_row(X)

    gausian_noise = np.random.normal( loc=0.0, scale=5, size=y_clean.shape )
    y_noisy = y_clean + gausian_noise

    return X, y_noisy, TRUE_WEIGHTS
    

def layer_direct_jacobian_transposed_random_data(layer_weights, datapoints_num=1000):
    X_input_shape = (layer_weights.shape[1]-1, datapoints_num)
    X_input = np.random.uniform(-1, 1, size=X_input_shape)
#      X_input = np.random.uniform(-10, 10, size=X_input_shape)
#      X_input_with_ones = add_ones_row(X_input)
    return X_input


def softmax_gradient_test_random_data(datapoint_size=4, datapoints_num=1000, classes_num=3):
    X_shape = (datapoint_size, datapoints_num)
    X = np.random.uniform(-10, 10, size=X_shape)

    W_shape = (datapoint_size + 1, classes_num)
    W = randn_norm_1(W_shape)

    C_shape = (classes_num, datapoints_num)
    labels = np.random.randint(classes_num, size=datapoints_num)
    C = to_one_hot(labels, classes_num)

#     assert C.shape == C_shape, f"{C.shape = }, {C_shape = }"

    return X, W, C


def randn_norm_1(shape):
    return np.random.randn(*shape) / np.exp(np.mean(np.log(shape)))


def to_one_hot(values, classes_num):
    return np.identity(classes_num)[:, values]


def to_classes(one_hot_vectors):
    return np.argmax(one_hot_vectors, axis=0)


def add_ones_row(X):
  datapoint_size, datapoints_num = X.shape

  X_with_ones = np.vstack([
      X,
      np.ones((1, datapoints_num))
  ])

  return X_with_ones


def show_with_legend(title="", x_label="", y_label=""):
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend()
  plt.show()


def epoch_generator(X, Y, minibatch_size):
    datapoint_size, datapoints_num = X.shape

    shuffled_indices = np.random.permutation(datapoints_num)
    minibatch_start = 0
    minibatch_end = min(minibatch_start + minibatch_size, datapoints_num)

    while minibatch_end != datapoints_num:
        minibatch_X = X[:, shuffled_indices[minibatch_start : minibatch_end]]
        minibatch_Y = Y[:, shuffled_indices[minibatch_start : minibatch_end]]
        yield minibatch_X, minibatch_Y

        minibatch_start += minibatch_size
        minibatch_end = min(minibatch_start + minibatch_size, datapoints_num)

   
def randomly_split_dataset(X, Y, split_percentage):
    datapoint_size, datapoints_num = X.shape
    classes_num, datapoints_num = Y.shape

    shuffled_indices = np.random.permutation(datapoints_num)

    part_1_size = np.int64(split_percentage * datapoints_num)
    part_1_indices = shuffled_indices[ : part_1_size]
    part_2_indices = shuffled_indices[part_1_size : ]

    X_1 = X[:, part_1_indices]
    Y_1 = Y[:, part_1_indices]

    X_2 = X[:, part_2_indices]
    Y_2 = Y[:, part_2_indices]

    return X_1, Y_1, X_2, Y_2


def load_dataset(path):
    dataset = scipy.io.loadmat(path)

    X_train = np.array(dataset['Yt'])
    Y_train = np.array(dataset['Ct'])

    X_val = np.array(dataset['Yv'])
    Y_val = np.array(dataset['Cv'])

    return X_train, Y_train, X_val, Y_val


def plot_LS_results(X, true_weights, y_noisy, learned_weights):
    plt.scatter(X, y_noisy, label="noisy data")
    draw_line(*true_weights, label="true function", color="red")
    draw_line(*learned_weights, label="learned function", color="green")
    show_with_legend(title="Least squares truth vs prediction")


def plot_training_process(train_losses_list, validation_losses_list, \
  train_success_percentages=None, validation_success_percentages=None, title=""):

  if train_success_percentages is None:
    plt.plot(train_losses_list, label="Train set")
    plt.plot(validation_losses_list, label="Validation set")
    plt.title(f"{title} loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

  else:
    fig, axis = plt.subplots(1, 2)
    train_success_array = np.array(train_success_percentages) * 100
    validation_success_array = np.array(validation_success_percentages) * 100

    axis[0].plot(train_losses_list, label="Train set")
    axis[0].plot(validation_losses_list, label="Validation set")
    axis[0].set_title(f"loss")
    axis[0].set_xlabel("epoch")
    axis[0].set_ylabel("loss")
    axis[0].legend()

    axis[1].plot(train_success_array, label="Train set")
    axis[1].plot(validation_success_array, label="Validation set")
    axis[1].set_title(f"classification success")
    axis[1].set_xlabel("epoch")
    axis[1].set_ylabel("success %")
    axis[1].legend()
    fig.suptitle(title)
    fig.set_figwidth(16)
    fig.set_figheight(4)
    fig.tight_layout()
    plt.show()


def draw_line(slope, intercept, label, color):
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, label=label, color=color)
    


