import pickle
import numpy as np
import csv
import os
from my_backprojection import My_backprojection
from my_backpropagation import My_backpropagation
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, samples_generator, make_circles, make_classification, make_s_curve
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


def main():
    # ---- settings:
    dataset = "two_different_blobs" #--> two_moons, cocentered_cricles, two_blobs, two_different_blobs, three_different_blobs, s_curve, uniform_grid
    method = "backprojection" #--> backpropagation, backprojection
    do_kernel = False
    kernel = "rbf"  # rbf, polynomial, poly, cosine, linear
    generate_synthetic_datasets_again = False
    plot_dataset_again = True
    n_neurons = [15, 20, None]
    learning_rate = 1e-5
    weightDecay_parameter = 0
    activation_function_middleLayers = "linear"  #--> sigmoid, linear, softplus, tanh, elu, softmax
    activation_function_lastLayer = "sigmoid"    #--> sigmoid, linear, softplus, tanh, elu, softmax
    procedure = "backward"  #--> forward, backward, forward_backward
    loss_function_middleLayers = "MSE"  #--> MSE
    loss_function_lastLayer = "MSE"  #--> MSE, cross_entropy
    batch_size = 5
    n_epochs = 1000
    step_checkpoint = 1


    # ---- dataset:
    if dataset == "test_data":
        X_train = np.ones((10, 200)) * 3
        Y_train = np.ones((30, 200)) * 0.5
    else:
        X_train, X_test, Y_train, Y_test, y_train, y_test = read_synthetic_dataset(dataset, generate_synthetic_datasets_again, plot_dataset_again)
        # transpose (to have columns as features and rows as samples):
        X_train = X_train.T
        X_test = X_test.T
        if len(Y_train.shape) > 1:
            Y_train = Y_train.T
        else:
            Y_train = Y_train.reshape((1, -1))
        n_classes = Y_train.shape[0]
        n_neurons[-1] = n_classes

    if method == "backprojection":
        my_backprojection = My_backprojection(X=X_train, Y=Y_train, y_train=y_train, n_neurons=n_neurons, learning_rate=learning_rate,
                                              activation_function_middleLayers=activation_function_middleLayers,
                                              activation_function_lastLayer=activation_function_lastLayer,
                                              batch_size=batch_size, procedure=procedure,
                                              loss_function_middleLayers=loss_function_middleLayers, loss_function_lastLayer=loss_function_lastLayer,
                                              dataset_name=dataset, weightDecay_parameter=weightDecay_parameter,
                                              do_kernel=do_kernel, kernel=kernel, X_test=X_test, y_test=y_test)
        my_backprojection.backprojection(max_epochs=n_epochs, step_checkpoint=step_checkpoint, save_variable_images=True, save_embedding_images=True)
    elif method == "backpropagation":
        my_backpropagation = My_backpropagation(X=X_train, Y=Y_train, y_train=y_train, n_neurons=n_neurons, learning_rate=learning_rate,
                                              activation_function_middleLayers=activation_function_middleLayers,
                                              activation_function_lastLayer=activation_function_lastLayer,
                                              batch_size=batch_size,
                                              loss_function_middleLayers=loss_function_middleLayers, loss_function_lastLayer=loss_function_lastLayer,
                                              dataset_name=dataset, weightDecay_parameter=weightDecay_parameter, X_test=X_test, y_test=y_test)
        my_backpropagation.backpropagation(max_epochs=n_epochs, step_checkpoint=step_checkpoint, save_variable_images=True, save_embedding_images=True)

def save_variable(variable, name_of_variable, path_to_save='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
        os.makedirs(path_to_save)
    file_address = path_to_save + name_of_variable + '.pckl'
    f = open(file_address, 'wb')
    pickle.dump(variable, f)
    f.close()

def load_variable(name_of_variable, path='./'):
    # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
    file_address = path + name_of_variable + '.pckl'
    f = open(file_address, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def read_synthetic_dataset(dataset, generate_synthetic_datasets_again, plot_dataset_again):
    # https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py
    # https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py
    # https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
    # settings:
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150), np.linspace(-7, 7, 150))
    rng = np.random.RandomState(42)
    # dataset:
    path_dataset = './datasets/' + dataset + "/"
    if generate_synthetic_datasets_again:
        if dataset == "two_moons":
            X, y = make_moons(n_samples=300, noise=.05, random_state=0)
        elif dataset == "cocentered_cricles":
            X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=1)
        elif dataset == "two_blobs":
            X, y = make_blobs(n_samples=300, centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5])
        elif dataset == "two_different_blobs":
            X, y = make_blobs(n_samples=300, centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3])
        elif dataset == "three_different_blobs":
            X, y = make_blobs(n_samples=300, centers=[[2, 2], [-2, -2], [2, -3]], cluster_std=[1.5, .3, 1.0])
        elif dataset == "s_curve":
            X_3d, y = make_s_curve(n_samples=300, random_state=0)
            X = np.column_stack((X_3d[:, 0], X_3d[:, 2]))
        elif dataset == "uniform_grid":
            x = np.linspace(0, 1, int(np.sqrt(300)))
            xx, yy = np.meshgrid(x, x)
            X = np.hstack([
                xx.ravel().reshape(-1, 1),
                yy.ravel().reshape(-1, 1),
            ])
            y = xx.ravel()
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
        if dataset != "s_curve" and dataset != "uniform_grid":
            enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            Y = enc.fit_transform(y.reshape((-1, 1)))
            Y_train = enc.transform(y_train.reshape((-1, 1)))
            Y_test = enc.transform(y_test.reshape((-1, 1)))
        else:
            Y = y
            Y_train = y_train
            Y_test = y_test
        save_variable(variable=X, name_of_variable="X", path_to_save=path_dataset)
        save_variable(variable=X_train, name_of_variable="X_train", path_to_save=path_dataset)
        save_variable(variable=X_test, name_of_variable="X_test", path_to_save=path_dataset)
        save_variable(variable=Y, name_of_variable="Y", path_to_save=path_dataset)
        save_variable(variable=Y_train, name_of_variable="Y_train", path_to_save=path_dataset)
        save_variable(variable=Y_test, name_of_variable="Y_test", path_to_save=path_dataset)
        save_variable(variable=y_train, name_of_variable="y_train_notEncoded", path_to_save=path_dataset)
        save_variable(variable=y_test, name_of_variable="y_test_notEncoded", path_to_save=path_dataset)
    else:
        X = load_variable(name_of_variable="X", path=path_dataset)
        X_train = load_variable(name_of_variable="X_train", path=path_dataset)
        X_test = load_variable(name_of_variable="X_test", path=path_dataset)
        Y = load_variable(name_of_variable="Y", path=path_dataset)
        Y_train = load_variable(name_of_variable="Y_train", path=path_dataset)
        Y_test = load_variable(name_of_variable="Y_test", path=path_dataset)
        y_train = load_variable(name_of_variable="y_train_notEncoded", path=path_dataset)
        y_test = load_variable(name_of_variable="y_test_notEncoded", path=path_dataset)
    if plot_dataset_again:
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # just plot the dataset first
        # cm_bright = plt.cm.RdBu
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        else:
            cm_bright = plt.cm.brg
        # cm_bright = ListedColormap(['#FF0000', '#0000FF', '#FF00FF'])
        ax = plt.subplot()
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                   edgecolors='k', marker="s")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        plt.show()
    return X_train, X_test, Y_train, Y_test, y_train, y_test

if __name__ == '__main__':
    main()