import numpy as np
import numpy.linalg as LA
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.special import softmax
import time
from sklearn.metrics.pairwise import pairwise_kernels

import warnings
warnings.filterwarnings("ignore")

class My_backprojection:

    def __init__(self, X, Y, y_train, n_neurons, learning_rate, activation_function_middleLayers,
                 activation_function_lastLayer, batch_size, procedure,
                 loss_function_middleLayers, loss_function_lastLayer, dataset_name, weightDecay_parameter,
                 do_kernel, kernel, X_test=None, y_test=None):
        assert batch_size <= X.shape[1]
        n_samples_divisable_by_batchSize = int(np.floor(X.shape[1] / batch_size)) * batch_size
        X = X[:, :n_samples_divisable_by_batchSize]
        Y = Y[:, :n_samples_divisable_by_batchSize]
        y_train = y_train[:n_samples_divisable_by_batchSize]
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_dimensions = X.shape[0]
        self.n_samples = X.shape[1]
        self.layers = None
        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)
        self.learning_rate = learning_rate
        self.activation_function_middleLayers = activation_function_middleLayers
        self.activation_function_lastLayer = activation_function_lastLayer
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.procedure = procedure
        self.loss_function_middleLayers = loss_function_middleLayers
        self.loss_function_lastLayer = loss_function_lastLayer
        self.dataset_name = dataset_name
        self.weightDecay_parameter = weightDecay_parameter
        if self.n_neurons[-1] == 1:
            assert self.activation_function_lastLayer != "softmax"  #--> softmax is for more than two classes
        self.do_kernel = do_kernel
        self.kernel = kernel
        if self.do_kernel:
            kernel_training_training = pairwise_kernels(X=self.X.T, Y=self.X.T, metric=self.kernel)
            self.kernel_training_training = self.normalize_the_kernel(kernel_matrix=kernel_training_training)

    def initialize_weights(self):
        U = [None] * self.n_layers
        for layer in range(self.n_layers):
            if layer == 0:
                if not self.do_kernel:
                    temp = self.n_dimensions
                else:
                    temp = self.n_samples
            else:
                temp = self.n_neurons[layer-1]
            U[layer] = np.random.rand(temp, self.n_neurons[layer])  # --> rand in [0,1)
        return U

    def backprojection(self, max_epochs=None, step_checkpoint=10, save_variable_images=False, save_embedding_images=False):
        path_save_base = './saved_files/'+ self.dataset_name +'/backprojection/'
        U = self.initialize_weights()
        epoch_index = -1
        objective_value_epochs = np.zeros((step_checkpoint, 1))
        time_of_epoch_average = 0
        while True:
            epoch_index = epoch_index + 1
            print("================ epoch: " + str(epoch_index))
            start_time = time.time()
            loss_of_batch = 0
            for batch_index in range(self.n_batches):
                # print("======== batch: " + str(batch_index))
                if not self.do_kernel:
                    X_batch = self.X[:, (batch_index*self.batch_size) : ((batch_index+1)*self.batch_size)]
                else:
                    X_batch = self.kernel_training_training[:, (batch_index * self.batch_size): ((batch_index + 1) * self.batch_size)]
                Y_batch = self.Y[:, (batch_index*self.batch_size) : ((batch_index+1)*self.batch_size)]
                if self.procedure == "forward":
                    for layer in range(self.n_layers):
                        U, gradient = self.update_layer_weights(layer=layer, U=U, X_batch=X_batch, Y_batch=Y_batch)
                        # print("layer: " + str(layer) + ", gradient norm: " + str(LA.norm(gradient)))
                elif self.procedure == "backward":
                    last_layer_index = self.n_layers - 1
                    for layer in range(last_layer_index, -1, -1):
                        U, gradient = self.update_layer_weights(layer=layer, U=U, X_batch=X_batch, Y_batch=Y_batch)
                elif self.procedure == "forward_backward":
                    if epoch_index % 2 == 0:
                        for layer in range(self.n_layers):
                            U, gradient = self.update_layer_weights(layer=layer, U=U, X_batch=X_batch, Y_batch=Y_batch)
                    else:
                        last_layer_index = self.n_layers - 1
                        for layer in range(last_layer_index, -1, -1):
                            U, gradient = self.update_layer_weights(layer=layer, U=U, X_batch=X_batch, Y_batch=Y_batch)
                loss_of_batch += self.calculate_loss(U=U, X_batch=X_batch, Y_batch=Y_batch)
            objective_value = loss_of_batch / self.n_batches
            end_time = time.time()
            time_of_epoch = end_time - start_time
            time_of_epoch_average = (time_of_epoch + ((epoch_index+1)*time_of_epoch_average)) / ((epoch_index+1)+1)
            print("////////////// loss of epoch: " + str(objective_value) + ", time of epoch: " + str(time_of_epoch) + ", average time: " + str(time_of_epoch_average))
            index_to_save = int(epoch_index % step_checkpoint)
            objective_value_epochs[index_to_save] = objective_value
            # save the information at checkpoints:
            if (epoch_index+1) % step_checkpoint == 0:
                print("Saving the checkpoint in epoch #" + str(epoch_index))
                checkpoint_index = int(np.floor(epoch_index / step_checkpoint))
                self.save_variable(variable=objective_value_epochs, name_of_variable="objective_value_epoch_"+str(epoch_index), path_to_save=path_save_base + 'objective_value/')
                self.save_np_array_to_txt(variable=objective_value_epochs, name_of_variable="objective_value_epoch_"+str(epoch_index), path_to_save=path_save_base + 'objective_value/')
                self.save_variable(variable=U, name_of_variable="U_epoch_"+str(epoch_index), path_to_save=path_save_base + 'U/')
                if save_variable_images:
                    self.save_image_of_weights(U, path_to_save=path_save_base + "/U_figs/epoch_"+str(epoch_index)+"/")
                if save_embedding_images and (len(self.y_train.shape) == 1):  #--> self.y_train should not be one-hot encoded
                    self.save_image_of_embedding_space(X_train=self.X, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test, U=U,
                                                       path_to_save=path_save_base + 'embedding_space/', epoch_index=epoch_index)
            # termination check:
            if max_epochs != None:
                if epoch_index >= max_epochs:
                    break


    def calculate_loss(self, U, X_batch, Y_batch):
        X_batch_encoded = self.test_the_trained_network(U=U, X=X_batch)
        if self.loss_function_lastLayer == "MSE":
            loss = LA.norm(X_batch_encoded - Y_batch)
        elif self.loss_function_lastLayer == "cross_entropy":
            # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
            n_neurons = Y_batch.shape[0]
            if n_neurons != 1: #--> for softmax activation function
                loss = 0
                for sample_index_in_batch in range(Y_batch.shape[1]):
                    X_batch_encoded[np.where(X_batch_encoded == 0)] = 0.001  # --> for avoiding division by zero
                    loss += -1 * np.sum(np.multiply(Y_batch[:, sample_index_in_batch], np.log(X_batch_encoded[:, sample_index_in_batch])))
            else: #--> for sigmoid activation function
                loss = 0
                for sample_index_in_batch in range(Y_batch.shape[1]):
                    X_batch_encoded[np.where(X_batch_encoded == 0)] = 0.001  # --> for avoiding division by zero
                    X_batch_encoded[np.where(X_batch_encoded == 1)] = 1 - 0.001  # --> for avoiding division by zero
                    loss += -1 * ( (Y_batch[:, sample_index_in_batch] * np.log(X_batch_encoded[:, sample_index_in_batch])) +
                                   ((1 - Y_batch[:, sample_index_in_batch]) * np.log(1 - X_batch_encoded[:, sample_index_in_batch])) )
        return loss

    def test_the_trained_network(self, U, X):
        last_layer_index = self.n_layers - 1
        for layer_index in range(self.n_layers):
            Z = U[layer_index].T @ X
            # print(Z)
            # input("p[")
            # print(str(layer_index) + ", " + str(LA.norm(X)))
            if layer_index != last_layer_index:
                X = self.get_activation_value(Z, method=self.activation_function_middleLayers)
                # print(LA.norm(X))
            else:
                # print(LA.norm(U[layer_index]))
                # print(LA.norm(Z))
                X = self.get_activation_value(Z, method=self.activation_function_lastLayer)
                # print("****, " + str(LA.norm(X)))
        # print(X)
        # input("ko")
        return X

    def update_layer_weights(self, layer, U, X_batch, Y_batch):
        ####### from first layer until the layer:
        X = X_batch
        for layer_index in range(layer):  # goes from the first layer until one layer before the layer
            Z = U[layer_index].T @ X
            X = self.get_activation_value(Z, method=self.activation_function_middleLayers)
            # print(X)
            # input("lo")
        ####### from last layer until the layer:
        Y = Y_batch
        last_layer_index = self.n_layers - 1
        for layer_index in range(last_layer_index, layer, -1):  # goes from the last layer until the layer itself
            if layer_index == last_layer_index:
                activation_inverse_value = self.get_activation_inverse(Y, method=self.activation_function_lastLayer)
            else:
                activation_inverse_value = self.get_activation_inverse(Y, method=self.activation_function_middleLayers)
            Y = U[layer_index] @ activation_inverse_value
        ####### a step of gradient descent:
        gradient = self.calculate_gradient(X_layerBefore=X, Y_layer=Y, U=U, layer=layer)
        U[layer] = U[layer] - (self.learning_rate * gradient) \
                            - (self.learning_rate * self.weightDecay_parameter * 2 * U[layer])
        return U, gradient

    def calculate_gradient(self, X_layerBefore, Y_layer, U, layer):
        Z = U[layer].T @ X_layerBefore
        #######
        if layer == self.n_layers-1:  #--> if last layer
            # print("..." + str(Z))
            activation_value = self.get_activation_value(Z, method=self.activation_function_lastLayer)
            # print("lop")
            activation_derivative_value = self.get_activation_derivative(Z, method=self.activation_function_lastLayer)
            # print("..." + str(activation_derivative_value))
        else:
            # print("...2 " + str(Z))
            activation_value = self.get_activation_value(Z, method=self.activation_function_middleLayers)
            activation_derivative_value = self.get_activation_derivative(Z, method=self.activation_function_middleLayers)
        #######
        if layer == 0:
            if not self.do_kernel:
                dd = self.n_dimensions
            else:
                dd = self.n_samples
        else:
            dd = self.n_neurons[layer - 1]
        gradient = np.zeros((dd, self.n_neurons[layer]))
        for sample_index_in_batch in range(self.batch_size):
            if layer == self.n_layers-1:  #-> if last layer
                if self.loss_function_lastLayer == "MSE":
                    temp1 = 2 * (activation_value[:, sample_index_in_batch] - Y_layer[:, sample_index_in_batch])
                elif self.loss_function_lastLayer == "cross_entropy":
                    # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
                    n_neurons = Y_layer.shape[0]
                    if n_neurons != 1: #--> for softmax activation function
                        activation_value[np.where(activation_value==0)] = 0.001  #--> for avoiding division by zero
                        temp1 = -1 * np.multiply( Y_layer[:, sample_index_in_batch], 1 / activation_value[:, sample_index_in_batch] )
                    else: #--> for sigmoid activation function
                        activation_value[np.where(activation_value==0)] = 0.001  #--> for avoiding division by zero
                        activation_value[np.where(activation_value==1)] = 1 - 0.001  #--> for avoiding division by zero
                        temp1 = -1 * ( (Y_layer[:, sample_index_in_batch] * (1 / activation_value[:, sample_index_in_batch])) -
                                       ((1 - Y_layer[:, sample_index_in_batch]) * (1/(1 - activation_value[:, sample_index_in_batch]))) )
                        # if np.isnan(temp1):
                        #     print(temp1)
                        #     print(Y_layer[:, sample_index_in_batch])
                        #     print(activation_value[:, sample_index_in_batch])
                        #     input("jojojoj")
            else:
                if self.loss_function_middleLayers == "MSE":
                    temp1 = 2 * (activation_value[:, sample_index_in_batch] - Y_layer[:, sample_index_in_batch])
            if ((layer == self.n_layers - 1) and (self.activation_function_lastLayer == "softmax")) or \
                    ((layer != self.n_layers - 1) and (self.activation_function_middleLayers == "softmax")):
                temp2 = activation_derivative_value[sample_index_in_batch]
            else:
                temp2 = np.diag(activation_derivative_value[:, sample_index_in_batch].ravel())
            # print(temp2)
            # temp2 = np.diag(activation_derivative_value[:, sample_index_in_batch].ravel())
            temp3 = np.kron(np.eye(self.n_neurons[layer]), X_layerBefore[:, sample_index_in_batch].T)
            gradient_vectorized = (temp3.T) @ (temp2) @ (temp1)
            gradient += gradient_vectorized.reshape((dd, self.n_neurons[layer]))
        return gradient

    def get_activation_value(self, z, method):
        if method == "sigmoid":
            f = 1 / (1 + np.exp(-1 * z))
        elif method == "linear":
            f = z
            cut_off = 10
            f[f >= cut_off] = cut_off
            f[f == np.inf] = cut_off
            f[f <= -1 * cut_off] = -1 * cut_off
            f[f == -1 * np.inf] = -1 * cut_off
        elif method == "softplus":
            # https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
            # http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf
            f = np.log(1 + np.exp(z))
            cut_off = 10
            f[f >= cut_off] = cut_off
            f[f == np.inf] = cut_off
        elif method == "tanh":
            cut_off = 10
            z[z >= cut_off] = cut_off
            z[z <= -1 * cut_off] = -1 * cut_off
            f = (np.exp(z) - np.exp(-1 * z)) / (np.exp(z) + np.exp(-1 * z))
        elif method == "elu":
            # https://en.wikipedia.org/wiki/Activation_function
            # https://www.tensorflow.org/api_docs/python/tf/nn/elu
            f = np.zeros(z.shape)
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    if z[i, j] <= 0:
                        f[i, j] = np.exp(z[i, j]) - 1
                    else:
                        f[i, j] = z[i, j]
            cut_off = 10
            f[f >= cut_off] = cut_off
            f[f == np.inf] = cut_off
        elif method == "softmax":
            # https://en.wikipedia.org/wiki/Activation_function
            # https://www.quora.com/Why-is-it-better-to-use-Softmax-function-than-sigmoid-function
            # cut_off = 100
            # z[z >= cut_off] = cut_off
            # z[z <= -1 * cut_off] = -1 * cut_off
            f = np.zeros(z.shape)
            for sample_in_batch in range(z.shape[1]):
                # f[:, sample_in_batch] = np.exp(z[:, sample_in_batch]) / np.sum(np.exp(z[:, sample_in_batch]))
                f[:, sample_in_batch] = softmax(z[:, sample_in_batch])
            # print(z)
            # print(f)
            # input("hi----")
            # cut_off = 10
            # f[f >= cut_off] = cut_off
            # f[f == np.inf] = cut_off
        return f

    def get_activation_derivative(self, z, method):
        if method == "sigmoid":
            # https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
            temp = self.get_activation_value(z, method="sigmoid")
            f = np.multiply(temp, 1-temp)
        elif method == "linear":
            f = np.ones(z.shape)
        elif method == "softplus":
            # cut_off = 10
            # z[z >= cut_off] = cut_off
            # temp = np.exp(z)
            # f = np.multiply(temp, 1/(1+temp))
            temp = np.exp(-1 * z)
            f = 1 / (1 + temp)
        elif method == "tanh":
            # https://en.wikipedia.org/wiki/Activation_function
            temp = self.get_activation_value(z, method="tanh")
            f = 1 - (temp ** 2)
        elif method == "elu":
            f = np.zeros(z.shape)
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    if z[i, j] <= 0:
                        f[i, j] = np.exp(z[i, j])
                    else:
                        f[i, j] = 1
        elif method == "softmax":
            # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            # https://en.wikipedia.org/wiki/Activation_function
            n_samples_in_batch = z.shape[1]
            n_neurons_in_layer = z.shape[0]
            f = [None] * n_samples_in_batch
            for sample_in_batch in range(n_samples_in_batch):
                f[sample_in_batch] = np.zeros((n_neurons_in_layer, n_neurons_in_layer))
            for sample_in_batch in range(n_samples_in_batch):
                for input_neuron_index in range(n_neurons_in_layer):
                    for output_neuron_index in range(n_neurons_in_layer):
                        if input_neuron_index == output_neuron_index:
                            delta_ = 1
                        else:
                            delta_ = 0
                        # f[sample_in_batch][input_neuron_index, output_neuron_index] = z[input_neuron_index, sample_in_batch] * (delta_ - z[output_neuron_index, sample_in_batch])
                        temp = z[input_neuron_index, sample_in_batch] * (delta_ - z[output_neuron_index, sample_in_batch])
                        if temp >= 100:
                            temp = 100
                        elif temp <= -100:
                            temp = -100
                        f[sample_in_batch][input_neuron_index, output_neuron_index] = temp
        return f

    def get_activation_inverse(self, z, method):
        if method == "sigmoid":
            # logit
            # assert np.sum(z < 0) == 0 and np.sum(z > 1) == 0
            z[z < 0] = 0.001
            z[z > 1] = 1 - 0.001
            cut_off = 10
            f = np.log(z / (1 - z))
            f[f >= cut_off] = cut_off
            f[f <= -1*cut_off] = -1*cut_off
            f[f == np.inf] = cut_off
            f[f == -1*np.inf] = -1*cut_off
        elif method == "linear":
            f = z
            cut_off = 10
            f[f >= cut_off] = cut_off
            f[f == np.inf] = cut_off
            f[f <= -1 * cut_off] = -1 * cut_off
            f[f == -1 * np.inf] = -1 * cut_off
        elif method == "softplus":
            z[z < 0] = 0.001
            cut_off = 10
            f = np.log(np.exp(z) - 1)
            f[f >= cut_off] = cut_off
            f[f <= -1 * cut_off] = -1 * cut_off
            f[f == np.inf] = cut_off
            f[f == -1 * np.inf] = -1 * cut_off
        elif method == "tanh":
            # http://wwwf.imperial.ac.uk/metric/metric_public/functions_and_graphs/hyperbolic_functions/inverses.html
            z[z < -1] = -1 + 0.001
            z[z > 1] = 1 - 0.001
            cut_off = 10
            f = 0.5 * np.log((1 + z) / (1 - z))
            f[f >= cut_off] = cut_off
            f[f <= -1 * cut_off] = -1 * cut_off
            f[f == np.inf] = cut_off
            f[f == -1 * np.inf] = -1 * cut_off
        elif method == "elu":
            z[z < -1] = -1 + 0.001
            f = np.zeros(z.shape)
            for i in range(z.shape[0]):
                for j in range(z.shape[1]):
                    if z[i, j] <= 0:
                        f[i, j] = np.log(z[i, j] + 1)
                    else:
                        f[i, j] = z[i, j]
            cut_off = 10
            f[f >= cut_off] = cut_off
            f[f <= -1 * cut_off] = -1 * cut_off
            f[f == np.inf] = cut_off
            f[f == -1 * np.inf] = -1 * cut_off
        elif method == "softmax":
            # https://math.stackexchange.com/questions/2786600/invert-the-softmax-function
            z[z < 0] = 0.001
            n_samples_in_batch = z.shape[1]
            f = np.zeros(z.shape)
            for sample_in_batch in range(n_samples_in_batch):
                f[:, sample_in_batch] = np.log(z[:, sample_in_batch])
                f[:, sample_in_batch] = f[:, sample_in_batch] + np.log(np.sum(np.exp(f[:, sample_in_batch])))
            cut_off = 10
            f[f <= -1 * cut_off] = -1 * cut_off
            f[f == -1 * np.inf] = -1 * cut_off
        return f

    def sigmoid(self, z):
        f = 1 / (1 + np.exp(-1 * z))
        return f

    def sigmoid_derivative(self, z):
        # https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
        temp = self.sigmoid(z)
        f = np.multiply(temp, 1-temp)
        return f

    def sigmoid_inverse(self, z):
        # logit
        # assert np.sum(z < 0) == 0 and np.sum(z > 1) == 0
        z[z < 0] = 0.001
        z[z > 1] = 1 - 0.001
        cut_off = 1
        f = np.log(z / (1 - z))
        f[f >= cut_off] = cut_off
        f[f <= -1*cut_off] = -1*cut_off
        f[f == np.inf] = cut_off
        f[f == -1*np.inf] = -1*cut_off
        return f

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))

    def save_image_of_weights(self, U, path_to_save="./"):
        # U --> list of --> n_neurons_of_layer_previous * n_neurons_of_layer
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        for layer_index in range(len(U)):
            U_layer = U[layer_index]
            plt.imshow(U_layer, cmap='gray')
            plt.axis('off')
            plt.colorbar()
            # plt.show()
            plt.savefig(path_to_save+"layer_"+str(layer_index)+".png")
            plt.clf()
        plt.clf()
        plt.close()

    def save_image_of_embedding_space(self, X_train, X_test, y_train, y_test, U, path_to_save, epoch_index):
        X_train = X_train.T  #--> samples become row-wise
        if X_test is not None:
            X_test = X_test.T  # --> samples become row-wise
            X = np.vstack((X_train, X_test))
        else:
            X = X_train
        ####################### plot the dataset first:
        # cm_bright = plt.cm.RdBu
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            cm_bright = ListedColormap(['#FF0000', '#0000FF'])  # red and blue
        else:
            # cm_bright = ListedColormap(['#FF0000', '#0000FF', '#008000'])  # red, blue, green
            cm_bright = plt.cm.brg
        ax = plt.subplot()
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # Plot the testing points
        if X_test is not None:
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k', marker="s")
        ####################### plot the embedding space:
        if not self.do_kernel:
            h = .02  # step size in the mesh
        else:
            h = .1  # step size in the mesh
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        if not self.do_kernel:
            encoded_space = self.test_the_trained_network(U=U, X=np.c_[xx.ravel(), yy.ravel()].T)
        else:
            train_and_test_data = np.column_stack((self.X, np.c_[xx.ravel(), yy.ravel()].T))
            kernel_training_test = pairwise_kernels(X=train_and_test_data.T, Y=train_and_test_data.T, metric=self.kernel)
            kernel_training_test = self.normalize_the_kernel(kernel_matrix=kernel_training_test)
            kernel_training_test = kernel_training_test[:self.n_samples, self.n_samples:]
            encoded_space = self.test_the_trained_network(U=U, X=kernel_training_test)
        n_neurons_output = encoded_space.shape[0]
        if n_neurons_output > 1:
            predicted_class = self.transform_multi_neuron_to_label(Y_predicted=encoded_space)
            encoded_space = predicted_class.reshape((1, -1))
        # print(encoded_space)
        # input("jojo")
        # Put the result into a color plot
        encoded_space = encoded_space.reshape(xx.shape)
        if n_classes == 2:
            cm_for_space = plt.cm.RdBu
        else:
            cm_for_space = plt.cm.brg
        plt.contourf(xx, yy, encoded_space, cmap=cm_for_space, alpha=.8)
        ####################### show/save plot:
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        plt.colorbar()
        # plt.show()
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        plt.savefig(path_to_save + "epoch_" + str(epoch_index) + ".png")
        plt.clf()
        plt.close()

    def transform_multi_neuron_to_label(self, Y_predicted):
        batch_size = Y_predicted.shape[1]
        predicted_class = np.zeros((batch_size,))
        # print(Y_predicted.shape)
        # print(Y_predicted)
        # input("hio")
        for sample_index_in_batch in range(batch_size):
            predicted_class[sample_index_in_batch] = np.argmax(Y_predicted[:, sample_index_in_batch])
            # print(predicted_class[sample_index_in_batch])
            # print(Y_predicted[:, sample_index_in_batch])
            # input("hio")
        return predicted_class

    def normalize_the_kernel(self, kernel_matrix):
        diag_kernel = np.diag(kernel_matrix)
        k = (1 / np.sqrt(diag_kernel)).reshape((-1, 1))
        normalized_kernel_matrix = np.multiply(kernel_matrix, k.dot(k.T))
        return normalized_kernel_matrix


