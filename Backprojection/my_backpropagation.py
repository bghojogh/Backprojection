import numpy as np
import numpy.linalg as LA
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

import warnings
warnings.filterwarnings("ignore")

class My_backpropagation:

    def __init__(self, X, Y, y_train, n_neurons, learning_rate, activation_function_middleLayers,
                 activation_function_lastLayer, batch_size,
                 loss_function_middleLayers, loss_function_lastLayer, dataset_name, weightDecay_parameter, X_test=None, y_test=None):
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
        self.loss_function_middleLayers = loss_function_middleLayers
        self.loss_function_lastLayer = loss_function_lastLayer
        self.dataset_name = dataset_name
        self.weightDecay_parameter = weightDecay_parameter
        if self.n_neurons[-1] == 1:
            assert self.activation_function_lastLayer != "softmax"  #--> softmax is for more than two classes

    def initialize_weights(self):
        U = [None] * self.n_layers
        for layer in range(self.n_layers):
            if layer == 0:
                temp = self.n_dimensions
            else:
                temp = self.n_neurons[layer-1]
            U[layer] = np.random.rand(temp, self.n_neurons[layer])  # --> rand in [0,1)
        return U

    def backpropagation(self, max_epochs=None, step_checkpoint=10, save_variable_images=False, save_embedding_images=False):
        path_save_base = './saved_files/'+ self.dataset_name +'/backpropagation/'
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
                X_batch = self.X[:, (batch_index*self.batch_size) : ((batch_index+1)*self.batch_size)]
                Y_batch = self.Y[:, (batch_index*self.batch_size) : ((batch_index+1)*self.batch_size)]
                for layer in range(self.n_layers-1, ):
                    U, gradient = self.update_network_weights(U=U, X_batch=X_batch, Y_batch=Y_batch)
                    # print("layer: " + str(layer) + ", gradient norm: " + str(LA.norm(gradient)))
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
            # print(str(layer_index) + ", " + str(LA.norm(X)))
            if layer_index != last_layer_index:
                X = self.get_activation_value(Z, method=self.activation_function_middleLayers)
                # print(LA.norm(X))
            else:
                # print(LA.norm(U[layer_index]))
                # print(LA.norm(Z))
                X = self.get_activation_value(Z, method=self.activation_function_lastLayer)
                # print("****, " + str(LA.norm(X)))
        return X

    def update_network_weights(self, U, X_batch, Y_batch):
        # feedforward:
        Z = [None] * self.n_layers
        X = X_batch
        for layer_index in range(self.n_layers):
            # Z[layer_index] = U[layer_index].T @ X
            temp_Z = U[layer_index].T @ X
            cut_off = 10
            temp_Z[temp_Z >= cut_off] = cut_off
            temp_Z[temp_Z <= -1 * cut_off] = -1 * cut_off
            Z[layer_index] = temp_Z
            # print(LA.norm(Z[layer_index]))
            if layer_index != (self.n_layers - 1):  # not last layer
                X = self.get_activation_value(Z[layer_index], method=self.activation_function_middleLayers)
            else:
                X = self.get_activation_value(Z[layer_index], method=self.activation_function_lastLayer)
        # input("jkl")
        # backpropagation:
        delta = [None] * self.n_layers
        for layer_index in range(len(delta)):
            delta[layer_index] = np.zeros((self.n_neurons[layer_index], self.batch_size))
        gradient = [None] * self.n_layers
        for layer_index in range(len(delta)):
            if layer_index != 0:
                gradient[layer_index] = np.zeros((self.n_neurons[layer_index-1], self.n_neurons[layer_index]))
            else:
                gradient[layer_index] = np.zeros((self.n_dimensions, self.n_neurons[layer_index]))
        last_layer_index = self.n_layers - 1
        for layer_index in range(last_layer_index, -1, -1):
            # print("lllllllllll;;;;;;;;;;;;;;" + str(layer_index))
            if layer_index != last_layer_index:
                activation_derivative_value = self.get_activation_derivative(Z[layer_index], method=self.activation_function_middleLayers)
                if layer_index != 0:
                    n_neurons_of_previous_layer = self.n_neurons[layer_index-1]
                else:
                    n_neurons_of_previous_layer = self.n_dimensions
                for neuron_index_1 in range(n_neurons_of_previous_layer):
                    temp1 = np.zeros((1, self.batch_size))
                    for neuron_index_2 in range(self.n_neurons[layer_index]):
                        temp1 += delta[layer_index][neuron_index_2, :] * U[layer_index][neuron_index_1, neuron_index_2]
                    if layer_index > 0:
                        delta[layer_index-1][neuron_index_1, :] = np.multiply(activation_derivative_value[neuron_index_1, :], temp1)
                    for neuron_index_2 in range(self.n_neurons[layer_index]):
                        if layer_index > 0:
                            gradient_of_batch = np.multiply(delta[layer_index][neuron_index_2, :], Z[layer_index-1][neuron_index_1, :])
                        else:
                            gradient_of_batch = np.multiply(delta[layer_index][neuron_index_2, :], X_batch[neuron_index_1, :])
                            # print(delta[layer_index+1])
                            # input("ho")
                        gradient[layer_index][neuron_index_1, neuron_index_2] = np.sum(gradient_of_batch)
                        # print(np.sum(gradient_of_batch))
                        # input("ho")
            else:
                activation_derivative_value = self.get_activation_derivative(Z[layer_index], method=self.activation_function_lastLayer)
                activation_value = self.get_activation_value(Z[layer_index], method=self.activation_function_lastLayer)
                if self.loss_function_lastLayer == "MSE":
                    derivative_error_wrt_activation = 2 * (activation_value[:, :] - Y_batch[:, :])
                elif self.loss_function_lastLayer == "cross_entropy":
                    # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
                    n_neurons = self.n_neurons[-1]
                    if n_neurons != 1: #--> for softmax activation function
                        activation_value[np.where(activation_value==0)] = 0.001  #--> for avoiding division by zero
                        derivative_error_wrt_activation = -1 * np.multiply( Y_batch[:, :], 1 / activation_value[:, :] )
                    else: #--> for sigmoid activation function
                        activation_value[np.where(activation_value==0)] = 0.001  #--> for avoiding division by zero
                        activation_value[np.where(activation_value==1)] = 1 - 0.001  #--> for avoiding division by zero
                        derivative_error_wrt_activation = -1 * ( (Y_batch[:, :] * (1 / activation_value[:, :])) -
                                                            ((1 - Y_batch[:, :]) * (1/(1 - activation_value[:, :]))) )
                # print(Z[layer_index].shape)
                # print(activation_derivative_value[0].shape)
                # print(activation_derivative_value[1].shape)
                # print(activation_derivative_value[2].shape)
                # print(derivative_error_wrt_activation[:, :].shape)
                delta[layer_index][:, :] = np.multiply(activation_derivative_value[:, :], derivative_error_wrt_activation[:, :])  #--> chain rule
                n_neurons_of_previous_layer = self.n_neurons[layer_index - 1]
                # print("lllllllllll;;;;;;;;;;;;;;" + str(layer_index))
                for neuron_index_1 in range(n_neurons_of_previous_layer):
                    for neuron_index_2 in range(self.n_neurons[layer_index]):
                        gradient_of_batch = np.multiply(delta[layer_index][neuron_index_2, :], Z[layer_index-1][neuron_index_1, :])
                        gradient[layer_index][neuron_index_1, neuron_index_2] = np.sum(gradient_of_batch)
                        # print(np.sum(gradient_of_batch))
                        # print(LA.norm(gradient_of_batch))
                # print(LA.norm(Z[layer_index-1]))
                # update the delta of previous layer of the last layer:
                activation_derivative_value_lastLayer = self.get_activation_derivative(Z[layer_index-1], method=self.activation_function_lastLayer)
                for neuron_index_1 in range(n_neurons_of_previous_layer):
                    temp1 = np.zeros((1, self.batch_size))
                    for neuron_index_2 in range(self.n_neurons[layer_index]):
                        temp1 += delta[layer_index][neuron_index_2, :] * U[layer_index][neuron_index_1, neuron_index_2]
                    delta[layer_index-1][neuron_index_1, :] = np.multiply(activation_derivative_value_lastLayer[neuron_index_1, :], temp1)
        # print(LA.norm(gradient[1]))
        ####### a step of gradient descent:
        for layer_index in range(self.n_layers):
            U[layer_index] = U[layer_index] - (self.learning_rate * gradient[layer_index]) \
                                            - (self.learning_rate * self.weightDecay_parameter * 2 * U[layer_index])
        return U, gradient

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
            cut_off = 10
            z[z >= cut_off] = cut_off
            z[z <= -1 * cut_off] = -1 * cut_off
            f = np.zeros(z.shape)
            for sample_in_batch in range(z.shape[1]):
                f[:, sample_in_batch] = np.exp(z[:, sample_in_batch]) / np.sum(np.exp(z[:, sample_in_batch]))
            # print(z)
            # print(f)
            # input("hi")
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
                        f[sample_in_batch][input_neuron_index, output_neuron_index] = z[input_neuron_index, sample_in_batch] * (delta_ - z[output_neuron_index, sample_in_batch])
            f_ = np.zeros((n_neurons_in_layer, n_samples_in_batch))
            for sample_in_batch in range(n_samples_in_batch):
                diag_ = np.diag(f[sample_in_batch])
                f_[:, sample_in_batch] = diag_
            f = f_
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
        h = .02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        encoded_space = self.test_the_trained_network(U=U, X=np.c_[xx.ravel(), yy.ravel()].T)
        # Put the result into a color plot
        n_neurons_output = encoded_space.shape[0]
        if n_neurons_output > 1:
            predicted_class = self.transform_multi_neuron_to_label(Y_predicted=encoded_space)
            encoded_space = predicted_class.reshape((1, -1))
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

        for sample_index_in_batch in range(batch_size):
            predicted_class[sample_index_in_batch] = np.argmax(Y_predicted[:, sample_index_in_batch])
            # print(predicted_class[sample_index_in_batch])
            # print(Y_predicted[:, sample_index_in_batch])
            # input("hio")

        return predicted_class

