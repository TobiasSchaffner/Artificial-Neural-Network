#!/usr/bin/python3
#
# Author: Tobias Schaffner
# Project: Artificial Neural Network
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import Neuron
import numpy as np
from enum import Enum


class ActivationFunction(Enum):
    """ Enum for the implemented activation functions """
    LINEAR = 0
    SIGMOID = 1

class Network:
    """ Implementation of a Network class holding a 2D Array of Neurons with columns: inputs x*hidden outputs """
    #TODO Propagations are atm only capable of one layer of neurons and one result

    # The width of the whole Network
    x_size = 0

    # The height of the whole Network
    y_size = 0

    # The function to apply after adding up all inputs
    activation_function_type = ActivationFunction.LINEAR

    # The momentum. This add a percentage of the last weight change to the new weight change
    momentum = 0

    # The neurons in a 2D Array
    neurons = []

    # The prediction of the Network
    result = 0.00

    def __init__(self, x_size, y_size, activation_function_type, momentum, input_neurons, output_neurons, hidden_neurons):
        self.x_size = x_size
        self.y_size = y_size
        self.activation_function_type = activation_function_type
        self.momentum = momentum
        for i in range(3):
            self.neurons.append([])
        for i in range(input_neurons):
            self.neurons[0].append(Neuron.Neuron(1 * (x_size / (len(self.neurons) + 1)),
                                                 ((i + 1) * y_size / (input_neurons + 1)),
                                                 hidden_neurons))

        for i in range(hidden_neurons):
            self.neurons[1].append(Neuron.Neuron(2 * (x_size / (len(self.neurons) + 1)),
                                                 ((i + 1) * y_size / (hidden_neurons + 1)),
                                                 output_neurons))

        for i in range(output_neurons):
            self.neurons[2].append(Neuron.Neuron(len(self.neurons) * (x_size / (len(self.neurons) + 1)),
                                                 ((i + 1) * y_size / (output_neurons + 1)),
                                                 0))

    def forward_propagation(self):
        """ Calculation of the values based on the weights """

        # Copy the target value
        self.result = self.neurons[len(self.neurons) - 1][0].value

        # reset all values to 0
        for column in range(len(self.neurons) - 1):
            for row in range(len(self.neurons[column + 1])):
                self.neurons[column + 1][row].sum = 0.0;

        # calculate weights and resulting values
        for column in range(len(self.neurons)):

            # Add weighted values
            for row in range(len(self.neurons[column])):
                for weight in range(len(self.neurons[column][row].weights)):
                    self.neurons[column + 1][weight].sum += self.neurons[column][row].value * self.neurons[column][row].weights[weight]

            # Apply Sigmoid Function
            for weight in range(len(self.neurons[column][0].weights)):
                self.neurons[column + 1][weight].value = self.activation_function(self.neurons[column + 1][weight].sum)

        # swap Target and result
        tmp = self.neurons[len(self.neurons) - 1][0].value
        self.neurons[len(self.neurons) - 1][0].value = self.result
        self.result = tmp

    def backward_propagation(self):
        """ Adjustment of the weights """

        weight_factor = self.activation_function_derived(self.neurons[len(self.neurons) - 1][0].sum) * (self.neurons[len(self.neurons) - 1][0].value - self.result)

        # Input Neuron Column with weights pointing to the hidden neurons
        for row in range(len(self.neurons[0])):
            for weight in range(len(self.neurons[0][row].weights)):
                self.neurons[0][row].weights[weight] += weight_factor * self.neurons[1][weight].weights[0] * self.activation_function_derived(self.neurons[1][weight].sum) + self.momentum * self.neurons[0][row].weights_delta[weight]

        # Hidden Neuron Column with weights pointing to the output neurons
        for row in range(len(self.neurons[len(self.neurons) - 2])):
            self.neurons[len(self.neurons) - 2][row].weights[0] += weight_factor * self.neurons[len(self.neurons) - 2][row].value + self.momentum * self.neurons[len(self.neurons) - 2][row].weights_delta[0]

    def activation_function(self, input):
        """ Applies the chosen Activation Function on the given input value """

        if self.activation_function_type == ActivationFunction.LINEAR:
            return input
        elif self.activation_function_type == ActivationFunction.SIGMOID:
            return self.sigmoid(input)
        else:
            print("Error! No such function")
            exit(1)

    def activation_function_derived(self, input):
        """ Applies the derived chosen Activation Function on the given input value """

        if self.activation_function_type == ActivationFunction.LINEAR:
            return 1
        elif self.activation_function_type == ActivationFunction.SIGMOID:
            return self.sigmoid_derived(input)
        else:
            print("Error! No such function")
            exit(1)

    def sigmoid(self, input):
        """ implementation of the sigmoid function """
        return 1 / (1 + np.exp(- input))

    def sigmoid_derived(self, input):
        """ implementation of the derived sigmoid function """
        return self.sigmoid(input)*(1-self.sigmoid(input))

    def __repr__(self):
        return "x_size: " + str(self.x_size) + "\ny_size: " + str(self.y_size) + str(self.neurons)