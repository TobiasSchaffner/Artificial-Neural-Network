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
import numpy as np

class Neuron:

    # Position in the Network
    x_position = 0
    y_position = 0

    # The sum of weighted inputs
    sum = 0.0

    # The values after applying the activation function
    value = 0.0

    # The weights for synapses to the Neurons in the next column
    weights = []

    # The change in weight after a backpropagation
    weights_delta = []

    def __init__(self, x_position, y_position, size_of_next_step):
        self.x_position = x_position
        self.y_position = y_position
        self.weights = np.random.uniform(low=0.0, high= 1.0, size=size_of_next_step)
        self.weights_delta = np.zeros(size_of_next_step)

    def __repr__(self):
        return "\nvalue: " + str(self.value) + " x_position: " + str(self.x_position) + " y_position " + str(self.y_position) + " weights: " + str(self.weights) + " weights_delta: " + str(self.weights_delta)