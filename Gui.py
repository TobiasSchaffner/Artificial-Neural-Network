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
import sys
import Network

from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (QGridLayout, QLineEdit, QPushButton, QAction, QTextEdit, QMainWindow, QFileDialog,
                             QLabel, QApplication, QProgressBar)


FUNCTION = Network.ActivationFunction.SIGMOID
MOMENTUM = 0.3
INPUT_NEURONS = 2
HIDDEN_NEURONS = 3
OUTPUT_NEURONS = 1
X_SIZE = 800
Y_SIZE = 800


class ANNGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_gui()

    def init_gui(self):
        self.text_edit = QTextEdit()
        self.setCentralWidget(self.text_edit)
        self.statusBar()
        self.network = Network.Network(X_SIZE, Y_SIZE, FUNCTION, MOMENTUM, INPUT_NEURONS, OUTPUT_NEURONS, HIDDEN_NEURONS)

        self.progress_label = QLabel(self)
        self.progress_label.move(20, 120)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(20, 150, 200, 25)
        self.progress_bar.hide()

        open_training_file = QAction(QIcon('open.png'), 'Run train File', self)
        open_training_file.setShortcut('Ctrl+T')
        open_training_file.setStatusTip('Open new File')
        open_training_file.triggered.connect(self.load_train_file)

        open_start_value_file = QAction(QIcon('open.png'), 'Open start Values', self)
        open_start_value_file.setShortcut('Ctrl+S')
        open_start_value_file.setStatusTip('Open new File')
        open_start_value_file.triggered.connect(self.loadStartValues)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(open_training_file)
        file_menu.addAction(open_start_value_file)

        self.evaluate_button = QPushButton('Evaluate', self)
        self.evaluate_button.move(20, 50)
        self.evaluate_button.clicked.connect(self.evaluate)

        self.result = QLabel(self)
        self.result.setText("Result: " + str(self.network.result)[:8])
        self.result.move(X_SIZE - 100, Y_SIZE - 100)

        self.init_grids()

        self.setGeometry(300, 300, X_SIZE, Y_SIZE)
        self.setWindowTitle('Artificial Neural Network')
        self.show()

    def evaluate(self):
        """ Run a forward- followed by a backpropagation """

        self.input()
        self.network.forward_propagation()
        self.network.backward_propagation()
        self.update_grids()

    def input(self):
        """ Get the input set by the user in the input fields """

        for column in (0, len(self.network.neurons) - 1):
            for row in range(len(self.network.neurons[column])):
                self.network.neurons[column][row].value = float(self.value_grid.itemAtPosition(column, row).widget().text())

    def update_grids(self):
        """ Update all lables with the new values calculated by the network """

        for column in range(len(self.network.neurons)):
            for row in range(len(self.network.neurons[column])):
                value = self.value_grid.itemAtPosition(column, row).widget()
                value.setText(str(self.network.neurons[column][row].value)[:8])
                value.setFixedSize(value.fontMetrics().boundingRect(value.text()).width() + 10, value.fontMetrics().boundingRect(value.text()).height())
                value.move(self.network.neurons[column][row].x_position - value.fontMetrics().boundingRect(
                    value.text()).width() / 2,
                         self.network.neurons[column][row].y_position - value.fontMetrics().boundingRect(
                             value.text()).height() / 2)

                value.update()
                for weight in range(len(self.network.neurons[column][row].weights)):
                    wgt = self.weight_grid.itemAtPosition(column, row * len(self.network.neurons[column + 1]) + weight).widget()
                    wgt.setText(str(self.network.neurons[column][row].weights[weight])[:8])
                    wgt.update()
        self.result.setText("Result: " + str(self.network.result)[:8])
        self.result.update()

    def init_grids(self):
        """ Init all lables and pictures for the setup given """

        neuron_picture = QPixmap("circle.png")
        self.neuron_picture_grid = QGridLayout()
        self.value_grid = QGridLayout()
        self.weight_grid = QGridLayout()

        for column in range(len(self.network.neurons)):
            for row in range(len(self.network.neurons[column])):
                pic = QLabel(self)
                pic.setPixmap(neuron_picture)
                pic.setFixedSize(neuron_picture.size())
                pic.move(self.network.neurons[column][row].x_position - neuron_picture.width()/2,
                         self.network.neurons[column][row].y_position - neuron_picture.height()/2)
                self.neuron_picture_grid.addWidget(pic, column, row)

                if column != 0 and column != len(self.network.neurons) - 1:
                    lbl = QLabel(self)
                    lbl.setStyleSheet("border: none;")
                else:
                    lbl = QLineEdit(self)
                lbl.setText(str(self.network.neurons[column][row].value)[:8])
                lbl.setFixedSize(lbl.fontMetrics().boundingRect(lbl.text()).width() + 10, lbl.fontMetrics().boundingRect(lbl.text()).height())
                lbl.move(self.network.neurons[column][row].x_position - lbl.fontMetrics().boundingRect(lbl.text()).width()/2,
                         self.network.neurons[column][row].y_position - lbl.fontMetrics().boundingRect(lbl.text()).height()/2)
                self.value_grid.addWidget(lbl, column, row)

                for weight in range(len(self.network.neurons[column][row].weights)):
                    wgt = QLabel(self)
                    wgt.setText(str(self.network.neurons[column][row].weights[weight])[:8])
                    x_position = self.network.neurons[column][row].x_position + 70
                    y_position = self.network.neurons[column][row].y_position + \
                                 0.2 * (self.network.neurons[column + 1][weight].y_position -
                                        self.network.neurons[column][row].y_position)
                    wgt.move(x_position, y_position - wgt.fontMetrics().boundingRect(wgt.text()).height()/2)
                    self.weight_grid.addWidget(wgt, column, row * len(self.network.neurons[column + 1]) + weight)

    def load_train_file(self):
        """ Load and execute a train file. Each line is a set of comma separated input and output values.
        Empty lines and lines with a # in front will be ignored. A *NUMBER will state how often the lines will be
        executed. """

        file_name = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        if file_name[0]:
            file = open(file_name[0], 'r')
            with file:
                lines = file.read().splitlines()
                data = []
                iterations = 1
                for line in lines:
                    if line.startswith('#') or len(line) == 0:
                        continue
                    elif line.startswith('*'):
                        iterations = int(line[1:])
                    else:
                        data.append(line)
                self.progress_label.setText("Loading: ")
                self.progress_bar.show()
                for i in range(iterations):
                    self.progress_bar.setValue((i/iterations)*100)
                    for dataset in data:
                        values = dataset.split(',')
                        if len(values) != len(self.network.neurons[0]) + \
                                len(self.network.neurons[len(self.network.neurons) - 1]):
                            print("Error! Train file and network do not fit for line: " + dataset + " expected length: " +
                                  str(len(values)) + " but was " + str(len(self.network.neurons[0]) +
                                  len(self.network.neurons[len(self.network.neurons) - 1])))
                            continue
                        value_index = 0
                        for column in (0, len(self.network.neurons) - 1):
                            for row in range(len(self.network.neurons[column])):
                                self.network.neurons[column][row].value = float(values[value_index])
                                value_index += 1
                        self.network.forward_propagation()
                        self.network.backward_propagation()
        self.progress_label.hide()
        self.progress_bar.hide()
        self.update_grids()

    def loadStartValues(self):
        """ Load a set of start values. The User has to care him self that the values provided fit to the network that
        was loaded. Each line is a comma separated Neuron with format value,weight[0],weight[1]...
        First line is first neuron in first colum, second line ist second Neuron in first column..."""

        file_name = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        if file_name[0]:
            f = open(file_name[0], 'r')
            with f:
                lines = f.read().splitlines()
                line_index = 0
                for column in range(len(self.network.neurons)):
                    for row in range(len(self.network.neurons[column])):
                        while lines[line_index].startswith('#') or len(lines[line_index]) == 0:
                            line_index += 1
                        values = lines[line_index].split(',')
                        self.network.neurons[column][row].value = float(values[0])
                        self.network.neurons[column][row].weights = list(map(float, values[1:]))
                        line_index += 1
        self.update_grids()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ANNGui()
    sys.exit(app.exec_())