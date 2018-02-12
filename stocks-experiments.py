import math
from numpy import zeros
from numpy import random
from prettytable import PrettyTable
total = 0
def sigmoid(x):
  return math.tanh(x)

def deltaSigmoid(y):
  return 1.0 - y**2

class network:
  def __init__(self, inputs, hidden, outputs):
    # Initialize inputs as well as bias
    self.inputs = inputs + 1

    # Initialize hidden units
    self.hidden = hidden

    # Initialize output units
    self.outputs = outputs

    # Initialize activation for input nodes
    self.input_activations = [1.0] * self.inputs

    # Initialize activation for hidden units
    self.hidden_activations = [1.0] * self.hidden

    # Initialize activation for output units
    self.outputs_activations = [1.0] * self.outputs
    
    # Initialize weight matrix between inputs and hidden units
    self.input_weights = random.rand(self.inputs, self.hidden)

    # Initialize weight matrix between hidden units and outputs
    self.output_weights = random.rand(self.hidden, self.outputs)

    # Learning rate alpha
    self.alpha = 0.5

    # Iterations to train network for
    self.iterations = 1000

  def update(self, inputs):
    # Compute activation for all inputs except bias
    for input in range(self.inputs - 1):
      self.input_activations[input] = inputs[input]

    # Compute activation for all hidden units
    for hidden in range(self.hidden):
      sum = 0.0
      for input in range(self.inputs):
          sum += self.input_activations[input] * self.input_weights[input][hidden]
      self.hidden_activations[hidden] = sigmoid(sum)

    # Compute activation for all output units
    for output in range(self.outputs):
      sum = 0.0
      for hidden in range(self.hidden):
        sum += self.hidden_activations[hidden] * self.output_weights[hidden][output]
      self.outputs_activations[output] = sigmoid(sum)

    return self.outputs_activations

  def backpropagate(self, targets):
    # Compute error at output units
    output_deltas = [0.0] * self.outputs
    for output in range(self.outputs):
      error = targets[output] - self.outputs_activations[output]
      output_deltas[output] = deltaSigmoid(self.outputs_activations[output]) * error

    # Compute error at hidden units
    hidden_deltas = [0.0] * self.hidden
    for hidden in range(self.hidden):
      error = 0.0
      for output in range(self.outputs):
        error += output_deltas[output] * self.output_weights[hidden][output]
      hidden_deltas[hidden] = deltaSigmoid(self.hidden_activations[hidden]) * error

    # Update output unit weights
    for hidden in range(self.hidden):
      for output in range(self.outputs):
        update = output_deltas[output] * self.hidden_activations[hidden]
        self.output_weights[hidden][output] = self.output_weights[hidden][output] + self.alpha * update

    # Update input unit weights
    for input in range(self.inputs):
      for hidden in range(self.hidden):
        update = hidden_deltas[hidden] * self.input_activations[input]
        self.input_weights[input][hidden] = self.input_weights[input][hidden] + self.alpha * update

    # Compute total error
    error = 0.0
    for target in range(len(targets)):
      error += 0.5 * (targets[target] - self.outputs_activations[target]) ** 2
    return error

  def test(self, pattern):
    # for pattern in patterns:
    print(pattern, '->', self.update(pattern))
    return self.update(pattern)[0]


  def train(self, patterns):
    for iteration in range(self.iterations):
      error = 0.0
      for pattern in patterns:
        inputs = pattern[0]
        targets = pattern[1]
        self.update(inputs)
        error += self.backpropagate(targets)
      if iteration % 100 == 0:
        print('error %-.5f' % error)

def average(data):
  sum = 0.0
  for datum in data:
    sum += datum
  return sum / len(data)

def maximum(data):
  return max(data)

def minimum(data):
  return min(data)

def normalize(price, minimum, maximum):
  return ((2 * price - (maximum + minimum)) / (maximum - minimum))

def denormalize(normalized, minimum, maximum):
  return (((normalized * (maximum-minimum)) / 2) + (maximum + minimum)) / 2

def slidingWindow(data):
  windows = []
  beginning = 0
  end = 5
  i = 0
  while i < 20:
    windows.append(data[beginning:end])
    beginning += 1
    end += 1
    i += 1
  return windows

def run():
  # Closing prices from dec 15 -> jan22
  dataset=[1064.19,1077.14,1070.68,1064.95,1063.63,1060.12,1056.74,1049.37,1048.14,1046.4,1065,1082.48,1086.4,1102.23,1106.94,1106.26,1102.61,1105.52,1122.26,1121.76,1131.98,1129.79,1137.51,1155.81]

  # Compute 5-day sliding windows
  windows = slidingWindow(dataset)

  # Compute the average, minimum, maximum and normalized prices from windows
  computed_data = []
  for window in windows:
    # Prepare [average, minimum, maximum]
    data = []
    ave = average(window)
    min = minimum(window)
    max = maximum(window)
    data.append(ave)
    data.append(min)
    data.append(max)
    computed_data.append(data)

    # Prepare normalized closing value
    closing = []
    closing.append(normalize(window[0], min, max))
    computed_data.append(closing)
  training_data = [computed_data]

  # Create network architecture consisting of 3 input nodes, 2 hidden units, and 1 output unit
  n = network(3, 2, 1)

  # Train the network
  n.train(training_data)

    #prediction Data(jan23->feb 09)
  #prediction_dataset = [743.09, 751.72, 753.20, 759.00, 766.61, 753.93, 752.67, 759.14, 718.77, 723.15, 708.14, 705.84, 691.02, 693.01]
  prediction_dataset = [1169.97, 1164.24, 1170.37, 1175.84, 1175.58, 1163.69, 1169.94, 1167.7, 1111.9,1055.8, 1080.6, 1048.58, 1001.52, 1037.78]
  # Create 5-day sliding windows for prediction data
  complete_dataset = dataset + prediction_dataset

  prediction_windows = []
  for prediction in prediction_dataset:
    location = complete_dataset.index(prediction)
    prediction_windows.append(complete_dataset[location - 4: location + 1])
  
  prediction_data = []
  predictions = []
  # Compute prediction windows in the form of [average, minimum, maximum]
  for window in prediction_windows:
    data = []
    ave = average(window)
    min = minimum(window)
    max = maximum(window)
    data.append(ave)
    data.append(min)
    data.append(max)
    prediction_data.append(data)

  # Test the network
  for i in xrange(len(prediction_data)):
    value = n.test(prediction_data[i])
    predictions.append(denormalize(value, prediction_data[i][1], prediction_data[i][2]))
    # print 'Normalized Prediction: ' + str(denormalize(value, prediction_data[i][1], prediction_data[i][2]))

  table = PrettyTable()
  x = PrettyTable(['Date', 'Predicted', 'Actual', 'Error Percentage'])
  x.padding_width = 1
  x.add_row(['Jan 23th', '%-.2f' % predictions[0], prediction_dataset[0], '%-.2f' % (((prediction_dataset[0]-predictions[0]) / (prediction_dataset[0])) * 100)])
  x.add_row(['Jan 24th', '%-.2f' % predictions[1], prediction_dataset[1], '%-.2f' % (((prediction_dataset[1]-predictions[1]) / (prediction_dataset[1])) * 100)])
  x.add_row(['Jan 25th', '%-.2f' % predictions[2], prediction_dataset[2], '%-.2f' % (((prediction_dataset[2]-predictions[2]) / (predictions[2] + prediction_dataset[2])) * 100)])
  x.add_row(['Jan 26th', '%-.2f' % predictions[3], prediction_dataset[3], '%-.2f' % (((prediction_dataset[3]-predictions[3]) / (prediction_dataset[3])) * 100)])
  x.add_row(['Jan 27th', '%-.2f' % predictions[4], prediction_dataset[4], '%-.2f' % (((prediction_dataset[4]-predictions[4]) / (prediction_dataset[4])) * 100)])
  x.add_row(['Jan 28th', '%-.2f' % predictions[5], prediction_dataset[5], '%-.2f' % (((prediction_dataset[5]-predictions[5]) / (prediction_dataset[5])) * 100)])
  x.add_row(['Jan 30th', '%-.2f' % predictions[6], prediction_dataset[6], '%-.2f' % (((prediction_dataset[6]-predictions[6]) / (prediction_dataset[6])) * 100)])
  x.add_row(['Jan 31th', '%-.2f' % predictions[7], prediction_dataset[7], '%-.2f' % (((prediction_dataset[7]-predictions[7]) / (prediction_dataset[7])) * 100)])
  x.add_row(['feb 1th', '%-.2f' % predictions[8], prediction_dataset[8], '%-.2f' % (((prediction_dataset[8]-predictions[8]) / (prediction_dataset[8])) * 100)])
  x.add_row(['fec 2nd', '%-.2f' % predictions[9], prediction_dataset[9], '%-.2f' % (((prediction_dataset[9]-predictions[9]) / (prediction_dataset[9])) * 100)])
  x.add_row(['jan 3rd-', '%-.2f' % predictions[10], prediction_dataset[10], '%-.2f' % (((predictions[10]-prediction_dataset[10]) / (prediction_dataset[10])) * 100)])
  x.add_row(['Jan 04th', '%-.2f' % predictions[11], prediction_dataset[11], '%-.2f' % (((prediction_dataset[11]-predictions[11]) / (prediction_dataset[11])) * 100)])
  x.add_row(['Jan 6th', '%-.2f' % predictions[12], prediction_dataset[12], '%-.2f' % (((prediction_dataset[12]-predictions[12]) / (prediction_dataset[12])) * 100)])
  x.add_row(['Jan 9thth', '%-.2f' % predictions[13], prediction_dataset[13], '%-.2f' % (((prediction_dataset[13]-predictions[13]) / (prediction_dataset[13])) * 100)])
  print x

run()
