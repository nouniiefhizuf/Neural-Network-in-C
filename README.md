# Neural-Network-in-C

This project demonstrates how to implement and train a simple feedforward neural network from scratch in C, with no external libraries required. The primary goal is to teach a neural network the XOR logic function—a classic problem that demonstrates a network's ability to learn non-linear relationships.

Features
Pure C implementation, no machine learning libraries

Simple feedforward neural network (2 input nodes, 2 hidden nodes, 1 output node)

Sigmoid activation function and its derivative

Manual implementation of forward pass and backpropagation (gradient descent)

Fully commented code, easy to understand and extend

Problem: XOR Function
The network is trained to approximate the XOR function:

(0, 0) → 0

(0, 1) → 1

(1, 0) → 1

(1, 1) → 0

This problem requires a network with at least one hidden layer, as XOR is not linearly separable.

Code Structure
main.c: All core implementation, including:

Data definitions (inputs, outputs)

Weight/bias initialization

Forward pass

Backpropagation and parameter updates

Result printing for all input combinations

Getting Started
Prerequisites
GCC or any C99-compliant compiler

Compile and Run
bash
gcc -o xor_nn main.c -lm
./xor_nn
Expected Output
After training, the program prints the network's prediction for each XOR input, showing how well it learned the function. Example:

text
Results after training:
Input: 0 0, Predicted: 0.0231, Expected: 0
Input: 0 1, Predicted: 0.9765, Expected: 1
Input: 1 0, Predicted: 0.9762, Expected: 1
Input: 1 1, Predicted: 0.0243, Expected: 0
Implementation Details
Number of inputs: 2

Number of hidden nodes: 2

Number of outputs: 1

Learning rate: 0.1

Training epochs: 10,000

Activation function: Sigmoid

You can experiment with network size, learning rate, or training epochs to see different results.
