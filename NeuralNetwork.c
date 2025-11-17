#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_INPUTS 2
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 1
#define NUM_TRAINING 4
#define LEARNING_RATE 0.1
#define EPOCHS 10000

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid (for backpropagation)
double dSigmoid(double x) {
    return x * (1 - x);
}

// Initialize weights with random numbers between 0 and 1
double initWeight() {
    return (double)rand() / RAND_MAX;
}

int main() {
    srand(time(NULL));

    // XOR training data
    double training_inputs[NUM_TRAINING][NUM_INPUTS] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double training_outputs[NUM_TRAINING][NUM_OUTPUTS] = {
        {0},
        {1},
        {1},
        {0}
    };

    // Initialize weights and biases
    double hidden_weights[NUM_INPUTS][NUM_HIDDEN];
    double output_weights[NUM_HIDDEN][NUM_OUTPUTS];
    double hidden_bias[NUM_HIDDEN];
    double output_bias[NUM_OUTPUTS];

    for (int i = 0; i < NUM_INPUTS; i++)
        for (int j = 0; j < NUM_HIDDEN; j++)
            hidden_weights[i][j] = initWeight();

    for (int i = 0; i < NUM_HIDDEN; i++)
        for (int j = 0; j < NUM_OUTPUTS; j++)
            output_weights[i][j] = initWeight();

    for (int i = 0; i < NUM_HIDDEN; i++)
        hidden_bias[i] = initWeight();

    for (int i = 0; i < NUM_OUTPUTS; i++)
        output_bias[i] = initWeight();

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int sample = 0; sample < NUM_TRAINING; sample++) {

            // --- Forward pass ---
            double hidden_layer[NUM_HIDDEN];
            for (int h = 0; h < NUM_HIDDEN; h++) {
                double activation = hidden_bias[h];
                for (int i = 0; i < NUM_INPUTS; i++)
                    activation += training_inputs[sample][i] * hidden_weights[i][h];
                hidden_layer[h] = sigmoid(activation);
            }

            double output_layer[NUM_OUTPUTS];
            for (int o = 0; o < NUM_OUTPUTS; o++) {
                double activation = output_bias[o];
                for (int h = 0; h < NUM_HIDDEN; h++)
                    activation += hidden_layer[h] * output_weights[h][o];
                output_layer[o] = sigmoid(activation);
            }

            // --- Backpropagation ---

            // Calculate output layer error and delta
            double output_error[NUM_OUTPUTS];
            double output_delta[NUM_OUTPUTS];
            for (int o = 0; o < NUM_OUTPUTS; o++) {
                output_error[o] = training_outputs[sample][o] - output_layer[o];
                output_delta[o] = output_error[o] * dSigmoid(output_layer[o]);
            }

            // Calculate hidden layer error and delta
            double hidden_error[NUM_HIDDEN];
            double hidden_delta[NUM_HIDDEN];
            for (int h = 0; h < NUM_HIDDEN; h++) {
                hidden_error[h] = 0.0;
                for (int o = 0; o < NUM_OUTPUTS; o++) {
                    hidden_error[h] += output_delta[o] * output_weights[h][o];
                }
                hidden_delta[h] = hidden_error[h] * dSigmoid(hidden_layer[h]);
            }

            // --- Update weights and biases ---
            // Output weights
            for (int h = 0; h < NUM_HIDDEN; h++) {
                for (int o = 0; o < NUM_OUTPUTS; o++) {
                    output_weights[h][o] += LEARNING_RATE * output_delta[o] * hidden_layer[h];
                }
            }
            // Output biases
            for (int o = 0; o < NUM_OUTPUTS; o++) {
                output_bias[o] += LEARNING_RATE * output_delta[o];
            }

            // Hidden weights
            for (int i = 0; i < NUM_INPUTS; i++) {
                for (int h = 0; h < NUM_HIDDEN; h++) {
                    hidden_weights[i][h] += LEARNING_RATE * hidden_delta[h] * training_inputs[sample][i];
                }
            }
            // Hidden biases
            for (int h = 0; h < NUM_HIDDEN; h++) {
                hidden_bias[h] += LEARNING_RATE * hidden_delta[h];
            }
        }
    }

    // Print results
    printf("Results after training:\n");
    for (int sample = 0; sample < NUM_TRAINING; sample++) {
        double hidden_layer[NUM_HIDDEN];
        for (int h = 0; h < NUM_HIDDEN; h++) {
            double activation = hidden_bias[h];
            for (int i = 0; i < NUM_INPUTS; i++)
                activation += training_inputs[sample][i] * hidden_weights[i][h];
            hidden_layer[h] = sigmoid(activation);
        }
        double output_layer[NUM_OUTPUTS];
        for (int o = 0; o < NUM_OUTPUTS; o++) {
            double activation = output_bias[o];
            for (int h = 0; h < NUM_HIDDEN; h++)
                activation += hidden_layer[h] * output_weights[h][o];
            output_layer[o] = sigmoid(activation);
        }
        printf("Input: %.0f %.0f, Predicted: %.4f, Expected: %.0f\n",
                training_inputs[sample][0], training_inputs[sample][1],
                output_layer[0], training_outputs[sample][0]);
    }

    return 0;
}
