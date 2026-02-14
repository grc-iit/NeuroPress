/**
 * simple_nn.cu — A minimal neural network built entirely on the GPU using CUDA.
 *
 * Network: 2 inputs → 4 hidden neurons (ReLU) → 1 output (Sigmoid)
 * Task:    Learn the XOR function
 *
 * This demonstrates the fundamental GPU kernels behind neural networks:
 *   1. Forward pass  (matrix multiply + activation)
 *   2. Loss compute  (mean squared error)
 *   3. Backward pass (gradient computation)
 *   4. Weight update  (gradient descent)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <curand_kernel.h>

// ============================================================================
// SECTION 1: GPU KERNELS — the building blocks
// ============================================================================

/**
 * KERNEL: Forward pass for a fully connected layer.
 *
 * Each thread computes ONE output neuron for ONE sample:
 *   out[sample][neuron] = bias[neuron] + sum(input[sample][k] * weight[k][neuron])
 *
 * Think of it like this: if you have 4 hidden neurons and 4 samples,
 * you launch 16 threads — each one independently computes its dot product.
 *
 * @param input       Input matrix        [num_samples x input_dim]
 * @param weights     Weight matrix       [input_dim x output_dim]
 * @param bias        Bias vector         [output_dim]
 * @param output      Output matrix       [num_samples x output_dim]
 * @param num_samples Number of samples (batch size)
 * @param input_dim   Number of input features
 * @param output_dim  Number of output neurons
 */
__global__ void forward_linear(const float* input, const float* weights,
                               const float* bias, float* output,
                               int num_samples, int input_dim, int output_dim) {
    int sample = blockIdx.x;   // which sample in the batch
    int neuron = threadIdx.x;  // which output neuron

    if (sample < num_samples && neuron < output_dim) {
        float sum = bias[neuron];
        for (int k = 0; k < input_dim; k++) {
            // input is [num_samples x input_dim], row-major
            // weights is [input_dim x output_dim], row-major
            sum += input[sample * input_dim + k] * weights[k * output_dim + neuron];
        }
        output[sample * output_dim + neuron] = sum;
    }
}

/**
 * KERNEL: ReLU activation — the simplest nonlinearity.
 *
 * ReLU(x) = max(0, x)
 *
 * Why do we need this? Without a nonlinear activation, stacking linear layers
 * is equivalent to a single linear layer. ReLU breaks that linearity,
 * allowing the network to learn complex patterns like XOR.
 *
 * Each thread handles one element — embarrassingly parallel.
 */
__global__ void relu_forward(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

/**
 * KERNEL: Sigmoid activation for the output layer.
 *
 * sigmoid(x) = 1 / (1 + exp(-x))
 *
 * Maps any value to the range (0, 1) — perfect for binary classification.
 * Our XOR output is 0 or 1, so sigmoid is the right choice here.
 */
__global__ void sigmoid_forward(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

/**
 * KERNEL: Compute Mean Squared Error loss AND the output gradient in one pass.
 *
 * MSE Loss = (1/N) * sum((predicted - target)^2)
 *
 * The gradient of MSE w.r.t. the prediction is simply:
 *   d_loss/d_pred = 2 * (predicted - target) / N
 *
 * But since we used sigmoid, we also multiply by the sigmoid derivative:
 *   sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 *
 * So: output_grad = 2*(pred - target)/N * pred*(1-pred)
 *
 * Each thread handles one sample.
 *
 * @param predictions  Network output      [num_samples x 1]
 * @param targets      Ground truth        [num_samples x 1]
 * @param output_grad  Gradient to propagate back [num_samples x 1]
 * @param loss         Single float to accumulate total loss (atomicAdd)
 * @param num_samples  Batch size
 */
__global__ void compute_loss_and_output_grad(const float* predictions,
                                              const float* targets,
                                              float* output_grad,
                                              float* loss,
                                              int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        float pred = predictions[idx];
        float target = targets[idx];
        float diff = pred - target;

        // Accumulate loss (atomic because multiple threads write to same location)
        atomicAdd(loss, diff * diff / num_samples);

        // Gradient: MSE derivative * sigmoid derivative
        float sigmoid_grad = pred * (1.0f - pred);
        output_grad[idx] = 2.0f * diff / num_samples * sigmoid_grad;
    }
}

/**
 * KERNEL: Backward pass — compute gradient of loss w.r.t. weights and biases.
 *
 * This is the "learning" step. Using the chain rule:
 *   d_loss/d_weight[k][neuron] = sum over samples: input[s][k] * grad_output[s][neuron]
 *   d_loss/d_bias[neuron]      = sum over samples: grad_output[s][neuron]
 *
 * Each thread computes the gradient for ONE weight element.
 *
 * @param input         Input to this layer        [num_samples x input_dim]
 * @param output_grad   Gradient from next layer   [num_samples x output_dim]
 * @param weight_grad   Computed weight gradients   [input_dim x output_dim]
 * @param bias_grad     Computed bias gradients     [output_dim]
 */
__global__ void backward_linear(const float* input, const float* output_grad,
                                float* weight_grad, float* bias_grad,
                                int num_samples, int input_dim, int output_dim) {
    int k = blockIdx.x;        // input dimension index
    int neuron = threadIdx.x;  // output dimension index

    if (k < input_dim && neuron < output_dim) {
        float grad_w = 0.0f;
        for (int s = 0; s < num_samples; s++) {
            grad_w += input[s * input_dim + k] * output_grad[s * output_dim + neuron];
        }
        weight_grad[k * output_dim + neuron] = grad_w;
    }

    // First "row" of threads also computes bias gradient
    if (k == 0 && neuron < output_dim) {
        float grad_b = 0.0f;
        for (int s = 0; s < num_samples; s++) {
            grad_b += output_grad[s * output_dim + neuron];
        }
        bias_grad[neuron] = grad_b;
    }
}

/**
 * KERNEL: Compute gradient to pass to the previous layer.
 *
 * input_grad[sample][k] = sum over neurons: output_grad[sample][neuron] * weight[k][neuron]
 *
 * This propagates the error signal backwards through the network.
 */
__global__ void backward_input_grad(const float* output_grad, const float* weights,
                                     float* input_grad,
                                     int num_samples, int input_dim, int output_dim) {
    int sample = blockIdx.x;
    int k = threadIdx.x;

    if (sample < num_samples && k < input_dim) {
        float grad = 0.0f;
        for (int neuron = 0; neuron < output_dim; neuron++) {
            grad += output_grad[sample * output_dim + neuron] * weights[k * output_dim + neuron];
        }
        input_grad[sample * input_dim + k] = grad;
    }
}

/**
 * KERNEL: ReLU backward — zero out gradients where the input was <= 0.
 *
 * ReLU derivative: 1 if x > 0, else 0.
 * So the gradient just "passes through" where the neuron was active.
 */
__global__ void relu_backward(float* grad, const float* pre_relu_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (pre_relu_output[idx] <= 0.0f) {
            grad[idx] = 0.0f;
        }
    }
}

/**
 * KERNEL: Stochastic Gradient Descent (SGD) weight update.
 *
 * The simplest optimizer:  weight = weight - learning_rate * gradient
 *
 * Each thread updates one parameter — perfectly parallel.
 */
__global__ void sgd_update(float* params, const float* grads, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= lr * grads[idx];
    }
}

/**
 * KERNEL: Initialize weights with random values using cuRAND.
 *
 * Good initialization matters! We use small random values centered near zero.
 * (Xavier/He initialization would be better for deep nets, but this suffices.)
 */
__global__ void init_weights(float* data, int size, unsigned long seed, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = (curand_uniform(&state) - 0.5f) * scale;
    }
}


// ============================================================================
// SECTION 2: HOST CODE — orchestrating the GPU kernels
// ============================================================================

int main() {
    printf("=== Simple GPU Neural Network: Learning XOR ===\n\n");

    // --- Network architecture ---
    const int INPUT_DIM   = 2;  // XOR has 2 inputs
    const int HIDDEN_DIM  = 4;  // 4 hidden neurons (enough for XOR)
    const int OUTPUT_DIM  = 1;  // 1 output (0 or 1)
    const int NUM_SAMPLES = 4;  // all 4 XOR combinations

    // --- Training hyperparameters ---
    const float LEARNING_RATE = 2.0f;  // aggressive LR works for this small problem
    const int   NUM_EPOCHS    = 10000;

    // --- Training data (XOR) ---
    // Prepared on the host (CPU), then copied to device (GPU)
    float h_input[NUM_SAMPLES * INPUT_DIM] = {
        0.0f, 0.0f,   // → 0
        0.0f, 1.0f,   // → 1
        1.0f, 0.0f,   // → 1
        1.0f, 1.0f    // → 0
    };
    float h_targets[NUM_SAMPLES * OUTPUT_DIM] = {
        0.0f, 1.0f, 1.0f, 0.0f
    };

    // --- Allocate GPU memory ---
    // d_ prefix = device (GPU) memory
    float *d_input, *d_targets;

    // Layer 1 (input → hidden): weights [2x4], bias [4]
    float *d_W1, *d_b1, *d_W1_grad, *d_b1_grad;

    // Layer 2 (hidden → output): weights [4x1], bias [1]
    float *d_W2, *d_b2, *d_W2_grad, *d_b2_grad;

    // Intermediate values (needed for backward pass)
    float *d_hidden_linear;  // output of layer 1 BEFORE ReLU
    float *d_hidden;         // output of layer 1 AFTER ReLU
    float *d_output;         // final network output (after sigmoid)

    // Gradients flowing backwards
    float *d_output_grad;    // gradient at the output
    float *d_hidden_grad;    // gradient at the hidden layer

    // Loss value
    float *d_loss;

    // Allocate everything on the GPU
    cudaMalloc(&d_input,   NUM_SAMPLES * INPUT_DIM * sizeof(float));
    cudaMalloc(&d_targets, NUM_SAMPLES * OUTPUT_DIM * sizeof(float));

    cudaMalloc(&d_W1,      INPUT_DIM * HIDDEN_DIM * sizeof(float));   // 2x4 = 8 weights
    cudaMalloc(&d_b1,      HIDDEN_DIM * sizeof(float));               // 4 biases
    cudaMalloc(&d_W1_grad, INPUT_DIM * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_b1_grad, HIDDEN_DIM * sizeof(float));

    cudaMalloc(&d_W2,      HIDDEN_DIM * OUTPUT_DIM * sizeof(float));  // 4x1 = 4 weights
    cudaMalloc(&d_b2,      OUTPUT_DIM * sizeof(float));               // 1 bias
    cudaMalloc(&d_W2_grad, HIDDEN_DIM * OUTPUT_DIM * sizeof(float));
    cudaMalloc(&d_b2_grad, OUTPUT_DIM * sizeof(float));

    cudaMalloc(&d_hidden_linear, NUM_SAMPLES * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_hidden,        NUM_SAMPLES * HIDDEN_DIM * sizeof(float));
    cudaMalloc(&d_output,        NUM_SAMPLES * OUTPUT_DIM * sizeof(float));

    cudaMalloc(&d_output_grad,   NUM_SAMPLES * OUTPUT_DIM * sizeof(float));
    cudaMalloc(&d_hidden_grad,   NUM_SAMPLES * HIDDEN_DIM * sizeof(float));

    cudaMalloc(&d_loss, sizeof(float));

    // --- Copy training data to GPU ---
    cudaMemcpy(d_input,   h_input,   NUM_SAMPLES * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, NUM_SAMPLES * OUTPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);

    // --- Initialize weights randomly on the GPU ---
    init_weights<<<1, INPUT_DIM * HIDDEN_DIM>>>(d_W1, INPUT_DIM * HIDDEN_DIM, 7, 2.0f);
    init_weights<<<1, HIDDEN_DIM * OUTPUT_DIM>>>(d_W2, HIDDEN_DIM * OUTPUT_DIM, 13, 2.0f);
    cudaMemset(d_b1, 0, HIDDEN_DIM * sizeof(float));   // biases start at zero
    cudaMemset(d_b2, 0, OUTPUT_DIM * sizeof(float));

    printf("Network: %d → %d (ReLU) → %d (Sigmoid)\n", INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM);
    printf("Total parameters: %d\n", INPUT_DIM*HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM*OUTPUT_DIM + OUTPUT_DIM);
    printf("Learning rate: %.1f\n", LEARNING_RATE);
    printf("Training for %d epochs...\n\n", NUM_EPOCHS);

    // ====================================================================
    // TRAINING LOOP — this is where all the GPU kernels come together
    // ====================================================================
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {

        // Reset loss to zero
        cudaMemset(d_loss, 0, sizeof(float));

        // -------------------------------------------------------
        // FORWARD PASS: push data through the network
        // -------------------------------------------------------

        // Layer 1: linear transform (input × W1 + b1)
        //   Launch: 4 blocks (one per sample) × 4 threads (one per hidden neuron)
        forward_linear<<<NUM_SAMPLES, HIDDEN_DIM>>>(
            d_input, d_W1, d_b1, d_hidden_linear,
            NUM_SAMPLES, INPUT_DIM, HIDDEN_DIM
        );

        // Save pre-ReLU values (needed for backward pass)
        cudaMemcpy(d_hidden, d_hidden_linear,
                   NUM_SAMPLES * HIDDEN_DIM * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        // Apply ReLU activation
        relu_forward<<<1, NUM_SAMPLES * HIDDEN_DIM>>>(
            d_hidden, NUM_SAMPLES * HIDDEN_DIM
        );

        // Layer 2: linear transform (hidden × W2 + b2)
        forward_linear<<<NUM_SAMPLES, OUTPUT_DIM>>>(
            d_hidden, d_W2, d_b2, d_output,
            NUM_SAMPLES, HIDDEN_DIM, OUTPUT_DIM
        );

        // Apply Sigmoid activation
        sigmoid_forward<<<1, NUM_SAMPLES * OUTPUT_DIM>>>(
            d_output, NUM_SAMPLES * OUTPUT_DIM
        );

        // -------------------------------------------------------
        // COMPUTE LOSS + OUTPUT GRADIENT
        // -------------------------------------------------------
        compute_loss_and_output_grad<<<1, NUM_SAMPLES>>>(
            d_output, d_targets, d_output_grad, d_loss, NUM_SAMPLES
        );

        // -------------------------------------------------------
        // BACKWARD PASS: propagate gradients backwards
        // -------------------------------------------------------

        // Layer 2 gradients: how should W2 and b2 change?
        backward_linear<<<HIDDEN_DIM, OUTPUT_DIM>>>(
            d_hidden, d_output_grad, d_W2_grad, d_b2_grad,
            NUM_SAMPLES, HIDDEN_DIM, OUTPUT_DIM
        );

        // Compute gradient flowing into the hidden layer
        backward_input_grad<<<NUM_SAMPLES, HIDDEN_DIM>>>(
            d_output_grad, d_W2, d_hidden_grad,
            NUM_SAMPLES, HIDDEN_DIM, OUTPUT_DIM
        );

        // Apply ReLU derivative (zero out gradients where ReLU was inactive)
        relu_backward<<<1, NUM_SAMPLES * HIDDEN_DIM>>>(
            d_hidden_grad, d_hidden_linear, NUM_SAMPLES * HIDDEN_DIM
        );

        // Layer 1 gradients: how should W1 and b1 change?
        backward_linear<<<INPUT_DIM, HIDDEN_DIM>>>(
            d_input, d_hidden_grad, d_W1_grad, d_b1_grad,
            NUM_SAMPLES, INPUT_DIM, HIDDEN_DIM
        );

        // -------------------------------------------------------
        // WEIGHT UPDATE: apply gradients using SGD
        // -------------------------------------------------------
        sgd_update<<<1, INPUT_DIM * HIDDEN_DIM>>>(d_W1, d_W1_grad, LEARNING_RATE, INPUT_DIM * HIDDEN_DIM);
        sgd_update<<<1, HIDDEN_DIM>>>(d_b1, d_b1_grad, LEARNING_RATE, HIDDEN_DIM);
        sgd_update<<<1, HIDDEN_DIM * OUTPUT_DIM>>>(d_W2, d_W2_grad, LEARNING_RATE, HIDDEN_DIM * OUTPUT_DIM);
        sgd_update<<<1, OUTPUT_DIM>>>(d_b2, d_b2_grad, LEARNING_RATE, OUTPUT_DIM);

        // -------------------------------------------------------
        // LOGGING: print progress every 500 epochs
        // -------------------------------------------------------
        if (epoch % 500 == 0 || epoch == NUM_EPOCHS - 1) {
            float h_loss;
            cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            printf("Epoch %5d | Loss: %.6f\n", epoch, h_loss);
        }
    }

    // ====================================================================
    // RESULTS: see what the network learned
    // ====================================================================
    printf("\n=== Results after training ===\n");
    float h_output[NUM_SAMPLES];
    cudaMemcpy(h_output, d_output, NUM_SAMPLES * OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    const char* inputs[] = {"0,0", "0,1", "1,0", "1,1"};
    for (int i = 0; i < NUM_SAMPLES; i++) {
        printf("  Input: [%s] → Predicted: %.4f  (Target: %.0f)  %s\n",
               inputs[i], h_output[i], h_targets[i],
               (fabsf(h_output[i] - h_targets[i]) < 0.1f) ? "✓" : "✗");
    }

    // ====================================================================
    // CLEANUP: free GPU memory
    // ====================================================================
    cudaFree(d_input); cudaFree(d_targets);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W1_grad); cudaFree(d_b1_grad);
    cudaFree(d_W2); cudaFree(d_b2); cudaFree(d_W2_grad); cudaFree(d_b2_grad);
    cudaFree(d_hidden_linear); cudaFree(d_hidden); cudaFree(d_output);
    cudaFree(d_output_grad); cudaFree(d_hidden_grad);
    cudaFree(d_loss);

    printf("\nDone! All computation happened on the GPU.\n");
    return 0;
}
