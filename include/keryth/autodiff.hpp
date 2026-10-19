#pragma once

#include "keryth/runtime.hpp"
#include <cmath>
#include <vector>

namespace keryth {

struct MLPLayer {
    Tensor w;
    Tensor b;
    Tensor grad_w;
    Tensor grad_b;

    MLPLayer(int64_t in_dim, int64_t out_dim)
        : w({in_dim, out_dim}, 0.0f),
          b({out_dim}, 0.0f),
          grad_w({in_dim, out_dim}, 0.0f),
          grad_b({out_dim}, 0.0f) {
        // Initialize weights with He/Xavier uniform
        float scale = std::sqrt(2.0f / static_cast<float>(in_dim + out_dim));
        for (int64_t i = 0; i < w.numel(); ++i) {
            float val = ((static_cast<float>((i * 73 + 17) % 100) / 100.0f) - 0.5f) * scale;
            w.set_flat(i, val);
        }
    }
};

class MLPNetwork {
public:
    MLPNetwork(int64_t in_dim, int64_t hidden_dim, int64_t out_dim)
        : layer1(in_dim, hidden_dim), layer2(hidden_dim, out_dim) {}

    // Forward pass:
    // Z1 = X * W1
    // A1 = Z1 + b1
    // H  = relu(A1)
    // Z2 = H * W2
    // Y_hat = Z2 + b2
    Tensor forward(const Tensor& x) {
        x_saved = x;
        z1_saved = Tensor::matmul(x, layer1.w);
        a1_saved = Tensor::add(z1_saved, layer1.b);
        h_saved = Tensor::relu(a1_saved);
        z2_saved = Tensor::matmul(h_saved, layer2.w);
        y_hat_saved = Tensor::add(z2_saved, layer2.b);
        return y_hat_saved;
    }

    // MSE loss: 0.5 * mean((Y_hat - Y)^2)
    float compute_loss(const Tensor& y_hat, const Tensor& target) {
        assert(y_hat.shape() == target.shape());
        float loss = 0.0f;
        for (int64_t i = 0; i < y_hat.numel(); ++i) {
            float diff = y_hat.at_flat(i) - target.at_flat(i);
            loss += 0.5f * diff * diff;
        }
        return loss / static_cast<float>(y_hat.shape()[0]);
    }

    // Backward pass (Vector-Jacobian Product propagation):
    // dY_hat = (Y_hat - target) / batch_size
    // dW2 = H^T * dY_hat
    // db2 = sum_axis0(dY_hat)
    // dH  = dY_hat * W2^T
    // dA1 = relu_backward(A1, dH)
    // dW1 = X^T * dA1
    // db1 = sum_axis0(dA1)
    void backward(const Tensor& target) {
        int64_t batch_size = y_hat_saved.shape()[0];
        float inv_n = 1.0f / static_cast<float>(batch_size);

        Tensor d_y_hat(y_hat_saved.shape());
        for (int64_t i = 0; i < y_hat_saved.numel(); ++i) {
            d_y_hat.set_flat(i, (y_hat_saved.at_flat(i) - target.at_flat(i)) * inv_n);
        }

        // Layer 2 gradients
        layer2.grad_w = Tensor::matmul(h_saved.transpose(), d_y_hat);
        layer2.grad_b = d_y_hat.sum_axis0();

        // Propagate into hidden layer
        Tensor d_h = Tensor::matmul(d_y_hat, layer2.w.transpose());
        Tensor d_a1 = Tensor::relu_backward(a1_saved, d_h);

        // Layer 1 gradients
        layer1.grad_w = Tensor::matmul(x_saved.transpose(), d_a1);
        layer1.grad_b = d_a1.sum_axis0();
    }

    // Gradient descent step: theta <- theta - lr * grad_theta
    void step(float lr) {
        for (int64_t i = 0; i < layer1.w.numel(); ++i) {
            layer1.w.set_flat(i, layer1.w.at_flat(i) - lr * layer1.grad_w.at_flat(i));
        }
        for (int64_t i = 0; i < layer1.b.numel(); ++i) {
            layer1.b.set_flat(i, layer1.b.at_flat(i) - lr * layer1.grad_b.at_flat(i));
        }
        for (int64_t i = 0; i < layer2.w.numel(); ++i) {
            layer2.w.set_flat(i, layer2.w.at_flat(i) - lr * layer2.grad_w.at_flat(i));
        }
        for (int64_t i = 0; i < layer2.b.numel(); ++i) {
            layer2.b.set_flat(i, layer2.b.at_flat(i) - lr * layer2.grad_b.at_flat(i));
        }
    }

    MLPLayer layer1;
    MLPLayer layer2;

private:
    Tensor x_saved;
    Tensor z1_saved;
    Tensor a1_saved;
    Tensor h_saved;
    Tensor z2_saved;
    Tensor y_hat_saved;
};

} // namespace keryth
