#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "keryth/autodiff.hpp"

using namespace keryth;

void test_finite_difference_gradient_check() {
    MLPNetwork mlp(2, 4, 1);

    // Synthetic batch: 3 samples of dim 2
    Tensor x({3, 2}, {
        1.0f, 0.5f,
        -0.5f, 1.2f,
        0.0f, -1.0f
    });
    Tensor target({3, 1}, {
        1.0f,
        0.0f,
        1.0f
    });

    // 1. Compute analytical gradients via VJP backward pass
    Tensor y_hat = mlp.forward(x);
    mlp.backward(target);

    // 2. Numerical gradient check for each parameter in layer1.w
    float eps = 1e-3f;
    for (int64_t i = 0; i < mlp.layer1.w.numel(); ++i) {
        float orig = mlp.layer1.w.at_flat(i);

        // f(theta + eps)
        mlp.layer1.w.set_flat(i, orig + eps);
        Tensor y_plus = mlp.forward(x);
        float loss_plus = mlp.compute_loss(y_plus, target);

        // f(theta - eps)
        mlp.layer1.w.set_flat(i, orig - eps);
        Tensor y_minus = mlp.forward(x);
        float loss_minus = mlp.compute_loss(y_minus, target);

        // Restore
        mlp.layer1.w.set_flat(i, orig);

        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        float ana_grad = mlp.layer1.grad_w.at_flat(i);

        float diff = std::fabs(num_grad - ana_grad);
        float denom = std::max(1e-4f, std::fabs(num_grad) + std::fabs(ana_grad));
        float rel_err = diff / denom;

        assert(rel_err < 1e-2f); // Within 1% tolerance for finite differences
    }

    // 3. Numerical gradient check for layer2.w
    for (int64_t i = 0; i < mlp.layer2.w.numel(); ++i) {
        float orig = mlp.layer2.w.at_flat(i);

        mlp.layer2.w.set_flat(i, orig + eps);
        Tensor y_plus = mlp.forward(x);
        float loss_plus = mlp.compute_loss(y_plus, target);

        mlp.layer2.w.set_flat(i, orig - eps);
        Tensor y_minus = mlp.forward(x);
        float loss_minus = mlp.compute_loss(y_minus, target);

        mlp.layer2.w.set_flat(i, orig);

        float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
        float ana_grad = mlp.layer2.grad_w.at_flat(i);

        float diff = std::fabs(num_grad - ana_grad);
        float denom = std::max(1e-4f, std::fabs(num_grad) + std::fabs(ana_grad));
        float rel_err = diff / denom;

        assert(rel_err < 1e-2f);
    }

    std::cout << "[PASS] test_finite_difference_gradient_check (Analytical VJP matches numerical gradient within tolerance)\n";
}

void test_mlp_training_convergence() {
    MLPNetwork mlp(2, 8, 1);

    // XOR / non-linear classification dataset
    Tensor x({4, 2}, {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    });
    Tensor target({4, 1}, {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    });

    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    float lr = 0.1f;

    for (int epoch = 0; epoch < 100; ++epoch) {
        Tensor y_hat = mlp.forward(x);
        float loss = mlp.compute_loss(y_hat, target);
        if (epoch == 0) {
            initial_loss = loss;
        }
        if (epoch == 99) {
            final_loss = loss;
        }

        mlp.backward(target);
        mlp.step(lr);
    }

    // Verify significant loss reduction during training
    assert(final_loss < initial_loss);
    std::cout << "[PASS] test_mlp_training_convergence (Initial loss: " << initial_loss
              << " -> Final loss: " << final_loss << " after 100 epochs)\n";
}

int main() {
    test_finite_difference_gradient_check();
    test_mlp_training_convergence();

    std::cout << "\nAll Autodiff & MLP tests passed successfully!\n";
    return 0;
}
