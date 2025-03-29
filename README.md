OneNeurone_DeepLearning
Logistic Regression Model with Gradient Descent for Binary Classification
Logistic Regression Model with Gradient Descent for Binary Classification**  

Description:
This Python script implements a logistic regression model from scratch using NumPy. It trains a binary classifier using gradient descent and visualizes the decision boundary. The model includes essential functions for training, prediction, and saving/loading parameters.

---

Key Components:

1. **Mathematical Functions:**
   - `sigmoid(z)`: Computes the sigmoid activation function.

2. **Model Initialization:**
   - `initialize_parameters(input_size)`: Initializes weights and bias with small random values.

3. **Forward Propagation:**
   - `forward_propagation(X, parameters)`: Computes the linear combination and applies the sigmoid function to generate predictions.

4. **Cost Computation:**
   - `compute_cost(A, Y)`: Computes binary cross-entropy loss to evaluate model performance.

5. **Backward Propagation:**
   - `backward_propagation(X, Y, A)`: Computes gradients of the cost function with respect to model parameters.

6. **Parameter Update:**
   - `update_parameters(parameters, grads, learning_rate)`: Updates weights and bias using gradient descent.

7. **Training the Model:**
   - `model(X, Y, num_iterations, learning_rate, print_cost)`: Trains the logistic regression model for a specified number of iterations.

8. **Making Predictions:**
   - `predict(X, parameters)`: Generates binary predictions based on learned parameters.

9. **Visualization:**
   - `plot_decision_boundary(X, Y, parameters)`: Plots the decision boundary in a 2D feature space.

10. **Saving and Loading Model Parameters:**
    - `save_parameters(parameters, filename)`: Saves trained model parameters using `pickle`.
    - `load_parameters(filename)`: Loads model parameters from a saved file.

11. **User Input Prediction:**
    - `predict_user_input(user_input, parameters, plot_data, X_train, Y_train)`: Predicts class labels for new data points and optionally visualizes results.

---

### **Usage Workflow:**
1. **Generate Data (Optional)**
   ```python
   from sklearn.datasets import make_blobs
   X, Y = make_blobs(n_samples=200, centers=2, random_state=42)
   Y = Y.reshape(-1, 1)  # Reshape for compatibility
   ```

2. **Train Model**
   ```python
   parameters, costs = model(X, Y, num_iterations=1000, learning_rate=0.1, print_cost=True)
   ```

3. **Make Predictions**
   ```python
   predictions, _ = predict(X, parameters)
   ```

4. **Visualize Decision Boundary**
   ```python
   plot_decision_boundary(X, Y, parameters)
   ```

5. **Save & Load Parameters**
   ```python
   save_parameters(parameters, "logistic_model.pkl")
   loaded_parameters = load_parameters("logistic_model.pkl")
   ```

6. **Predict New Data**
   ```python
   new_data = np.array([[2, 1], [-1, 3]])
   predict_user_input(new_data, parameters)
   ```

### **Applications:**
- Binary classification tasks (e.g., spam detection, medical diagnosis, fraud detection).
- Educational purposes for understanding logistic regression and gradient descent.
- Small-scale machine learning projects.

This implementation provides a complete logistic regression model with a structured workflow for training, prediction, and visualization. 🚀
