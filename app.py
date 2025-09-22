import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

st.set_page_config(page_title="Neural Network Backpropagation Demo", layout="wide")
st.title("🔄 Backpropagation Algorithm Visualizer")

# Sidebar controls
st.sidebar.header("Network Settings")
n_inputs = st.sidebar.slider("Number of input neurons", 2, 5, 3)
n_hidden = st.sidebar.slider("Number of hidden neurons", 2, 6, 4)
n_outputs = st.sidebar.slider("Number of output neurons", 1, 3, 1)
learning_rate = st.sidebar.slider("Learning rate", 0.01, 1.0, 0.1, 0.01)
epochs = st.sidebar.slider("Epochs", 1, 50, 10)

# Example dataset (generalized)
if n_inputs == 2 and n_outputs == 1:
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
else:
    X = np.random.randint(0, 2, size=(4, n_inputs))
    y = np.random.randint(0, 2, size=(4, n_outputs))

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(n_inputs, n_hidden)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_outputs)
b2 = np.zeros((1, n_outputs))

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_derivative(x): return sigmoid(x)*(1-sigmoid(x))

# Loss function (Mean Squared Error)
def compute_loss(y, a2):
    return np.mean(0.5 * (y - a2) ** 2)

# Graph drawing function
def draw_network(W1, W2, A1=None, A2=None, deltas=None, step="Forward Pass", epoch=None, loss=None):
    G = nx.DiGraph()
    pos = {}
    labels = {}

    # Input layer
    for i in range(n_inputs):
        node = f"I{i+1}"
        G.add_node(node)
        pos[node] = (0, -i)
        labels[node] = node

    # Hidden layer
    for j in range(n_hidden):
        node = f"H{j+1}"
        G.add_node(node)
        pos[node] = (1, -j)
        labels[node] = node

    # Output layer
    for k in range(n_outputs):
        node = f"O{k+1}"
        G.add_node(node)
        pos[node] = (2, -k)
        labels[node] = node

    # Add edges
    for i in range(n_inputs):
        for j in range(n_hidden):
            G.add_edge(f"I{i+1}", f"H{j+1}")
    for j in range(n_hidden):
        for k in range(n_outputs):
            G.add_edge(f"H{j+1}", f"O{k+1}")

    plt.figure(figsize=(8,5))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color="lightblue", node_size=2000, arrowsize=20)
    
    if A1 is not None:
        for idx, val in enumerate(A1[0]):
            plt.text(1, -idx, f"{val:.2f}", fontsize=10, color="red")
    if A2 is not None:
        for idx, val in enumerate(A2[0]):
            plt.text(2, -idx, f"{val:.2f}", fontsize=10, color="green")
    if deltas is not None:
        for idx, val in enumerate(deltas[0]):
            plt.text(2.2, -idx, f"δ={val:.2f}", fontsize=10, color="orange")
    
    # Display weights, biases and loss
    if epoch is not None:
        plt.text(2.5, -1.5, f"Epoch {epoch}", fontsize=12, color="black")
        plt.text(2.5, -2, f"Updated W1: {W1}", fontsize=8, color="black")
        plt.text(2.5, -2.5, f"Updated b1: {b1}", fontsize=8, color="black")
        plt.text(2.5, -3, f"Updated W2: {W2}", fontsize=8, color="black")
        plt.text(2.5, -3.5, f"Updated b2: {b2}", fontsize=8, color="black")
        if loss is not None:
            plt.text(2.5, -4, f"Loss: {loss:.4f}", fontsize=10, color="blue")

    plt.title(step)
    st.pyplot(plt.gcf())
    plt.close()

# Animation
placeholder = st.empty()

for epoch in range(epochs):
    epoch_loss = 0
    for i in range(len(X)):
        # Forward pass
        z1 = np.dot(X[i:i+1], W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)

        # Compute loss for the epoch
        loss = compute_loss(y[i:i+1], a2)
        epoch_loss += loss

        with placeholder.container():
            draw_network(W1, W2, A1=a1, A2=a2, step=f"Epoch {epoch+1} - Forward Pass", epoch=epoch+1, loss=loss)
        time.sleep(0.8)

        # Backpropagation
        error = y[i:i+1] - a2
        delta2 = error * sigmoid_derivative(z2)
        delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(z1)

        with placeholder.container():
            draw_network(W1, W2, A1=a1, A2=a2, deltas=delta2, step=f"Epoch {epoch+1} - Backpropagation", epoch=epoch+1, loss=loss)
        time.sleep(0.8)

        # Update weights
        W2 += learning_rate * np.dot(a1.T, delta2)
        b2 += learning_rate * delta2
        W1 += learning_rate * np.dot(X[i:i+1].T, delta1)
        b1 += learning_rate * delta1

    # Average loss for the epoch
    epoch_loss /= len(X)
    st.text(f"Epoch {epoch+1} - Average Loss: {epoch_loss:.4f}")

st.success("🎉 Training complete! Final weights and biases updated.")
