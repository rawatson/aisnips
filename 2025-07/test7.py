#2025-07-04
'''
Using python, generate a simple neural network that does recognition on the MNIST database.  Use only basic libraries like numpy -- we want to show our work and illustrate how this works.  Keep the code concise.

The model should mirror the structure from the 3blue1brown video on neural networks, transcript copied below for reference:

(copied transcript from chapter 1 video on ML)
'''
#!/usr/bin/env python3
# mnist_numpy_net.py
import numpy as np, urllib.request, pathlib, time

# ------- Data loader (downloads once, ~17 MB) -------------------------------
def load_mnist():
    url  = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
    path = pathlib.Path("mnist.npz")
    if not path.exists():
        print("Downloading MNIST …")
        urllib.request.urlretrieve(url, path)

    with np.load(path, allow_pickle=False) as f:
        x_train, y_train = f["x_train"], f["y_train"]
        x_test,  y_test  = f["x_test"],  f["y_test"]

    # Flatten & scale to [0, 1]
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test  = x_test.reshape(-1, 784) / 255.0
    # One-hot labels
    y_train = np.eye(10)[y_train]
    y_test  = np.eye(10)[y_test]
    return x_train, y_train, x_test, y_test

# ------- Tiny NN framework --------------------------------------------------
def sigmoid(z):       return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z): s = sigmoid(z); return s * (1 - s)

class NeuralNet:
    def __init__(self, sizes, lr=0.1):
        self.lr, self.sizes = lr, sizes
        # Heuristic init ~ N(0, √(1/n_in))
        self.W = [np.random.randn(o, i) * np.sqrt(1/i)
                  for i, o in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros((o, 1)) for o in sizes[1:]]

    def forward(self, a0):
        """Return (activations, z-vectors) for all layers."""
        activ, zs = [a0], []
        a = a0
        for W, b in zip(self.W, self.b):
            z = W @ a + b
            a = sigmoid(z)
            zs.append(z); activ.append(a)
        return activ, zs

    def update_batch(self, x, y):
        """One SGD step on a mini-batch (x,y) with shape (m, …)."""
        a0 = x.T                     # (784, m)
        activ, zs = self.forward(a0)
        delta = activ[-1] - y.T      # cross-entropy derivative
        nabla_W = [None]*len(self.W)
        nabla_b = [None]*len(self.b)
        # Gradient for output layer
        nabla_W[-1] = delta @ activ[-2].T
        nabla_b[-1] = delta.sum(axis=1, keepdims=True)
        # Back-prop through hidden layers
        for ℓ in range(2, len(self.sizes)):
            z    = zs[-ℓ]
            sp   = sigmoid_prime(z)
            delta = (self.W[-ℓ+1].T @ delta) * sp
            nabla_W[-ℓ] = delta @ activ[-ℓ-1].T
            nabla_b[-ℓ] = delta.sum(axis=1, keepdims=True)
        # Gradient-descent step
        m = x.shape[0]
        self.W = [W - (self.lr/m)*dW for W, dW in zip(self.W, nabla_W)]
        self.b = [b - (self.lr/m)*db for b, db in zip(self.b, nabla_b)]

    def accuracy(self, x, y):
        preds = np.argmax(self.predict(x), axis=1)
        return (preds == np.argmax(y, axis=1)).mean()

    def predict(self, x):
        return self.forward(x.T)[0][-1].T  # shape (n_samples, 10)

    def fit(self, x_train, y_train, epochs=10, batch=64, x_val=None, y_val=None):
        n = x_train.shape[0]
        for e in range(1, epochs+1):
            perm = np.random.permutation(n)
            for k in range(0, n, batch):
                idx = perm[k:k+batch]
                self.update_batch(x_train[idx], y_train[idx])
            msg = f"Epoch {e:02d}: train {self.accuracy(x_train, y_train)*100:4.1f}%"
            if x_val is not None:
                msg += f"  |  test {self.accuracy(x_val, y_val)*100:4.1f}%"
            print(msg)

# ------- Train --------------------------------------------------------------
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()
    net = NeuralNet([784, 16, 16, 10], lr=0.5)
    t0 = time.time()
    net.fit(x_train, y_train, epochs=10, batch=64, x_val=x_test, y_val=y_test)
    print(f"Finished in {time.time()-t0:.1f}s")
