I kicked this off to test GPT5.  It blew me away, generating this mostly intact with just one prompt.

Now, it did take me a few tries to figure out the right prompt, but once it's working -- man, it's good!

Prompt for reference:

Create a single page HTML / JS to visualize the usage of a neural net to parse data from MNIST.  Fundamentally, the model should use 2 hidden layers, each with 16 nodes, and classify each digit as 0-9.  The user should be able to tweak the model in various interesting ways, draw their own digit, and visualize the flow of data through the system.  Use MNIST data from https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png, which stores the data as a 784 column, 65000 row PNG, with each row representing a digit.

Do NOT use tensorflow.js or other advanced machine learning libraries.  Use plain javascript as much as possible.  You can use more basic external libraries for general math ops if needed, but ensure that the core ML logic is shown in the JS code