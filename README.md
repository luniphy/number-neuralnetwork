[![CI](https://github.com/luniphy/number-neuralnetwork/actions/workflows/ci.yml/badge.svg)](https://github.com/luniphy/number-neuralnetwork/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](https://www.python.org/)
[![Qt](https://img.shields.io/badge/PyQt-2CDE85?logo=Qt&logoColor=fff)](https://www.qt.io/)
[![Docker](https://img.shields.io/badge/Docker-%230db7ed.svg?&logo=docker&logoColor=white)](https://hub.docker.com/r/luniphys/number-neuralnetwork)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


# Number Neural Network

A neural network that can recognize handwritten digits. It is built from scratch without any machine learning frameworks (no TensorFlow, PyTorch, etc.) and trained using the [<b>MNIST dataset</b>](https://en.wikipedia.org/wiki/MNIST_database). A PyQt6 GUI is included for interactive drawing and training.

<p align="center">
    <img src="docs/images/network_image.png" width="900" alt="Network diagram">
</p>


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Results](#results)
- [GUI](#gui)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Testing](#testing)
- [Docker](#docker)
- [Mathematics](#mathematics)
- [Acknowledgments](#acknowledgments)
- [License](#license)


## Overview

This educational project demonstrates a neural network that has the following architecture:

- Input: <b>MNIST</b> grayscale images (28 x 28 pixels, represented by 784 input neurons)
- Neurons: 784 -> 16 -> 16 -> 10
- Activation function: Sigmoid
- Cost function: Squared error
- Training: Self-implemented gradient-based backpropagation


## Features

- Implementation in plain Python and NumPy
- Automatic <b>MNIST</b> data download if required (training + test data)
- Training, evaluation and testing (CI) scripts
- Interactive PyQt6 GUI with possibilities to:
    - Draw digits on a 28 x 28 pixel canvas
    - Evaluate them (probability distribution)
    - Use a pretrained model or train a fresh model yourself
    - Self-trainable model stored locally in `data/`
- Buildable/ downloadable Docker image included


## Results

The included pre-trained model reaches approximately 94.84% accuracy after 281 training cycles (about 60 hours total training time).

<p align="center">
    <img src="docs/images/cost_plot_trained.svg" width="550" alt="Training cost curve">
</p>


## GUI

<p align="center">
    <img src="docs/images/gui_example.png" width="800" alt="GUI">
</p>


## Project Structure

```
number-neuralnetwork/
├─ .github/workflows/ci.yml     # CI workflow
├─ data/
│  ├─ MNIST/                    # MNIST dataset
│  └─ models/                   # Pre- and self-trained model
├─ docs/images                  # Documentation images
├─ src/neuralnetwork/
│  ├─ __init__.py
│  ├─ paths.py                  # Centralized path definitions
│  ├─ training.py               # Network and data setup + training
│  ├─ evaluation.py             # Accuracy and cost evaluation
│  ├─ gui.py                    # Interactive PyQt6 GUI: Draw, train
│  └─ assets/                   # GUI images and data, icons, styling
├─ tests/
│  └─ test_training.py          # Automated tests
├─ DOCKERFILE                   # Buildable Docker image
├─ LICENSE
├─ README.md
├─ requirements.txt             # Program dependencies
└─ setup.py
```


## Usage

Run all following commands from the repository root.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch GUI

```bash
python src/neuralnetwork/gui.py
```

### Train a Model

```bash
python src/neuralnetwork/training.py
```

`training.py` also acts as the backend engine of GUI.

### Evaluate a Trained Model

```bash
python src/neuralnetwork/evaluation.py
```

The evaluation script shows average cost, total misclassifications, accuracy and a random samples' probability distribution.

Notes:

- Training a fresh network is computationally expensive and can take many hours depending on hardware.
- In `gui.py` the self-trainable model is trained by <b>MNIST</b> test data, whereas in `training.py` by the training data, which is 6 times bigger. This makes GUI is significantly faster.


## Testing

The mathematical functions of the training algorithm are covered by automated tests in `test_training.py`. It is executed in the CI pipeline on every push.

To run the tests manually (from root):

```bash
pytest
```

Make sure `pytest` is installed via `pip install pytest`.


## Docker

For the GUI, a Dockerfile is included to provide a reproducible runtime environment with all required dependencies and relevant data.

### Build the image

From the repository root, build the Docker image by:

```bash
docker build -t number-neuralnetwork .
```

### Pull from Docker Hub

A prebuilt image is also available on [Docker Hub](https://hub.docker.com/r/luniphys/number-neuralnetwork):

```bash
docker pull luniphys/number-neuralnetwork
```

### Run the container

The container needs access to the host display. To do so run:

```bash
xhost +local:docker
```

To run the container:

```bash
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  number-neuralnetwork
```

After run, revoke display access:

```bash
xhost -local:docker
```

### Notes

- The commands were tested and executed on **Linux Mint**. Other distributions may require different display configurations. The above should work for most **Ubuntu**-based distributions though.


## Mathematics

The network computes each layer activation by:

$$
a^{(n)} = \sigma \left( W^{(n)} a^{(n-1)} + b^{(n)} \right), \quad n = 1,2,3
$$

$\sigma$ represents the sigmoid activation function:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

The objective is to minimize the squared error cost function:

$$
C = \sum_{k=1}^{n_3} \left(a_k^{(3)} - y_k\right)^2
$$

where $y$ is the encoded target vector for the true digit.

To minimize $C$, the network uses the following gradient:

$$
\frac{\partial C}{\partial w_{ij}^{(3)}} = 2 \left(a_i^{(3)} - y_i \right) \cdot \sigma^{\prime} \left(z_i^{(3)} \right) \cdot a_j^{(2)}
$$

$$
\frac{\partial C}{\partial b_{i}^{(3)}} = 2 \left(a_i^{(3)} - y_i \right) \cdot \sigma^{\prime} \left(z_i^{(3)} \right)
$$

$$
\frac{\partial C}{\partial w_{ij}^{(2)}} = \sigma^{\prime} \left(z_i^{(2)} \right) \cdot a_j^{(1)} \cdot \sum_{k=1}^{n_3} 2 \left(a_k^{(3)} - y_k \right) \cdot \sigma^{\prime} \left(z_k^{(3)} \right) \cdot w_{ki}^{(3)}
$$

$$
\frac{\partial C}{\partial b_{i}^{(2)}} = \sigma^{\prime} \left(z_i^{(2)} \right) \cdot \sum_{k=1}^{n_3} 2 \left(a_k^{(3)} - y_k \right) \cdot \sigma^{\prime} \left(z_k^{(3)} \right) \cdot w_{ki}^{(3)}
$$

$$
\frac{\partial C}{\partial w_{ij}^{(1)}} = \sigma^{\prime} \left(z_i^{(1)} \right) \cdot a_j^{(\text{in})} \cdot \sum_{k=1}^{n_3} 2 \left(a_k^{(3)} - y_k \right) \cdot \sigma^{\prime} \left(z_k^{(3)} \right) \cdot \sum_{l=1}^{n_2} w_{kl}^{(3)} \cdot \sigma^{\prime} \left(z_l^{(2)} \right) \cdot w_{li}^{(2)}
$$

$$
\frac{\partial C}{\partial b_{i}^{(1)}} = \sigma^{\prime} \left(z_i^{(1)} \right) \cdot \sum_{k=1}^{n_3} 2 \left(a_k^{(3)} - y_k \right) \cdot \sigma^{\prime} \left(z_k^{(3)} \right) \cdot \sum_{l=1}^{n_2} w_{kl}^{(3)} \cdot \sigma^{\prime} \left(z_l^{(2)} \right) \cdot w_{li}^{(2)}
$$


## Acknowledgments

The project approach and mathematical inspiration came from the neural network series by the fabulous <b>3Blue1Brown</b>.

https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi


## License

MIT © [luniphy](https://github.com/luniphy)
