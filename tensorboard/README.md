## Monitoring model's parameters using Tensorboard

In this folder the `main.py` code is related to training a deep neural network and monitoring parameters of network while training. In other words, network parameters like weights/biases/activations for each layer and loss/accuracy values are stored and prepared to use with Tensorboard tool for monitoring.

### How to use:

* First run `python main.py` to trian the network and store logs signals of graph model.
* Second while training the model or after training run `$ tensorboard --logdir=logs/`.
* launch your browser and go to tensorboard address (for example http://pcname:6006).
