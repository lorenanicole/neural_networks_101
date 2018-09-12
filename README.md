# neural_networks_101

Example of a Python single layer perceptron neural network using sklearn datasets

### Associated Workshop Slides

The accompanying slides for this repository can be found [here](https://goo.gl/eb3tF9).

### Setup 

This is a Python 3 project. You can use either the `single_layer_perceptron.py` file or spin up a Jupyter notebook.

First you'll need to create a `virtualenv` and install the requirements. These instructions presume a [Mac setup](https://gist.github.com/pandafulmanda/730a9355e088a9970b18275cb9eadef3). For windows setup follow these [instructions](http://timmyreilly.azurewebsites.net/setup-a-virtualenv-for-python-3-on-windows/). 

To start:

```
virtualenv -p python /path/to/python3 neural_networks_env
source path/to/neural_networks_env/bin/activate
(neural_networks_env) pip install -r requirements.txt
```

### Running the Python Code

Once the requirements have been installed and you've activated your `virtualenv`:

```
(neural_networks_env) python single_layer_perceptron.py
```

### Optional: Jupyter Notebook

```
(neural_networks_env) jupyter notebook
```
