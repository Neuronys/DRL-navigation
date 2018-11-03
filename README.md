# DRL-navigation
Navigation Project of the Udacity's Deep Reinforcement Learning Nano Degree

</br>

## Project details
This project uses an Unity 3D environment. This environment is a flat area with bananas. There are 2 types of bananas: the yellow bananas and the blue bananas. The objective is to collect yellow bananas and avoid blue bananas. 

A reward of +1 is given for collecting a yellow banana and -1 for collecting blue banana.

There are 4 actions available: 0 - move forward, 1 - move backward, 2 - move left, 3 - move right.

With these rules, the state space has 37 dimensions.

This AI gym environment is considered solved when an average reward of +13 for 100 consecutive episodes is reached.

</br>

## Getting started
The project is written using Jupyter Notebook.
The first command of the notebook needs to be run to install all the required packages:

```
!pip -q install ./python
```


</br>

## Repository Content
The project consists of 10 files:
* Navigation.ipynb - run this file in Jupyter Notebook
* agent.py - the Agent class
* model.py - the Deep Neural Networks models
* hyperparams.py - tthe hyper parameters used
* model_DQH2.pth - saved trained model to use (Deep Q Network with 2 hidden layers)
* model_DDQH2.pth - saved trained model to use (Double Deep Q Network with 2 hidden layers)
* model_DQH3.pth - saved trained model to use (Deep Q Network with 3 hidden layers)
* model_DDQH3.pth - saved trained model to use (Double Deep Q Network with 3 hidden layers)
* Report.pdf - description of the implementation
* and this README.md file

