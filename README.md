# Udacity-DRLND-ContinuousControl
Udacity Deep Reinforcement Learning Nanodegree - Continuous Control Project

## 1. Explanations about the Environment
In this environment, the agent's goal is manipulating the arm to reach the moving green ball.
Details of the environment is below:
 - Observation space: 33
 - Action space: 4(Continuous, Action range is -1 to 1)
 - Reward: If agent reaches the goal location, rewards +0.1 per step.
 
## 2. Installation
### 1. Install Dependencies
 #### If you're using `conda`:
  > ```
  > conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
  > conda install numpy scipy matplotlib 
  > ```
  
 #### Or using `pip`:
 > ```
 > # Linux or MacOS only:
 > pip install torch
 >
 > # Windows only:
 > pip install https://download.pytorch.org/whl/cu90/torch-1.0.1-cp36-cp36m-win-amd64.wml
 >
 > pip install torchvision
 > pip install numpy scipy matplotlib
 > ```
 ### 3. Install Unity ML-Agents
 You should install ml-agents **version 0.4**.<br>
 (Because of the API version of the environment, latest version is not compatible)
 ```
 pip install mlagents==0.4
 ```
 For more information about Unity ML-Agents, please visit the
 <a href="https://github.com/Unity-Technologies/ml-agents/blob/master/docs/">official documentation</a>.
### 2. Install the Environment
##### First, you have to download the environment from link below:
> For Linux: <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip">Download</a><br>
> For Mac OS X: <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip">Download</a><br>
> For Windows(32bit): <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip">Download</a><br>
> For Windows(64bit): <a href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip">Download</a>

##### Second, extract the downloaded environment.<br>
##### Finally, you **should change the `env_path` to environment excutable's path** in `config.py` file.<br>
 > e.g. If your environment executable's path is `./path/to/executable/Reacher.exe'`,<br>
 > you can change the `config.py`'s `env_path` to<br>
 > `config['env_path'] = 'path/to/executable/Reacher.exe'`

## 3. Train the Agent
Use `train.py` to train the agent.
```
python train.py
```
You can modify some hyperparameters before training by editing `config.py`.<br>
When the agent achives the goal score or when you hit `Ctrl + C`(KeyboardInturrupt), the training process will be stopped.<br>
After training process stops, the trained weight data and the reward plot will be saved in the `./save` directory.<br>
(You can change this path by editing `config.py`)

## 4. Result
