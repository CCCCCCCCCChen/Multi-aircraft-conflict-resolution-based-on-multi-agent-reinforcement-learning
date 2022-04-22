# Conflict resolution environment

This is the conflict resolution environment for the Challenge #2 of the EUROCONTROL Innovation Masterclass

## Statement of work

Artificial intelligence has been declared successful in providing decision support in a variety of real-world applications. Many of these accomplishments have been made possible by recent advancements in reinforcement learning (RL) algorithms. In short, RL algorithms can be used to discover the best strategy (policy in the machine learning jargon) for a wide range of difficult tasks simply by learning from the experiences of the agent interacting with the environment. The policy is typically a neural network that takes as input the state of the environment as observed by the agent and determines the best action to maximize the return (cumulative discounted reward).

Recently, EUROCONTROL has implemented a reinforcement learning system for training Air Traffic Control (ATC) policies. The current system is composed of (1) a relatively simple ATC simulator that generates experiences, and (2) a learner based on the Proximal Policy Optimisation (PPO) algorithm that uses these experiences to continuously improve the policy. Initial results showed that the optimal policy that minimises the losses of separation and the environmental impact could be learned from scratch with RL. These promising findings encouraged us to take you on-board of this challenge!

At present, however, the ATC simulator does not consider the vertical dimension (i.e., all aircraft are assumed to be at the same altitude), and consequently the policy can only learn speed and/or heading resolution actions. Furthermore, the simulator does not include uncertainty, meaning that the policy may not perform well in real-life situations, where uncertainty is inevitable. Last but not least, the PPO algorithm was not explicitly designed for multi-agent environments, and therefore other algorithms like Actor-Attention-Critic for Multi-Agent Reinforcement Learning (MAAC) or Deep Coordination Graphs (DCG) may achieve better performance.

We will provide you with the skeleton of a basic 2D ATC simulator (the environment) built on the Gym framework in this challenge (in Python). You will tailor this simulator by adding:

* The observation function (what do agents observe from the environment to take actions?)
* The reward function (how are agents reward or penalised by their actions?)
* The action space (e.g., heading change, speed change), which can be discrete or continuous 

[source code of the Environment](https://github.com/ramondalmau/atcenv/blob/main/atcenv/env.py)

and then you will train the optimal policy using a reinforcement learning algorithm of your choice. 

We also encourage you to explore any of the following bonus tasks:
* Implement the vertical dimension in the simulation environment
* Implement weather in the simulator (e.g, consider the effect of wind)
* Implement uncertainty in the simulation environment. For instance, due to measurement errors, the position observed by the agents may not perfectly correspond to the actual one, or agents may not react instantaneously to resolution actions. 

The jury will consider the following factors when evaluating the solutions proposed by the various teams:
* The performance of the policy (e.g., number of conflicts, extra distance / environmental impact, number of resolution actions)
* The learnt policy's realism and scalability to any number of agents/flights
* The originality and appropriateness of the approach
* The clarity of the presentation
* Bonus task will be positively considered as well

## References
---
[Dalmau, R. and Allard, E. "Air Traffic Control Using Message Passing Neural Networks and Multi-Agent Reinforcement Learning", 2020. 10th SESAR Innovation Days](https://www.researchgate.net/publication/352537798_Air_Traffic_Control_Using_Message_Passing_Neural_Networks_and_Multi-Agent_Reinforcement_Learning)
---

## Download

```bash
git clone https://github.com/ramondalmau/atcenv.git
```

## Installation 

The environment has been tested with Python 3.8 and the versions specified in the requirements.txt file

```bash
cd atcenv
pip install -r requirements.txt
python setup.py install
```

## Usage

```python
from atcenv import Environment

# create environment
env = Environment()

# reset the environment
obs = env.reset()

# set done status to false
done = False

# execute one episode
while not done:
    # compute the best action with your reinforcement learning policy
    action = ...

    # perform step
    obs, rew, done, info = env.step(action)
    
    # render (only recommended in debug mode)
    env.render()

env.close()
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
