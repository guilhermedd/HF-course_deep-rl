# Last update: 2025-07-17
[Continue](https://huggingface.co/learn/deep-rl-course/unit1/hands-on)

# Reinforcement Learning, the big picture:
- An agent will learn to act in an environment by trial and error, receiving rewards and penalties as feedback.
## Formal definition:
`Reinforcement learning is a framework for solving control tasks (also called decision problems) by building agents that learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.`

# The RL process:
- The **Agent** makes an action in the **Environment**.
- The **Environment** responds to the action and provides a **Reward**.
- The action generates a new **State** in the **Environment**.
- The **Agent** uses the **Reward** and the new **State** to learn and improve its future actions.

## Terminology:
- **S<sub>0</sub>**: State at time 0.
- **A<sub>0</sub>**: Action at time 0.
- **R<sub>0</sub>**: Reward at time 0.
- **S<sub>1</sub>**: State at time 1.
- **Markov Decision Process** (MDP): It is the RL process. The markov Property implies that our agent need only the current state to decide what action to take and not the history of all the states and actions they took before.
- **State**: Complete description of the state of the world (no hidden info).
- **Observation**: Partial description of the state.
- **Action Space**: Set of all possible actions in an environment.

# **Rewards and Discounts** 
The only feedback from the environment to the agent. 
How the agent knows if it is doing well or not. 
The cumulative reward is the sum of all rewards received by the agent.

In real life, we can't just all them. 
The rewards that come sooner are more likely to happen, since they are more predictable than the long-term future.

To discount the rewards, we proceed as follows:
1. We define a discount factor `γ` (gamma) between 0 and 1. Most of the time, it is set between 0.95 and 0.99.
    - The larger the gamma, the smaller the discount. This means that the agent will care more about the long-term rewards.
    - The smaller the gamma, the more the agent will care about the immediate rewards.
2. Each reward will be discounted by gamma to the exponent of the time step.
    - For example, if the reward is received at time `t`, it will be discounted by γ<sup>t</sup>.
    - R(T) = Sum(γ<sup>k</sup> * r<sub>t + k + 1</sub>) for all k in the time steps.

# **Tasks**
A task is a instance of a RL problem. We can have two types of tasks: episodic and continuing.

## Episodic task
We have a starting point and a ending point (terminal state).
This creates an episode: a list of States, Actions, Rewards and new States.

## Continuing task
We don't have a terminal state. In this case, the agent must learn how to choose the best actions and simultaneosly interact with the environment.

# **The Exploration/Exploitation trade-off**
- **Exploration**: Exploring the environment by trying random actions in order to find more about the environment.
- **Exploitation**: Using the knowledge already acquired to choose the best action (maximize the reward).

The trade-off avoids the agent to get stuck in a local optimum, where it only exploits the knowledge it has and doesn't explore new actions that could lead to better rewards.

We need to balance how much we explore the environment and how much we exploit what we already know.

# **Main approaches for solving RL problems**
We need to build an Agent that can select the actions that maximize its expected cumulative reward.

## The policy π (the agent's brain)
The policy is a function that tells us what actions to take given the state we are in. 
It defines the agent's behavior at a given time.

The policy is the function we want to learn.
There are two approaches to learn the optimal policy:
- Policy-Based Methods: Teaching the agent which action to take given the current state.
- Value-Based Methods: Teaching the agent to learn which state is more valuable and then take the action that leads to the more valuable states

## Policy-Based Methods
We learn the policy directly.

This function will define a mapping from each state to the best corresponding action.
Alternatively, we can learn a probability distribution over the actions for each state.

## Value-Based Methods
We learn a value function that maps a state to the expected value of being at that state.

The value of a state is the expected discounted return the agent can get it it starts at that state and follows the policy.
At each step, the agent will choose the action that leads to the state with the highest value.

# **The "Deep" in Reinforcement Learning**
The "Deep" in Deep Reinforcement Learning refers to the use of deep neural networks to approximate the policy or value function.# HF-course_deep-rl
# HF-course_deep-rl
# HF-course_deep-rl
