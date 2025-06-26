- Deep RL is a type of ML where an agent learns how to behave by performing actions and seeing the results 
- Main idea is that an AI agent will learn from the environment by interacting with it (through trial and error) and receiving rewards (negative or positive) as feedback

#### The RL Framework 
![Framework](img1.png)
- Agent receives S<sub>o</sub> from the environment (ex, receives first frame of our game)
- Based on S<sub>o</sub>, agent takes action A<sub>o</sub> (ex, agent moves to the right)
- Environment goes to new state S<sub>1</sub> (ex, new frame)
- Environment gives reward R<sub>1</sub> to agent (ex, we're not dead = positive reward +1)

- RL loop outputs a sequence: state -> action -> reward -> next state 
	- Agents goal is to maximize the cumulative reward (ie, the expected return)

**The Reward Hypothesis: Central Idea of RL**
- RL is based on the reward hypothesis 
	- Goal is described as maximization of the expected return 
	- So, to have the best behaviour you must take actions which maximize the expected cumulative reward

**Markov Property**
- Markov Decision Process (MDP)
- In short, Markov Property implies that our agent needs only the current state to decide what action to take and **not** the history of all states and actions prior 

**Observations/States Space**
- Observations/states - the information our agent gets from the environment (ex, frame of a game, value of a certain stock)
- Differentiation between *observation* and *state*: 
	- *State s* -- a **complete description of the state** of the world (no hidden info) in a fully observed environment 
		- ex, in chess game - we have access to the entire board so we receive a state from the environment (ie, fully observed)
	- *Observation o* -- a **partial description of the state**, in a partially observed environment 
		- ex, in Super Mario Bros - only see part of the level close to the player so we receive an observation 

**Action Space**
- Set of all possible actions in an environment - which can come from a *discrete* or *continuous space*
	- Discrete space - the number of possible actions is **finite**
		- ex, super mario bros - have finite set of actions since there's only 4 directions
	- Continuous space - number of possible actions is **infinite**
		- ex, self driving car agent - has infinite number of possible actions since it can turn 20°, 21,1°, 21,2°, honk, turn right 20°, etc. 

**Rewards and the Discounting**
- Reward is fundamental in RL since its the only feedback for the agent 
	- Cumulative reward at each time step *t* can be written as: `R(τ) = rₜ₊₁ + rₜ₊₂ + rₜ₊₃ + rₜ₊₄ + …`
	which is equivalent to: `R(τ) = Σₖ₌₀^∞ rₜ₊ₖ₊₁`
	- But we can't just add them like that 
	- The rewards that come sooner (at beginning of the game) are more likely to happen since they're more predictable than the long-term future reward


#### Type of Tasks
- Task: an instance of an RL problem. There are two types = episodic and continuing 

**Episodic Task**
- Have a starting point and ending point (terminal state)
- Ultimately creates episodes = list of States, Actions, Rewards and new States 
- ex, super mario bros - an episode begins at the launch of a new level and ends when you're killing/reached the end of the level 

**Continuing Tasks**
- Tasks which continue forever (ie, no terminal state)
- The agent must learn how to choose the best actions and simultaneously interact with the environment 

#### Exploration/Exploitation Trade-Off
- Exploration: exploring the environment by trying random actions in order to find more info about the environment 
- Exploitation: exploiting known information to maximize the reward 
- Note: goal of RL agent is to maximize the expected cumulative reward

**Example**
![Mouse image](img4.png)
- Mouse can have an infinite amount of small cheese (+1 each) but at the top at the maze there's a huge sum of cheese (+1000)
	- Exploitation - our agent will never reach the huge sum of cheese but will rather exploit the nearest source of rewards (even if they're smaller)
	- Exploration - if agent does some more exploration, it can discover the bigger reward 
- So, this is the trade off - balancing how much we explore the env and how much we exploit what we already know 
#### 2 Main Approaches for RL Problems
***How do we build an RL agent that can select the actions that maximize its expected cumulative reward?***

**The Policy π: the agent’s brain**
- The Policy π is the agent's brain 
- Its the function which tells us what action to take given the state we're in - ie, defines agent's behaviour at a given time 
- Policy is the function we want to learn and our goal is to find the optimal policy π* - ie, the one which maximizes expected return 
	- We find π* through training 
- 2 approaches to train agent to find optimal policy π*:
	1. Directly: teach the agent to learn which action to take given the current state (**Policy-Based Methods**)
	2. Indirectly: teach the agent to learn which state is more valuable **(Value-Based Methods)** 

**Policy-Based Methods**
- Learn a policy function directly 
- A few possible methods
	- Function can define a mapping from each state to the best corresponding action
	- Could define a probability distribution over the set of possible actions at that state 

- 2 types of policies:
	1. Deterministic: a policy at a given state will always return the same action `a = π(s)`
		- action = policy(state)
		- State S<sub>o</sub> -> π( S<sub>o</sub>) ->  A<sub>o</sub> (action is moving to the right)
	2. Stochastic: outputs a probability distribution over actions 
	`π(a | s) = P[A | s]`
	- Policy(actions | state): probability distribution over the set of actions given the state

**Value-Based Methods**
- Learn a value function that maps a state to the expected value of being at that state
	- Value of a state: expected discounted return the agent can get if it starts in that state and then acts according to the chosen policy 
	- `v_π(s) = E_π [ Rₜ₊₁ + γRₜ₊₂ + γ²Rₜ₊₃ + … | Sₜ = s ]`
- With value function - the policy will select the state with the biggest value to attain the goal

#### The *Deep* in Reinforcement Learning
- Deep RL = uses deep NN to solve RL problems 

#### Hands On: Lunar Lander Agent Notes
*Note: the code implementation is found in this folder*

**Understanding Gymnasium**
- A new version of the Gym library 
- The lib provides two things:
	- An interface that allows you to create RL environments 
	- Collection of environments (gym-control, atari, box2D)
- With gymnasium:
	1. Create our env using `gymnasium.make()`
	2. Reset the environment to initial state:` observation = env.reset()`
	3. At each step:
		1. Get an action using our model
		2. Using` env.step(action)` - we perform this action in the env and obtain:
			- `observation`: The new state (st+1)
			- `reward`: The reward we get after executing the action
			- `terminated`: Indicates if the episode terminated (agent reach the terminal state)
			- `truncated`: indicates a time limit or if an agent goes out of bounds of the environment for instance
			- `info`: A dictionary that provides additional information (depends on the environment)
		- If the episode is terminated - reset the environment to its initial state with `observation = env.reset()`

**LunarLander Environment**
*Goal: train our agent to land correctly on the moon. The agent will learn to adapt its speed and position (horizontal, vertical and angular) to land correctly*

`Observation Space Shape (8,)`: 
- Observation is a vector of size 8 where each value contains different information:
	- Horizontal pad coordinate (x)
	- Vertical pad coordinate (y)
	- Horizontal speed (x)
	- Vertical speed (y)
	- Angle
	- Angular speed
	- If the left leg contact point has touched the land (boolean)
	- If the right leg contact point has touched the land (boolean)

`Action Space`
- *Set of all possible actions the agent can take* 
- In this case, we have 4 actions available:
	- Action 0: Do nothing
	- Action 1: Fire left orientation engine
	- Action 2: Fire the main engine
	- Action 3: Fire right orientation engine

`Reward Function`
- *The function that'll give a reward at each timestep*
- After every step, a reward is granted - the total reward of an episode is **sum of the rewards for all steps within that episode**
- For each step, the reward:
	- Is increased/decreased the closer/further the lander is to the landing pad
	- Is increased/decreased the slower/faster the lander is moving
	- Is decreased the more the lander is tilted (angle not horizontal)
	- Is increased by 10 points for each leg that is in contact with the ground
	- Is decreased by 0.03 points each frame a side engine is firing
	- Is decreased by 0.3 points each frame the main engine is firing
	- Episode will receive -100 or +100 for crashing or landing safely respectively 
	- Episode is considered a solution if it scores at least 200 points 

`Vectorized Environment`
- *A method for stacking multiple independent environments into a single one of 16*

**Creating the Model**
- Using Stable Baselines3 (SB3) - reliable implementations of RL algorithms in PyTorch
- Specifically, PPO (Proximal Policy Optimization) - one of the SOTA deep RL models and is a combination of value-based and policy-based RL learning methods 
- SB3 is easy to set up
	1. Create environment 
	2. Define and instantiate the model you want to use: `model = PPO(MlpPolicy)`
	3. Train the agent with `model.learn` and define number of training timesteps