- Deep RL is a type of ML where an agent learns how to behave by performing actions and seeing the results 
- Main idea is that an AI agent will learn from the environment by interacting with it (through trial and error) and receiving rewards (negative or positive) as feedback

#### The RL Framework 
![[Pasted image 20250616233820.png]]
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