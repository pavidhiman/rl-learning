#### 2 Types of Value-Based Methods
*Value-based methods learn a value function which maps a state to the expected value of being in that state. The value of a state is the expected discounted return the agent can get.*

- Value-based rL focuses on learning *how good* each action is -- so, it returns a score for every state (or every state-action pair)
	- So, the optimal policy is derived from these states 
		- ex, if we want a policy which takes actions that'll always lead to the biggest reward - we'll create a Greedy Policy 
	- In the case of value-based method, you never train the policy - the policy is a **simple pre-specified function** which uses values (given by value-function) to select its actions 
- Main difference:
	- Policy-based training: optimal policy (π*) is found by training the policy directly
	- Value-based training: finding optimal value function (Q* or V*) leads to having an optimal policy 
	- Link between value and policy: `π*(s) = arg maxₐ Q*(s, a)`
		- `π∗(s)` = optimal policy at state *s* (ie, action agent should pick if it wants highest long-term reward)
		- `argmax_a`= argument *a* that maximizes the expression
		- `Q∗(s,a)​` = optimal Q-value of (state and action) 
		- In short, look at every possible action you could take in state *s* and check its Q-value (tells you how much total reward you can expect if you take that action) and thus, choose the action with the highest score

**State-value Function**
`V_π(s) = E_π [ G_t | S_t = s ]`
- `V_π(s)`: value of state s
- `E_π`: expected return 
- `S_t = s`: agent starts at state s 

- For each state: the state-value function outputs the expected return **if the agent starts in that state and then follows the policy forever after** 

**Action-value Function**
- Outputs the expected return if the agent starts in that states, **takes that action** and then follows the policy forever after 
- `Q_π(s, a) = E_π [ G_t | S_t = s, A_t = a ]`
- Main difference is:
	- State-value function: calculate value of state `S_t`
	- Action-value function: calculate value of state-action pair (`S_t, A_t`) - hence the value of taking that action at that state

**Bellman Eq'n**
- Regardless of which value function we choose (state-value or action-value function), the returned value = expected return
	- Problem is to calculate each value of state or state-action pair - must **sum** all the rewards an agent can get if it starts at that state (this is computationally expensive)
