[https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/](https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/)

[https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/)

[https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)

[https://github.com/aikorea/awesome-rl](https://github.com/aikorea/awesome-rl)

[https://medium.com/deep-math-machine-learning-ai/ch-12-1-model-free-reinforcement-learning-algorithms-monte-carlo-sarsa-q-learning-65267cb8d1b4](https://medium.com/deep-math-machine-learning-ai/ch-12-1-model-free-reinforcement-learning-algorithms-monte-carlo-sarsa-q-learning-65267cb8d1b4)

Beyond the agent and the environment, one can identify four main subelements of a reinforcement learning system: a **policy**, a **reward signal**, a **value function**, and, optionally, a **model of the environment**.

A **policy** defines the learning agent’s way of behaving at a given time. Roughly speaking, a policy is a mapping from perceived states of the environment to actions to be taken when in those states. It corresponds to what in psychology would be called a set of stimulus–response rules or associations. In some cases the policy may be a simple function or lookup table, whereas in others it may involve extensive computation such as a search process.

A **reward signal** defines the goal of a reinforcement learning problem. On each time step, the environment sends to the reinforcement learning agent a single number called the reward. The agent’s sole objective is to maximize the total reward it receives over the long run. The reward signal thus defines what are the good and bad events for the agent. In a biological system, we might think of rewards as analogous to the experiences of pleasure or pain. They are the immediate and defining features of the problem faced by the agent. The reward signal is the primary basis for altering the policy; if an action selected by the policy is followed by low reward, then the policy may be changed to select some other action in that situation in the future. In general, reward signals may be stochastic functions of the state of the environment and the actions taken.

Whereas the reward signal indicates what is good in an immediate sense, a **value function** specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Whereas rewards determine the immediate, intrinsic desirability of environmental states, values indicate the long-term desirability of states after taking into account the states that are likely to follow and the rewards available in those states. For example, a state might always yield a low immediate reward but still have a high value because it is regularly followed by other states that yield high rewards. Or the reverse could be true. To make a human analogy, rewards are somewhat like pleasure \(if high\) and pain \(if low\), whereas values correspond to a more refined and farsighted judgment of how pleased or displeased we are that our environment is in a particular state.

Rewards are in a sense primary, whereas values, as predictions of rewards, are secondary. Without rewards there could be no values, and the only purpose of estimating values is to achieve more reward. Nevertheless, it is values with which we are most concerned when making and evaluating decisions. **Action choices** are made based on value judgments. We seek actions that bring about states of highest value, not highest reward, because these actions obtain the greatest amount of reward for us over the long run. Unfortunately, it is much harder to determine values than it is to determine rewards. Rewards are basically given directly by the environment, but values must be estimated and re-estimated from the sequences of observations an agent makes over its entire lifetime. In fact, the most important component of almost all reinforcement learning algorithms we consider is a method for efficiently estimating values. The central role of value estimation is arguably the most important thing that has been learned about reinforcement learning over the last six decades.

The fourth and final element of some reinforcement learning systems is a **model** of the environment. This is something that mimics the behavior of the environment, or more generally, that allows inferences to be made about how the environment will behave. For example, given a state and action, the model might predict the resultant next state and next reward. Models are used for planning, by which we mean any way of deciding on a course of action by considering possible future situations before they are actually experienced. Methods for solving reinforcement learning problems that use models and planning are called model-based methods, as opposed to simpler model-free methods that are explicitly trial-and-error learners—viewed as almost the opposite of planning. In

Say you are an agent, and your goal is to play chess. At every time step, you choose any **action **from the set of possible moves in the game. Your opponent is part of the environment; she responds with her own move, and the **state **you receive at the next time step is the configuration of the board, when it’s your turn to choose a move again. The **reward **is only delivered at the end of the game, and, let’s say, is +1 if you win, and -1 if you lose.

![](/assets/Screenshot 2019-07-09 at 3.58.24 PM.png)

This is an **episodic task**, where an episode finishes when the game ends. The idea is that by playing the game many times, or by interacting with the environment in many episodes, you can learn to play chess better and better.

* **task **is an instance of the reinforcement learning \(RL\) problem.
* **Continuing tasks **are tasks that continue forever, without end.
* **Episodic tasks **are tasks with a well-defined starting and ending point.
  * In this case, we refer to a complete sequence of interaction, from start to finish, as an **episode**
  * Episodic tasks come to an end whenever the agent reaches a **terminal state**

# Summary {#summary}

[![](https://d17h27t6h515a5.cloudfront.net/topher/2017/September/59c29f47_screen-shot-2017-09-20-at-12.02.06-pm/screen-shot-2017-09-20-at-12.02.06-pm.png)The agent-environment interaction in reinforcement learning. \(Source: Sutton and Barto, 2017\)](https://classroom.udacity.com/nanodegrees/nd009t/parts/d0974b3b-0900-4f45-bb95-f7f867a10329/modules/27a1333b-340e-4473-a4ab-082de8354cdd/lessons/86acfc34-0551-4cc6-8de4-a1ab2e66b5af/concepts/ee28399b-f809-4e2b-936b-5a88d7297899#)

### The Setting, Revisited {#the-setting-revisited}

---

* The reinforcement learning \(RL\) framework is characterized by an **agent **learning to interact with its **environment.**
* At each time step, the agent receives the environment's **state **\(_the environment presents a situation to the agent\)_, and the agent must choose an appropriate **action **in response. One time step later, the agent receives a **reward **\(_the environment indicates whether the agent has responded appropriately to the state_\) and a new **state**.
* All agents have the goal to maximize expected **cumulative reward**, or the expected sum of rewards attained over all time steps.

### Episodic vs. Continuing Tasks {#episodic-vs-continuing-tasks}

---

* A **task **is an instance of the reinforcement learning \(RL\) problem.
* **Continuing tasks **are tasks that continue forever, without end.
* **Episodic tasks **are tasks with a well-defined starting and ending point.
  * In this case, we refer to a complete sequence of interaction, from start to finish, as an **episode**.
  * Episodic tasks come to an end whenever the agent reaches a **terminal state**.

### The Reward Hypothesis {#the-reward-hypothesis}

---

* **Reward Hypothesis**: All goals can be framed as the maximization of \(expected\) cumulative reward.

### ![](/assets/Screenshot 2019-06-25 at 7.52.22 AM.png)Dynamic Programming {#cumulative-reward}

In **policy iteration **algorithms, you start with a random policy, then find the value function of that policy \(policy evaluation step\), then find a new \(improved\) policy based on the previous value function, and so on. In this process, each policy is guaranteed to be a strict improvement over the previous one \(unless it is already optimal\). Given a policy, its value function can be obtained using the Bellman operator.

In **value iteration**, you start with a random value function and then find a new \(improved\) value function in an iterative process, until reaching the optimal value function. Notice that you can derive easily the optimal policy from the optimal value function. This process is based on the optimality Bellman operator

In some sense, both algorithms share the same working principle, and they can be seen as two cases of the generalized policy iteration. However, the optimality Bellman operator contains a max operator, which is non linear and, therefore, it has different features. In addition, it's possible to use hybrid methods between pure value iteration and pure policy iteration.

### **Prediction and Control**

The difference between **prediction** and **control** is to do with goals regarding the policy. The policy describes the way of acting depending on current state, and in the literature is often noted as𝜋\(𝑎\|𝑠\)π\(a\|s\), the probability of taking action𝑎awhen in state𝑠s.

> So, my question is for prediction, predict what?

A prediction task in RL is where the policy is supplied, and the goal is to measure how well it performs. That is, to predict the expected total reward from any given state assuming the function𝜋\(𝑎\|𝑠\)π\(a\|s\)is fixed.

> for control, control what？

A control task in RL is where the policy is not fixed, and the goal is to find the optimal policy. That is, to find the policy𝜋\(𝑎\|𝑠\)π\(a\|s\)that maximises the expected total reward from any given state.

A control algorithm based on value functions \(of which Monte Carlo Control is one example\) usually works by also solving the prediction problem, i.e. it predicts the values of acting in different ways, and adjusts the policy to choose the best actions at each step. As a result, the output of the value-based algorithms is usually an approximately optimal policy and the expected future rewards for following that policy.

### On-policy vs Off-policy

There are two ideas to take away the Exploring Starts assumption: -

On-policy methods:

* Learning while doing the job 
* Learning policy π from the episodes that generated using π - 

Off-policy methods:

* Learning while watching other people doing the job 
* Learning policy π from the episodes generated using another policy u



