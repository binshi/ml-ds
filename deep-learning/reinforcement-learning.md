  
Beyond the agent and the environment, one can identify four main subelements of a reinforcement learning system: a **policy**, a **reward signal**, a **value function**, and, optionally, a **model of the environment**.

A **policy** defines the learning agent’s way of behaving at a given time. Roughly speaking, a policy is a mapping from perceived states of the environment to actions to be taken when in those states. It corresponds to what in psychology would be called a set of stimulus–response rules or associations. In some cases the policy may be a simple function or lookup table, whereas in others it may involve extensive computation such as a search process.

A **reward signal** defines the goal of a reinforcement learning problem. On each time step, the environment sends to the reinforcement learning agent a single number called the reward. The agent’s sole objective is to maximize the total reward it receives over the long run. The reward signal thus defines what are the good and bad events for the agent. In a biological system, we might think of rewards as analogous to the experiences of pleasure or pain. They are the immediate and defining features of the problem faced by the agent. The reward signal is the primary basis for altering the policy; if an action selected by the policy is followed by low reward, then the policy may be changed to select some other action in that situation in the future. In general, reward signals may be stochastic functions of the state of the environment and the actions taken.

Whereas the reward signal indicates what is good in an immediate sense, a **value function** specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Whereas rewards determine the immediate, intrinsic desirability of environmental states, values indicate the long-term desirability of states after taking into account the states that are likely to follow and the rewards available in those states. For example, a state might always yield a low immediate reward but still have a high value because it is regularly followed by other states that yield high rewards. Or the reverse could be true. To make a human analogy, rewards are somewhat like pleasure \(if high\) and pain \(if low\), whereas values correspond to a more refined and farsighted judgment of how pleased or displeased we are that our environment is in a particular state.

Rewards are in a sense primary, whereas values, as predictions of rewards, are secondary. Without rewards there could be no values, and the only purpose of estimating values is to achieve more reward. Nevertheless, it is values with which we are most concerned when making and evaluating decisions. **Action choices** are made based on value judgments. We seek actions that bring about states of highest value, not highest reward, because these actions obtain the greatest amount of reward for us over the long run. Unfortunately, it is much harder to determine values than it is to determine rewards. Rewards are basically given directly by the environment, but values must be estimated and re-estimated from the sequences of observations an agent makes over its entire lifetime. In fact, the most important component of almost all reinforcement learning algorithms we consider is a method for efficiently estimating values. The central role of value estimation is arguably the most important thing that has been learned about reinforcement learning over the last six decades.

The fourth and final element of some reinforcement learning systems is a **model** of the environment. This is something that mimics the behavior of the environment, or more generally, that allows inferences to be made about how the environment will behave. For example, given a state and action, the model might predict the resultant next state and next reward. Models are used for planning, by which we mean any way of deciding on a course of action by considering possible future situations before they are actually experienced. Methods for solving reinforcement learning problems that use models and planning are called model-based methods, as opposed to simpler model-free methods that are explicitly trial-and-error learners—viewed as almost the opposite of planning. In



Say you are an agent, and your goal is to play chess. At every time step, you choose any **action **from the set of possible moves in the game. Your opponent is part of the environment; she responds with her own move, and the **state **you receive at the next time step is the configuration of the board, when it’s your turn to choose a move again. The **reward **is only delivered at the end of the game, and, let’s say, is +1 if you win, and -1 if you lose.

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

### ![](/assets/Screenshot 2019-06-25 at 7.52.22 AM.png) {#cumulative-reward}



