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

### Cumulative Reward {#cumulative-reward}

---

* The **return at time step t **is G&lt;sub&gt;t&lt;sub&gt; := R\_{t+1} + R\_{t+2} + R\_{t+3} + \ldots
  G
  t
  ​
  :
  =
  R
  t
  +
  1
  ​
  +
  R
  t
  +
  2
  ​
  +
  R
  t
  +
  3
  ​
  +
  …
* The agent selects actions with the goal of maximizing expected \(discounted\) return. \(
  _Note: discounting is covered in the next concept._
  \)

  


