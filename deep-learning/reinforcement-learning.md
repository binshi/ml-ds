[https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/](https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/)

[https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/)

[https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)

[https://github.com/aikorea/awesome-rl](https://github.com/aikorea/awesome-rl)

[https://joshgreaves.com/reinforcement-learning/understanding-rl-the-bellman-equations/](https://joshgreaves.com/reinforcement-learning/understanding-rl-the-bellman-equations/)

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

The **state value function** describes the value of a state when following a policy. It is the expected return when starting from state![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-ae1901659f469e6be883797bfd30f4f8_l3.svg "s") acting according to our policy![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-26d6788550ffd50fe94542bb3e8ee615_l3.svg "\pi"):

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-814f9fadd2ab8ee9cb85e12999a17eec_l3.svg "\\[V^{\pi}\(s\) = \mathbb{E}\_{\pi} \big\[R\_t \| s\_t = s \big\] \\]") -- \(1\)

The **action value function** tells us the value of taking an action in some state when following a certain policy. It is the expected return given the state and action under![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-26d6788550ffd50fe94542bb3e8ee615_l3.svg "\pi"):

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-5171067fa940af561a4eebe7d3c2d190_l3.svg "\\[Q^{\pi}\(s, a\) = \mathbb{E}\_{\pi} \big\[ R\_t \| s\_t = s, a\_t = a \big\] \\]")-- \(2\)

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-5bed2094d6826d2cb5b8f63cef61b30e_l3.svg "\mathcal{P}") is the **transition probability**. If we start at state![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-ae1901659f469e6be883797bfd30f4f8_l3.svg "s")and take action![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-5c53d6ebabdbcfa4e107550ea60b1b19_l3.svg "a")we end up in state![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-7bc7053f2932cafc2f41141faa219498_l3.svg "s&apos;")with probability![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-12aae9389c7680f52974720aa071b6fb_l3.svg "\mathcal{P}\_{s s&apos;}^{a}").

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-460feeb1f51606e1755e5b10cb6ee697_l3.svg "\\[\mathcal{P}\_{s s&apos;}^{a} = Pr\(s\_{t+1} = s&apos; \| s\_t = s, a\_t = a\)\\]")

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-f751bd5163b0f1cb61f7974f3b249369_l3.svg "\mathcal{R}\_{s s&apos;}^{a}") is another way of writing the **expected \(or mean\) reward** that we receive when starting in state![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-ae1901659f469e6be883797bfd30f4f8_l3.svg "s"), taking action![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-5c53d6ebabdbcfa4e107550ea60b1b19_l3.svg "a"), and moving into state![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-7bc7053f2932cafc2f41141faa219498_l3.svg "s&apos;").  
![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-780b8ccdb127cddc56fab4e2ce32d3d5_l3.svg "\\[\mathcal{R}\_{s s&apos;}^{a} = \mathbb{E}\[ r\_{t+1} \| s\_t = s, s\_{t+1} = s&apos;, a\_t = a \]\\]")

Finally, with these in hand, we are ready to derive the Bellman equations. We will consider the Bellman equation for the state value function. Using the definition for return, we could rewrite equation \(1\) as follows:

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-e5ae4c2fb5d42ec59599abb215da65ad_l3.svg "\\[V^{\pi}\(s\) =\mathbb{E}\_{\pi} \Big\[r\_{t+1} + \gamma r\_{t+2} + \gamma^2 r\_{t+3} + ... \| s\_t = s \Big\] = \mathbb{E}\_{\pi} \Big\[ \sum\_{k=0}^{\infty} \gamma^k r\_{t+k+1} \| s\_t = s \Big\]\\]")

If we pull out the first reward from the sum, we can rewrite it like so:

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-fd233324765a62145b2cd184ea79bfdc_l3.svg "\\[V^{\pi}\(s\) = \mathbb{E}\_{\pi} \Big\[r\_{t+1} + \gamma \sum\_{k=0}^{\infty} \gamma^k r\_{t+k+2} \| s\_t = s \Big\]\\]")

The expectation here describes what we expect the return to be if we continue from state![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-ae1901659f469e6be883797bfd30f4f8_l3.svg "s") following policy![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-26d6788550ffd50fe94542bb3e8ee615_l3.svg "\pi"). The expectation can be written explicitly by summing over all possible actions and all possible returned states. The next two equations can help us make the next step.

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-ca9f4855134a11198035c0954a7d552d_l3.svg "\\[\mathbb{E}\_{\pi} \[r\_{t+1} \| s\_t = s\] = \sum\_{a} \pi\(s, a\) \sum\_{s&apos;} \mathcal{P}\_{s s&apos;}^{a} \mathcal{R}\_{s s&apos;}^{a}\\]")

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-59ddf9765906161069a1c1095de964f4_l3.svg "\\[\mathbb{E}\_{\pi} \Big\[ \gamma \sum\_{k=0}^{\infty} \gamma^k r\_{t+k+2} \| s\_t = s \Big\] = \sum\_{a} \pi\(s, a\) \sum\_{s&apos;} \mathcal{P}\_{s s&apos;}^{a} \gamma \mathbb{E}\_{\pi} \Big\[ \sum\_{k=0}^{\infty} \gamma^k r\_{t+k+2} \| s\_{t+1} = s&apos; \Big\]\\]")

By distributing the expectation between these two parts, we can then manipulate our equation into the form:

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-32cdfdc98b2312015a69a4567e6f331b_l3.svg "\\[V^{\pi}\(s\) = \sum\_{a} \pi \(s, a\) \sum\_{s&apos;} \mathcal{P}\_{s s&apos;}^{a} \Bigg\[ \mathcal{R}\_{s s&apos;}^{a} +\gamma \mathbb{E}\_{\pi} \Big\[ \sum\_{k=0}^{\infty} \gamma^k r\_{t+k+2}  \| s\_{t+1} = s&apos; \Big\] \Bigg\]\\]")

Now, note that equation \(1\) is in the same form as the end of this equation. We can therefore substitute it in, giving us

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-702aff0ad857f6d6bb470e10b43de4e9_l3.svg "\\[V^{\pi}\(s\) = \sum\_{a} \pi \(s, a\) \sum\_{s&apos;} \mathcal{P}\_{s s&apos;}^{a} \Big\[ \mathcal{R}\_{s s&apos;}^{a} + \gamma V^{\pi}\(s&apos;\) \Big\] \\]") -- \(3\)

The Bellman equation for the action value function can be derived in a similar way. The specific steps are included at the end of this post for those interested. The end result is as follows:

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-90b37bda96ae8a5a5cee8b8d5b799b27_l3.svg "\\[Q^{\pi}\(s,a\) = \sum\_{s&apos;} \mathcal{P}\_{s s&apos;}^{a} \Big\[ \mathcal{R}\_{s s&apos;}^{a} + \gamma \sum\_{a&apos;} \pi \(s&apos;, a&apos;\) Q^{\pi}\(s&apos;, a&apos;\) \Big\]\\]") --\(4\)

The importance of the Bellman equations is that they let us express values of states as values of other states. This means that if we know the value of![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-e1ca7f575bfa511f2754fe9d99096594_l3.svg "s\_{t+1}"), we can very easily calculate the value of![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-e86795deb37ff5f5055e741b17eb25d7_l3.svg "s\_t"). This opens a lot of doors for iterative approaches for calculating the value for each state, since if we know the value of the next state, we can know the value of the current state. The most important things to remember here are the numbered equations. Finally, with the Bellman equations in hand, we can start looking at how to calculate optimal policies and code our first reinforcement learning agent.

**Deriving the Bellman equation for the Action Value Function**

Following much the same process as for when we derived the Bellman equation for the state value function, we get this series of equations, starting with equation \(2\):

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-071e92db4f4c26ef2961ab200f0b1d12_l3.svg "\\[Q^{\pi}\(s, a\) = \mathbb{E}\_{\pi} \Big\[ r\_{t+1} + \gamma r\_{t+2} + \gamma^2 r\_{t+3} + ... \| s\_t = s, a\_t = a \Big\] = \mathbb{E}\_{\pi} \Big\[ \sum\_{k = 0}^{\infty} \gamma^k r\_{t + k + 1} \| s\_t = s, a\_t = a \Big\]\\]")

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-47ece2d37bc8fd413541fd6ced3c29a2_l3.svg "\\[Q^{\pi}\(s,a\) = \mathbb{E}\_{\pi} \Big\[ r\_{t+1} + \gamma \sum\_{k=0}^{\infty}\gamma^k r\_{t+k+2} \| s\_t = s, a\_t = a \Big\]\\]")

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-a2de582ca44d7985c83aefc9374c3f3c_l3.svg "\\[Q^{\pi}\(s,a\) = \sum\_{s&apos;} \mathcal{P}\_{s s&apos;}^{a} \Bigg\[ \mathcal{R}\_{s s&apos;}^{a} + \gamma \mathbb{E}\_{\pi} \Big\[ \sum\_{k=0}^{\infty} \gamma^k r\_{t+k+2} \| s\_{t+1} = s&apos; \Big\] \Bigg\]\\]")

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-11c7c8ff51bac03e422203a64e4e0268_l3.svg "\\[Q^{\pi}\(s,a\) = \sum\_{s&apos;} \mathcal{P}\_{s s&apos;}^{a} \Bigg\[ \mathcal{R}\_{s s&apos;}^{a} + \gamma \sum\_{a&apos;} \mathbb{E}\_{\pi} \Big\[ \sum\_{k=0}^{\infty} \gamma^k r\_{t+k+2} \| s\_{t+1} = s&apos;, a\_{t+1} = a&apos; \Big\] \Bigg\]\\]")

![](https://joshgreaves.com/wp-content/ql-cache/quicklatex.com-99c8f3a5ae50d58f9ca64c714a63f62c_l3.svg "\\[Q^{\pi}\(s,a\) = \sum\_{s&apos;} \mathcal{P}\_{s s&apos;}^{a} \Big\[ \mathcal{R}\_{s s&apos;}^{a} + \gamma \sum\_{a&apos;} \pi \(s&apos;, a&apos;\) Q^{\pi}\(s&apos;, a&apos;\) \Big\]\\]")

### On-policy vs Off-policy

There are two ideas to take away the Exploring Starts assumption: -

On-policy methods:

* Learning while doing the job 
* Learning policy π from the episodes that generated using π - 

Off-policy methods:

* Learning while watching other people doing the job 
* Learning policy π from the episodes generated using another policy u

Monte-Carlo Reinforcement learning:

* MC methods learn directly from episodes of experience

* MC is model-free: no knowledge of MDP transitions / rewards

* MC learns from complete episodes: no bootstrapping

* MC uses the simplest possible idea: value = mean return

Caveat: can only apply MC to episodic MDPs

* All episodes must terminate

Dynamic programming, MonteCarlo methods and Temporal Difference methods:

* TD exploits Markov property Usually more efficient in Markov environments

* MC does not exploit Markov property Usually more effective in non-Markov environments

* **Bootstrapping: update involves an estimate**

  MC does not bootstrap

  DP, TD bootstraps

  **Sampling: update samples an expectation**

  MC, TD samples

  DP does not sample

Monte-Carlo

→ it only works for episodic tasks

→ it can only learn from complete sequences

→it has to wait until the end of the episode to get the reward

TD

→ it only for both episodic and continuous tasks

→it can learn from incomplete sequences

→ it will only wait until the next time step to update the value estimates.

![](/assets/Screenshot 2019-07-29 at 6.01.26 PM.png)

![](/assets/Screenshot 2019-07-29 at 6.01.00 PM.png)![](/assets/Screenshot 2019-07-29 at 6.01.12 PM.png)

