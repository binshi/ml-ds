Say you are an agent, and your goal is to play chess. At every time step, you choose any **action **from the set of possible moves in the game. Your opponent is part of the environment; she responds with her own move, and the **state **you receive at the next time step is the configuration of the board, when it’s your turn to choose a move again. The **reward **is only delivered at the end of the game, and, let’s say, is +1 if you win, and -1 if you lose.

This is an **episodic task**, where an episode finishes when the game ends. The idea is that by playing the game many times, or by interacting with the environment in many episodes, you can learn to play chess better and better.

* **task **is an instance of the reinforcement learning \(RL\) problem.
* **Continuing tasks **are tasks that continue forever, without end.
* **Episodic tasks **are tasks with a well-defined starting and ending point.
  * In this case, we refer to a complete sequence of interaction, from start to finish, as an **episode**
  * Episodic tasks come to an end whenever the agent reaches a **terminal state**
    .



