# Meta-RL
A variant implementation of the Meta-RL algorithm according to [(Wang et al 2016)](https://arxiv.org/abs/1611.05763) and [(Duan et al 2016)](https://arxiv.org/abs/1611.02779) in PyTorch applied to the Bandit problem. The architecture is shown below. There are two simplifications to the original algorithms. First, no observation is included in the input but only a<sub>t-1</sub> and r<sub>t-1</sub> from the previous time step. The Wang paper adds time index t as observation while the Duan paper suggests a zero value placeholder as observation for all time steps. Second, value is assumed to be the same for all time steps and esitmated as the average reward in an episode. The advantage function is computed accordingly as r<sub>t</sub> minus the average. These modifications make sense because there is really only one state in the Bandit problem and actions taken at different time steps are independant from each other. GRU is choosen over LSTM because it is simpler. The hidden state is reset for each episode.

![architecture](https://github.com/guyeresearch/meta-rl/blob/master/bandit/figures/schematic.png)

### Independant two-arm Bandit
Atfer traning in the independant two-arm Bandit problems for 20,000 episodes according to the papers using A2C, the model is tested for 1,000 episodes. The figure below shows the fraction of wrong pulls at each time step across the 1000 episodes. We can see the model learns quickly with the wrong pull fraction decreaing dramatically after first several steps.

![ti](https://github.com/guyeresearch/meta-rl/blob/master/bandit/figures/t_vs_mean_wrong_pull.png)


However, the wrong pull fraction does not settle to zero even in later time steps. It suggests there are tasks that the model fails to learn. The following figure plots the number of wrong pulls in an episode against the difference in expected reward  between the two arms. We can see the model can learn all tasks with a difference above 0.5 and fails to learn some of the tasks with difference below 0.5. This makes sense as small difference is harder to learn. It also indicates there is definitely space to improve the current model. 

![df](https://github.com/guyeresearch/meta-rl/blob/master/bandit/figures/mean_diff_vs_wrong_pulls.png)

### Dependant two-arm Bandit
Next, a similar model is trained for the dependant two-arm Bandit problems and tested for 1,000 episodes. The figure below shows the performance for both the independant and dependant models tested in the dependant environments. We can see both model learns quickly after first several time steps with the dependant model performing better as it settles to a lower fraction of wrong pulls.

![df](https://github.com/guyeresearch/meta-rl/blob/master/bandit/figures/eg_symbol_map.png)

The fourth figure shows the number of wrong pulls by the dependant model in an episode against the difference in expected reward  between the two arms. It clearly demonstrates that in the dependat two-arm tasks the model learns the easy tasks easier and hard tasks harder.

![df](https://github.com/guyeresearch/meta-rl/blob/master/bandit/figures/mean_dff_vs_wrong_dependant.png)

Overall, the implementation recapitulates the results described in the Wang paper.
