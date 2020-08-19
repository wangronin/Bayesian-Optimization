# Bayesian Optimization Library

Experimental Bayesian Optimization library.

## Structure of the Implementation

* `BayesOpt.py`: the Bayesian Optimization algorithm
* `InfillCriteria.py`: the decision-making acquisition functions.
* `Surrogate.py`: the implementation/wrapper of random forests model.
* `SearchSpace.py`: implementation of the configuration space specification. It is used to specify the search / configure space conveniently.
* `optimizer/`: the optimization algorithm to maximize the infill-criteria, two algorithms are implemented:
      1. **CMA-ES**: Covariance Martix Adaptation Evolution Strategy for _continuous_ optimization problems.
      2. **MIES**: Mixed-Integer Evolution Strategy for mixed-integer/categorical optimization problems.

## The Main File: BayesOpt.py

The parameters of the class constructor are listed below:
**the following should be updated**

* search_space : instance of SearchSpace type
* obj_func : callable,
        the objective function to optimize
* surrogate: surrogate model, currently support either GPR or random forest
* minimize : bool,
        minimize or maximize
* noisy : bool,
        is the objective stochastic or not?
* eval_budget : int,
        maximal number of evaluations on the objective function
* max_iter : int,
        maximal iteration
* n_init_sample : int,
        the size of inital Design of Experiment (DoE),
        default: 20 * dim
* n_point : int,
        the number of candidate solutions proposed using infill-criteria,
        default : 1
* n_jobs : int,
        the number of jobs scheduled for parallelizing the evaluation.
        Only Effective when n_point > 1
* backend : str,
        the parallelization backend, supporting: 'multiprocessing', 'MPI', 'SPARC'
* optimizer: str,
        the optimization algorithm for infill-criteria,
        supported options: 'MIES' (Mixed-Integer Evolution Strategy for random forest), 'BFGS' (quasi-Newtion for GPR)

Note that the dimensionality is retrieved from the search_space argument.

## Infill Criteria

The following infill-criteria are implemented in the library:

* Expected Improvement (EI)
* Probability of Improvement (PI) / $\epsilon$-Probability of Improvement
* Upper Confidence Bound (UCB)
* Moment-Generating Function of Improvement (MGFI): proposed in Hao's SMC'17 paper
* Generalized Expected Improvement (GEI)

For sequential working mode, Expected Improvement is used by default. For parallelization mode ($q$-point strategy), MGFI is enabled by default.

## Surrogate Model

The meta (surrogate)-model used in Bayesian optimization. The basic requirement for such a model is to provide the uncertainty quantification (either empirical or theorerical) for the prediction. To easily handle the categorical data, __random forest__ model is used by default. The implementation here is based the one in _scikit-learn_, with modifications on uncertainty quantification.

## Search Space

To ease the work on specifiying mixed-integer search space, a SearchSpace class is implemented.

## A brief Introduction to Bayesian Optimization

Bayesian optimization is __sequential design strategy__ that does not require the derivatives of the objective function and is designed to solve expensive global optimization problems. Compared to alternative optimization algorithms (or other design of experiment methods), the very distinctive feature of this method is the usage of a __posterior distribution__ over the (partially) unknown objective function, which is obtained via __Bayesian inference__. This optimization framework is proposed by Jonas Mockus and Antanas Zilinskas, et al.

Formally, the goal is to approach the global optimum, using a sequence of variables:
$$\mathbf{x}_1,\mathbf{x}_2, \ldots, \mathbf{x}_n \in S \subseteq \mathbb{R}^d,$$
which resembles the search sequence in stochastic hill-climbing, simulated annealing and (1+1)-strategies. The only difference is that such a sequence is __not__ necessarily random and it is actually deterministic (in principle) for Bayesian optimization. In order to approach the global optimum, this algorithm iteratively seeks for an optimal choice as the next candidate variable adn the choice can be considered as a decision function:
\[\mathbf{x}_{n+1} = d_n\left(\{\mathbf{x}_i\}_{i=1}^n, \{y_i\}_{i=1}^n \right), \quad y_i = f(\mathbf{x}_i) + \varepsilon,\]
meaning that it takes the history of the optimization in order to make a decision. The quality of a decision can be measured by the following loss function that is the optimality error or optimality gap:
$$\epsilon(f, d_n) = f(\mathbf{x}_n) - f(\mathbf{x}^*),\quad \text{in the objective space,}$$
or
$$\epsilon(f, d_n) = ||\mathbf{x}_n - \mathbf{x}^*||,\quad \text{in the decision space,}$$
Using this error measure, the step-wise optimization task can be formulated as:
$$d_n^* = \operatorname{arg\,min}_{d_n}\epsilon(f, d_n)$$
This optimization task requires the full knowledge on the objective function $f$, which is not available (of course...). Alternatively, in Bayesian optimization, it is assumed that the objective function belongs to a function family or _function space_, e.g. $\mathcal{C}^1(\mathbb{R}^d):$ the space of continuous functions on $\mathbb{R}^d$ that have continuous first-order derivatives.

Without loss of generality, let's assume our objective function is in such a function space:
$$f\in \mathcal{C}^1\left(\mathbb{R}^d\right)$$
Then, it is possible to pose a prior distribution of _function_ $f$ in $\mathcal{C}^1\left(\mathbb{R}^d\right)$ (given that this space is also measurable):
$$f \sim P(f)$$
This prior distribution models our belief on $f$ before observing any data samples from it. The likelihood can be obtained by observing (noisy) samples on this function $f$, which is the  joint probability (density) of observing the data set given the function $f$:
$$P(\{\mathbf{x}_i\}_{i=1}^n, \{y_i\}_{i=1}^n | f)$$ 
Then, using the Bayes rule, the __posterior__ distribution of $f$ is calculated as:
$$\underbrace{P(f | \{\mathbf{x}_i\}_{i=1}^n, \{y_i\}_{i=1}^n)}_{\text{posterior}} \propto \underbrace{P(\{\mathbf{x}_i\}_{i=1}^n, \{y_i\}_{i=1}^n | f)}_{\text{likelihood/evidence}}\underbrace{P(f)}_{\text{prior}}$$
Intuitively, the posterior tells us that how the function $f$ distributes once some data/evidence from it are available. At this point, our knowledge on $f$ is better than nothing, but it is represented as a distribution, containing uncertainties. Therefore, the optimal decision-making task can be tackled by optimizing the expected loss function:
$$
d_n^{BO} = \operatorname{arg\,min}_{d_n}\mathbb{E}\left[\epsilon(f, d_n) \; |\; \{\mathbf{x}_i\}_{i=1}^n, \{y_i\}_{i=1}^n)\right]\\
\quad\quad\quad\;\:= \operatorname{arg\,min}_{d_n}\int_{\mathcal{C}^1\left(\mathbb{R}^d\right)}\epsilon(f, d_n) \mathrm{d} P(f | \{\mathbf{x}_i\}_{i=1}^n, \{y_i\}_{i=1}^n)
$$
In practice, the loss function $\epsilon$ is not used because there is not knowledge on the global optimum of $f$. Instead, the improvement between two iterations/steps (as defined in section $1$) is commonly used. The expectation in Eq.~\ref{eq:infill} is the so-called __infill-criteria__, __acquisition function__ or __selection criteria__. Some commonly used ones are: Expected Improvement (EI), Probability of Improvement (PI) and Upper (lower) Confidence Bound (UCB).

As for the prior distribution, the mostly used one is Gaussian and such a distribution on functions is __Gaussian process__ (random field). It is also possible to consider the Gaussian process as a _surrogate_ or simple a model on the unknown function $f$. In this sense, some other models are also often exploited, e.g. Student's t process and random forest (in SMAC and SPOT). However, the usage of random forest models brings me some additional thinkings (see the following sections).

The same algorithmic idea was re-advertised in the name of "Efficient Global Optimization"" (EGO) by Donald R. Jones. As pointed out in Jone's paper on taxonomy of global optimization methods, the bayesian optimization can be viewed as a special case of a broader family of similar algorithms, that is call "global optimization based on response surfaces", Model-based Optimization (MBO) (some references here from Bernd Bischl) or Sequential MBO.

## Reference
To be added...