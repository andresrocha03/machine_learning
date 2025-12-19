# Statistics

## Statistical formalisation of a problem

In modern probability theory, we start with a sample space Ω, which represents all the possible outcomes of an experiment. Each element 𝜔 ∈ Ω is one particular outcome (for example, a sequence of “defective” or “non-defective” objects in a sample).

An event is a subset of Ω. For example, “at least two defective objects” is the set of all outcomes where this condition holds. The collection of all events is denoted by 𝐹, called a σ-algebra (tribe). This σ-algebra is a family of subsets of Ω that is stable under union, intersection, and complementation, ensuring that probabilities can be consistently assigned to these sets.

Often, we are not interested in the full detailed outcome, but only in a summary of it. To capture this, we define an observation space 𝑋, which may be different from Ω. For example, if we only care about the number of defective objects, then 𝑋 = {0,1,…,𝑛}. This observation space also has its own σ-algebra 𝐵(𝑋).

A random variable is then a measurable function:

𝑋:(Ω,𝐹)→(𝑋,𝐵(𝑋)),

which maps each outcome 𝜔 ∈ Ω to an observation in  𝑋. For instance, it may count how many defective objects appear in the sample.

In some simple situations, the observation space coincides with the sample space (e.g., rolling a die, where both Ω and X are {1,2,3,4,5,6}).

Finally, notice that up to this point, we have not introduced any probability law 𝑃. This law is a measure defined on the σ-algebra (Ω,F) that assigns probabilities to events. In statistics, however, the true law 𝑃 is unknown. The goal of statistical analysis is to use the observed data (the value of X) to extract information about the underlying probability law that generated it.


## Model parametrization
we can define a family of probabilities P from our observation space (X, B(X)). Each probability law from P can be defined by a parameter *θ* ∈ Θ, where Θ is a set called parameters space. If P = {Pθ , θ ∈ Θ}, we call P a statistical model.
