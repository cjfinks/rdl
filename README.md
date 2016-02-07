# TODO

### Introduction
* Need to define "k-sparse" before using it for the first time
* Derive geometric properties of A from Spark Condition?
* Motivate "data spread" as a necessary condition.

### Deterministic Uniqueness Theorem
* Figure: |A - BPD| and |b - PDa| as a function of noise
  * Complete (ICA)
  * Overcomplete (SCA)
* Calculate C and C' to overlay on figures

### Can we make N smaller?
* Leverage vector space structure
* Figure: |A - BPD| as a function of N

### Probabilistic theorems
* Need proofs
  * Thm 2 idea: Create "virtual data" spanning all possible supports through lin. comb. of data with supports in T. Then do same proof as in HS11. 
  * Thm 2 idea: Figure out how spark cond. on A and consecutive support set T restrict the support sets of B to be consecutive as well (intersection inequality?)
* General random sampling - given N samples with random k-supports, what are the odds of every element in [m] being the intersection of k support sets which each have k+1 samples?

### Combinatorial Matrix Theory
* Is the proof of Lemma 3 good enough?
