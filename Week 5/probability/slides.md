---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://cover.sli.dev
# some information about your slides (markdown enabled)
title: Probability Theory
info: |
  ## Slidev Starter Template
   
  

  Learn more at [Sli.dev](https://sli.dev)
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
---

# Probability Theory

Probability for Machine Learning

<!--
<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    Press Space for next page <carbon:arrow-right class="inline"/>
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/slidevjs/slidev" target="_blank" alt="GitHub" title="Open in GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>
-->

<!--
The last comment block of each slide will be treated as slide notes. It will be visible and editable in Presenter Mode along with the slide. [Read more in the docs](https://sli.dev/guide/syntax.html#notes)
-->

---
transition: fade-out
---

# What is Probability Theory?
- Mathematical framework for representing uncertain statements
- Provides a means of quantifying uncertainity and axioms for deriving new uncertain statements

## Probability in Machine Learning
Probability is used in 2 major ways:
  * Tells us how AI systems should reason
  * Theoretically analyse the behaviour of proposed AI systems

<br>
<br>


<!--
You can have `style` tag in markdown to override the style for the current page.
Learn more: https://sli.dev/features/slide-scope-style
-->

<style>
h1 {
  background-color: #2B90B6;
  background-image: linear-gradient(45deg, #4EC5D4 10%, #146b8c 20%);
  background-size: 100%;
  -webkit-background-clip: text;
  -moz-background-clip: text;
  -webkit-text-fill-color: transparent;
  -moz-text-fill-color: transparent;
}
</style>

---
transition: slide-up
level: 2
---

# Probability shprobability

- Machine Learning always deals with uncertain quantities and, sometimes, nondeterministic quantities.

Uncertainity can stem from 3 possible sources:
- Inherent stochasticity in the system being modeled
- Incomplete observability
- Incomplete modeling

---
transition: slide-up
level: 2
---

It's almost clear that we need a means to represent and reason out uncertainity, but it's not immediately obvious that probability theory can provide all the tools needed for AI applications.

## Frequentist probability
Probability theory was originally developed to analyze frequencies of events.

For example, it's easy to see how probality theory can be used to study events like flipping a fair coin

## Bayesian probability
What if an event can't be repeated infinitely?

In this case we use probability to represent a **degree of belief**, with absolute certainity of either of two possible outcomes.

---
transition: slide-up
level: 2
---

# Probability Distributions
- Describe how likely a random variable, or a set of them, takes on each of its possible states, depending on whether the variables are discrete or continuous.

    For example $x_1$ and $x_2$ are some of the possible states that a random variable $x$ can take
<ol>
<li><u>Probability Mass Function, P</u></li>

Describes a probability distribution over discrete variables

PMF assigns probability to every possible variable specific to the data attribute. The probability for all possible variables shouldn’t exceed 1. Each probability concerning a variable has to lie between (included) 0 and 1, and all of them have to add up to 1.

<li><u>Probability Density Function, p</u></li>

Describes a probability distribution over continuous variables

Continuous variables aren't finite therefore we use an integral to define PDF. The probability of every possible continuous value has to be greater than or equal to 0. The integration of all probabilities has to be equal to 1.

</ol>
---
transition: slide-up
level: 2
---

# Let's get Math-y

<li><u>Probability Mass Function, P</u></li>


PMF gives the probability that a discrete random variable 𝑋 takes on a particular value 𝑥; it maps each each possible value of 𝑋 to its probability.

The math:
- For discrete random variable 𝑋, the PMF is denoted by 𝑃(𝑋=𝑥), where:

<li>𝑃(𝑋=𝑥) ≥ 0 for all possible values 𝑥</li>
<li>The sum of the probabilities over all possible values of 𝑋 is 1</li>
$$
\begin{aligned}
\sum_x P(X = x) = 1
\end{aligned}
$$
<li>The PMF gives the exact probability of each value of 𝑋</li>



---
transition: slide-up
level: 2
---

Example:
- Let's say 𝑋 represents the result of a fair six-sided dice roll. Possible values of 𝑋 are 1, 2, 3, 4, 5 and 6. Because the dice is fair:
$$
\begin{aligned}
P(X = 1) = P(X = 2) = ... = P(X = 6) = \frac{1}{6}
\end{aligned}
$$

- So, PMF $P(x)$ is:
$$
P(x) = \begin{cases} 
\frac{1}{6} & \text{for } x = 1, 2, 3, 4, 5, 6, \\
0 & \text{otherwise}.
\end{cases}
$$

- Summing up the probabilities:
$$
\begin{aligned}
\sum_{x=1}^{6} \frac{1}{6} = 6 * \frac{1}{6} = 1
\end{aligned}
$$

---
transition: slide-up
level: 2
---

In summary, PMF is just a fancy way of saying "Here's the list of all possible outcomes and their chances of happening!"

For a fair coin, the PMF shows that every outcome has an equal chance, $\frac{1}{2}$.
​

The probabilities in a PMF always add up to 1, because something must happen!


---
transition: slide-up
level: 2
---

<li><u>Probability Density Function, p</u></li>

PDF tells us how **dense** the probability is at any point 𝑥.

To get the probability of 𝑥 falling within a certain interval, we integrate the PDF over that interval.

The math:
- For a continuous random variable 𝑋, PDF $p(x)$ has the following properties:

  1. $p(x) ≥ 0$ for all 𝑥
  2. The area under the entire PDF curve equals 1:
$$
\begin{aligned}
\int_{-\infty}^{\infty} p(x) \,dx = 1
\end{aligned}
$$
  3. The probability that 𝑋 falls within an interval $[𝑎,𝑏]$ is given by the integral of the PDF over that interval:
$$
\begin{aligned}
P(a \leq X \leq b) = \int_{a}^{b} p(x) \, dx.
\end{aligned}
$$


---
transition: slide-up
level: 2
---

Example:

Let's say the PDF of a random variable $X$ is given by:
$$
p(x) = \begin{cases} 
\frac{1}{10} & \text{if } 0 \leq x \leq 10 \\
0 & \text{otherwise}.
\end{cases}
$$

This PDF represents a uniform distribution over the interval $[0,10]$

Calculating the total area under the curve:
$$
\int_{0}^{10} \frac{1}{10} \, dx = \frac{1}{10} * (10 - 0) = 1
$$

To find the probability that $X$ is between 2 and 7, calculate the integral over that interval:
$$
\begin{aligned}
P(2 \leq X \leq 7) = \int_{2}^{7} \, dx \frac{1}{10} * (7 - 2) = 0.5
\end{aligned}
$$

There's a 50% chance that $X$ will be between 2 and 7.

---
transition: slide-up
level: 2
---

# Other Probability Distributions

1. <u>Marginal Probability Distribution</u>

Probability distribution over a subset of variables

- Suppose we have discrete random variables x and y and we know $P(x,y)$, we can find $P(x)$ using the sum rule:
$$
\forall x \in x, \, P(x = x) = \sum_y P(x = x, y = y)
$$

- For continuous variables, we use integration instead of summation:
$$
p(x) = \int_ p(x,y) \, dy
$$



---

2. <u>Conditional Probability Distribution</u>

This is the probability of an event occuring, given that some other event has happened.

It is denoted as y $= y$ given x $=x$ as $P(y = y | x = x)$, and can be computed as follows:
$$
P(y = y | x = x) = \frac{P(y = y, x = x)}{P(x = x)}
$$
given $P(x = x) > 0$


---
transition: slide-up
level: 2
---

# Bayes' Rule

$$
P(A|B) = P(A) * \frac{P(B|A)}{P(B)}
$$

Tells us how to calculate a conditional probability with information we already have.

It can be applied to any type of events, with any number of discrete or continuous outcomes.

Think of it in terms of two events – a hypothesis (which can be true or false) and evidence (which can be present or absent).

We can re-write the formula as:
$$
P(Hypothesis|Evidence) = P(Hypothesis) * \frac{P(Evidence|Hypothesis)}{P(Evidence)}
$$

---
transition: slide-up
level: 2
---

## Let's break it down...

The Bayes' formula has 4 parts:

1. Posterior probability, $P(A|B)$ - updated probability after the evidence is considered
2. Prior probability, $P(A)$ - the initial belief
3. Likelihood, $P(B|A)$ - probability of the evidence, given the belief is true
4. Marginal probability, $P(B)$ - probability of the evidence, under any circumstance

---

Example:

I spoke to Kepha earlier and he was his jolly self. I want to estimate the probability of him showing up for ML class today.

- Step 1 -> Posterior probability of showing up given jolly mood
- Step 2 -> Estimate the prior probability of showing up as 40%
- Step 3 -> Estimate the likelihood probability of showing up for ML, given jolly mood as 80%
- Step 4 -> Estimate the marginal probability of the jolly mood. This could be because:
  - he's had a good day (40% of the time, by 80% probability)
  - any other reason (60% of the time, by maybe 6% probability)

Piecing it all together:
$$
P(class|jolly) = P(class) * \frac{P(jolly|class)}{P(jolly|class) + P(jolly|no class)}
$$

$$
= 0.4 * \frac{0.8}{(0.4 * 0.8) + (0.6 * 0.06)}
= 0.89
$$

---

# Resources

1. [Deep Learning - chapter 3](https://www.deeplearningbook.org/)
2. [Bayes' Rule](https://www.freecodecamp.org/news/bayes-rule-explained/)
