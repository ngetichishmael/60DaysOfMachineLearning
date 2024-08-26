# Naïve Bayes Classifier

The Naïve Bayes classifier is a supervised machine learning algorithm that is used for classification tasks such as text classification. They use principles of probability to perform classification tasks.

## Problem statements
1. Classify customer reviews in positive and negative classes.
2. Identify which loan applicants are safe or risky
3. Predict which patients can suffer from diabetes disease.
4. Classify emails or messages into spam and not spam classes.
5. Build a recommendation system for your customers

## A little more information about the Naïve Bayes Classifier ..
It is based on the Bayes' Theorem, therefore also called *probabilistic classifier*.

It's a generative algorithm
    It seeks to model the distribution of inputs of a given class or category and does not learn which features are most important to differentiate between classes.

It works under the following assumptions:

1. Predictors in a Naïve Bayes model are conditionally independent, or unrelated to any of the other feature in the model.

2. All features contribute equally to the outcome.


## How Naïve Bayes Classifier works

It uses **conditional probabiliy** and **prior probability** to calculate the posterior probabilities using the following formula: 

$$
\text {posterior probability} = \frac{{conditional \, probability} \times prior \, probability}{evidence \, a.k.a \, "stabilizer"} 
$$

### Example
Calculate the probability of playing sports in given weather conditions

#### Approach 1 - using a single feature
1. Calculate the prior probability for given class labels

2. Find Likelihood probability with each attribute for each class
3. Put these value in Bayes Formula and calculate posterior probability.
4. See which class has a higher probability, given the input belongs to the higher probability class.

Consider the following data that shows whether it's safe to play or not:

| Whether  | Play |
|----------|------|
| Sunny    | No   |
| Sunny    | No   |
| Overcast | Yes  |
| Rainy    | Yes  |
| Rainy    | No   |
| Overcast | Yes  |
| Sunny    | Yes  |
| Rainy    | Yes  |
| Rainy    | Yes  |
| Sunny    | No   |
| Overcast | Yes  |
| Sunny    | Yes  |
| Overcast | Yes  |
| Rainy    | No   |

To simplify prior and posterior calculation, we'll use a frequency tables which contains the occurrence of labels for all features and 2 likelihood tables, one for prior probabilities of labels and the other for the posterior probability

Frequency table:
| Whether  | No | Yes |
|----------|----|-----|
| Overcast | 0  | 4   |
| Sunny    | 3  | 2   |
| Rainy    | 2  | 3   |
| **Total**| 5  | 9   |

Prior probabilities:
| Whether  | No    | Yes   |           |
|----------|-------|-------|-----------|
| Overcast | 0     | 4     | = $\frac{4}{14}$= 0.29 |
| Sunny    | 3     | 2     | = $\frac{5}{14}$ = 0.36 |
| Rainy    | 2     | 3     | = $\frac{5}{14}$ = 0.36 |
| **Total**| **5** | **9** | $\frac{5}{14}$ = 0.36 and $\frac{9}{14}$ = 0.64 |

Posterior probabilities:
| Whether  | No | Yes | Posterior Probability for No | Posterior Probability for Yes |
|----------|----|-----|------------------------------|-------------------------------|
| Overcast | 0  | 4   | $\frac{0}{5}$ = 0                      | $\frac{4}{9}$ = 0.44                    |
| Sunny    | 3  | 2   | $\frac{2}{5}$ = 0.4                    | $\frac{3}{9}$ = 0.33                    |
| Rainy    | 2  | 3   | $\frac{2}{5}$ = 0.4                    | $\frac{2}{9}$ = 0.22                    |
| **Total**| 5  | 9   |                              |                               |

1. Calculate the probability of playing when the whether is overcast:

i. Probability of playing:

P(Yes | Overcast) = P(Overcast | Yes) P(Yes) / P (Overcast) ..........(1)

Step 1: Calculate the prior and posterior probabilities
- Prior
    P(Overcast) = $\frac{4}{14}$= 0.29
    P(Yes) = $\frac{9}{14}$ = 0.64
- Posterior
    P(Overcast|Yes) = $\frac{4}{9}$ = 0.44

Step 2: Substitute in equation 1:
    P(Yes | Overcast) = 0.44 * 0.64 / 0.29 = **0.98**

ii. Probability of not playing:

P(No | Overcast) = P(Overcast | No) P(No) / P (Overcast) .........(2)

Step 1: Calculate the prior and posterior probabilities
- Prior
    P(Overcast) = $\frac{4}{14}$= 0.29
    P(No) = $\frac{5}{14}$ = 0.36
- Posterior
    P(Overcast|No) = $\frac{0}{9}$ = 0

Step 2: Substitute in equation 2:
P (No | Overcast) = 0 * 0.36 / 0.29 = **0**

##### Conclusion
The probability of a `Yes` class is higher. So, if the weather is overcast a player can play.

#### Approach 2 - using a multiple features
1. Calculate prior probability for given class labels.

2. Calculate conditional probability with each attribute for each class.
3. Multiply same class conditional probability.
4. Multiply prior probability with the step 3 probability.
5. See which class has higher probability; the class with the higher probability belongs to the given input set.

Consider the following table that shows the weather, temperature ans whether it's safe to play or not:

| Whether  | Temperature | Play |
|----------|-------------|------|
| Sunny    | Hot         | No   |
| Sunny    | Hot         | No   |
| Overcast | Hot         | Yes  |
| Rainy    | Mild        | Yes  |
| Rainy    | Cool        | Yes  |
| Rainy    | Cool        | No   |
| Overcast | Cool        | Yes  |
| Sunny    | Mild        | No   |
| Sunny    | Cool        | Yes  |
| Rainy    | Mild        | Yes  |
| Sunny    | Mild        | Yes  |
| Overcast | Mild        | Yes  |
| Overcast | Hot         | Yes  |
| Rainy    | Mild        | No   |

To calculate the probability of playing when the weather is overcast and the temperature is mild:

i. Probability of playing

P(Play=Yes | Weather=Overcast, Temp=Mild) = P(Weather=Overcast, Temp=Mild | Play=Yes)P(Play=Yes) ........(1)

P(Weather=Overcast, Temp=Mild | Play= Yes)= P(Overcast | Yes) P(Mild | Yes) ........(2)

1. Prior probability: P(Yes)= $\frac{9}{14}$ = 0.64

2. Posterior probability: P(Overcast |Yes) = 4/9 = 0.44 P(Mild |Yes) = $\frac{4}{9}$ = 0.44

3. Substitute posterior probability in equation (2) 
    P(Weather=Overcast, Temp=Mild | Play= Yes) = 0.44 * 0.44 = 0.1936

4. Substitute prior and posterior probabilities in equation (1) 

    P(Play= Yes | Weather=Overcast, Temp=Mild) = 0.1936 * 0.64 = 0.124

ii. Probability of not playing

P(Play = No | Weather=Overcast, Temp=Mild) = P(Weather=Overcast, Temp=Mild | Play= No)P(Play=No) ..........(3)

P(Weather=Overcast, Temp=Mild | Play= No)= P(Weather=Overcast |Play=No) P(Temp=Mild | Play=No) ..........(4)

1. Prior probability: P(No)= 5/14 = 0.36

2. Posterior Probability: P(Weather=Overcast | Play=No) = 0/9 = 0 P(Temp=Mild | Play=No)=2/5=0.4

3. Substitute posterior probability in equation (4)
    
    P(Weather=Overcast, Temp=Mild | Play= No) = 0 * 0.4= 0

4. Substitute prior and posterior probabilities in equation (3) 
    
    P(Play= No | Weather=Overcast, Temp=Mild) = 0*0.36=0

##### Conclusion
The probability of a `Yes` class is higher. So, if the weather is overcast and the temperature is mild, a player can play.


## Task
Read about the zero probability problem
