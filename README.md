# dslr

[What's logistic regression?](https://aws.amazon.com/what-is/logistic-regression)

["Gradient Descent for Logistics Regression in Python"](https://medium.com/@IwriteDSblog/gradient-descent-for-logistics-regression-in-python-18e033775082)

A [Youtube playlist on logistic regression](https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe) by StatQuest with Josh Starmer.


https://heena-sharma.medium.com/logistic-regression-7773c76522d6
https://heena-sharma.medium.com/logistic-regression-python-implementation-from-scratch-without-using-sklearn-d3fca7d3dae7
https://www.kaggle.com/code/sagira/logistic-regression-math-behind-without-sklearn

### Computing Correlations using ANOVA

#### Mathematical Formulas

To compute the correlation between each house (categorical variable) and the scores of the students of that house for a given discipline (continuous variable) we perform a test called [ANOVA](https://datascience.stackexchange.com/questions/893/how-to-get-correlation-between-two-categorical-variable-and-a-categorical-variab) (Analysis of Variance).

[This test](https://en.wikipedia.org/wiki/F-test#Formula_and_calculation) allows us to ascertain whether a given house has an impact on the scores of its students in a given discipline.

To perform ANOVA for each discipline we apply the following formula:

$\displaystyle F = \frac{Between-Group\;Variance}{Within-Group\;Variance}$

Between-Group Variance, or Mean Square Between (MSB), measures how much the group means (house means) deviate from the overall mean. This shows the variation due to the house effect.

To calculate MSB, we need to calculate the Sum of Squares Between (SSB), the measure of the variation between the means of different groups (or categories). It is used to quantify how much the average scores of students in different Hogwarts houses differ from the overall average score.

It is computed as follows:

$\displaystyle SSB = \sum _{i=1}^{K}n_{i}({\bar {Y}}_{i\cdot }-{\bar {Y}})^{2}$

$\displaystyle MSB = \frac{SSB}{K - 1}$

where:
- $\displaystyle {\bar {Y}}_{i\cdot }$ denotes the mean in the *i*th group (e.g., the mean of arithmancy score of students in the *i*th house),
- $\displaystyle n_{i}$ is the number of observations in the *i*th group (in our case, the number of students in the *i*th house),
- $\displaystyle {\bar {Y}}$ denotes the overall mean of the data,
- $\displaystyle K$ denotes the number of groups (here the houses).

When performing ANOVA, SSB helps assess whether the observed group differences are statistically significant. The MSB derived from SSB is then compared against the within-group variance (or error variance) to determine if the differences between group means are greater than what could be attributed to random chance.

Within-Group Variance, or Mean Square Within (MSW), measures how much the individual scores within each house deviate from their respective house means. This shows the variation due to individual differences within each house.

Within-Group Variance reflects the dispersion of data points around the group mean within each group. It tells you how much variability exists within each group (e.g., within each house in our example).
A smaller within-group variance indicates that the observations in each group are closer to their group mean, while a larger within-group variance indicates that the observations are more spread out.

To calculate MSW we compute the Sum of Squares Within (SSW), also known as the Sum of Squares Error (SSE), used to measure the variability within each group around their respective group means. This measurement helps in understanding how much of the total variability in the data is due to differences within the groups rather than differences between the groups.

Here are the formulas we use:

$\displaystyle SSW = \sum _{i=1}^{K}\sum _{j=1}^{n_{i}}\left(Y_{ij}-{\bar {Y}}_{i\cdot }\right)^{2}$

$\displaystyle MSB = \frac{SSW}{N - K}$

where:
- $\displaystyle K$ is the number of groups,
- $\displaystyle Y_{ij}$ is the *j*th observation in the *i*th out of $\displaystyle K$ groups,
- $\displaystyle {\bar {Y}}_{i\cdot }$ is the mean of the *i*th group,
- $\displaystyle n_{i}$ is the number of observations in the *i*th group,
- $\displaystyle N$ is the overall sample size.

__F-Statistic__ is then calculated as:

$\displaystyle F=\frac{MSB}{MSW}$

This __F-statistic__ follows the F-distribution with degrees of freedom $\displaystyle d_{1}=K-1$ (numerator) and $\displaystyle d_{2}=N-K$ (denominator) under the null hypothesis that all group means are equal.

#### Python Implementation

[The one-way ANOVA](https://dzone.com/articles/correlation-between-categorical-and-continuous-var-1) tests the null hypothesis that two or more groups have the same population mean. It may be performed in Python using the [f_oneway function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html) from the [stats module](https://docs.scipy.org/doc/scipy/reference/stats.html) of SciPy.

However, we will compute it by ourselves. Here is how we perform the calculation of MSB:

```python
for col in df_cleaned.columns[1:]:
    discipline = df_cleaned[col]

    ssb_total = 0
    overall_mean = custom_mean(discipline)

    for df_house in df_cleaned["Hogwarts House"].unique():
        # Get the discipline scores for the current house
        house_scores = discipline[df_cleaned["Hogwarts House"] == df_house]
        house_scores = house_scores[pd.notna(house_scores)]

        # Calculate the mean for the current house
        house_mean = custom_mean(house_scores)

        # Number of observations in the current house
        n_j = len(house_scores)

        # Accumulate SSB
        ssb_total += n_j * (house_mean - overall_mean) ** 2
    
    # Calculate the mean SSB
    msb = ssb_total / (len(df_cleaned["Hogwarts House"].unique()) - 1)

    # Print SSB and MSB
    print("Total SSB:", ssb_total)
    print("MSB:", msb)
```
and here is how we compute MSW:

```python
for col in df_cleaned.columns[1:]:
    discipline = df_cleaned[col]

    ssw_total = 0

    for df_house in df_cleaned["Hogwarts House"].unique():
        # Get the discipline scores for the current house
        house_scores = discipline[df_cleaned["Hogwarts House"] == df_house]
        house_scores = house_scores[pd.notna(house_scores)]

        # Calculate the mean for the current house
        house_mean = custom_mean(house_scores)

        # For each observation of the current house
        for el in house_scores:
            # Accumulate SSW
            ssw_total += (el - house_mean) ** 2

    # Calculate the mean SSW
    msw = ssw_total / (len(df_cleaned) - len(df_cleaned["Hogwarts House"].unique()))

    # Print SSW and MSW
    print("Total SSW:", ssw_total)
    print("MSW:", msw)
```
### __*p*-value__

A [__*p*-value__](https://en.wikipedia.org/wiki/P-value#Definition_and_interpretation) is used to verify the [__null hypothesis__](https://en.wikipedia.org/wiki/Null_hypothesis) $\displaystyle H_{0}$, the claim that the effect being studied does not exist.
The __null hypothesis__ is rejected if the __*p*-value__ is less than or equal to a predefined threshold value called $\displaystyle \alpha$, which is referred to as the significance level. $\displaystyle \alpha$ is not derived from the data, but rather is set by the researcher before examining the data. It is commonly set to 0.05.

To calculate __*p*-value__ we need to calculate the __cumulative distribution function__.

#### The __Cumulative Distribution Function__

The __cumulative distribution function__ (__CDF__) of a real-valued random variable $\displaystyle X$, or just distribution function of $\displaystyle X$, evaluated at $\displaystyle x$, is the probability that $\displaystyle X$ will take a value less than or equal to $\displaystyle x$.

To do that we will use the `scipy.stats.f` class, representing the __F-distribution__, and providing methods to work with the F-distribution. We will use in particular its method [`cdf`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html).

#### __Pearson's Chi-Squared Test__

To test the distribution of observations in the case of two or more categorical variables, as, for example, between our "Hogwarts House" and "Best Hand" columns, we can use a [__chi-squared test__](https://en.wikipedia.org/wiki/Chi-squared_test) such as [__Pearson's chi-squared test__](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test), also called __Pearson's $\displaystyle \chi ^{2}$ test__, a __*p*-value__ test applied to sets of categorical data to evaluate how likely it is that any observed difference between the sets arose by chance.

To perform it I used the function [chisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html) of the SciPy module `stats`.


### Mini-Batch Gradient Descent Algorithm

https://www.geeksforgeeks.org/ml-mini-batch-gradient-descent-with-python

https://github.com/iamkucuk/Logistic-Regression-With-Mini-Batch-Gradient-Descent/blob/master/logistic_regression_notebook.ipynb

To add mathematical formulas in the present README file I followed the article ["Cheat Sheet: Adding Math Notation to Markdown"](https://www.upyesp.org/posts/makrdown-vscode-math-notation).