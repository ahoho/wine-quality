* 1. Superhuman
    * Scalable
    * Early detection in manufacuturing process
    * Potentially more consistent -- (perhaps mention tightness of CIs?)
        * One expert could vary on the same wine
    * Not necessarily susceptible to cultural biases, etc.
    * Won't get drunk, can work at all times of day
* 2. What makes a good wine?
    * Use exploratory data analysis
    * Correlation plots btw quality and different variables
    * Use linear regression -- Will-Cox "percentage change in this variable leads to one variable"
    * Corroborate the linear regression results with the feature importances
    * Consider acknowledging interactions between the components, but mention that analyses are focusing on each individually (ceteris paribus)
* 3. Superwine
    * Use linear regression equation, report the linear combination that can do that
        * "Solve" for the regression equation, giving an example
        * Use existing bounds on the data, find a solution
        * If we don't hit "10", we say that you could, in theory, increase these variables to get a solution, but there's no reason to think that the relationships are linear -- bad idea
        *  Could be interactions
    * Caveat with: there is no reason to believe that relationship is linear, etc.
    * SVM superwine: 
      array([[ 1.81772591,  0.10092426,  0.0390194 ,  2.53233584,  0.05755234,
               4.42206705,  2.57005439,  0.99872838,  3.10172859,  0.60026195,
               9.27585392,  0.27978731]])
    * Linear superwine:
    array([[  2.80940270e+00,   7.69610411e-02,   6.93147181e-01,
        3.13549422e+00,   1.19285709e-02,   4.93806460e+00,
        1.94591015e+00,   9.87420000e-01,   3.80000000e+00,
        1.08180517e+00,   1.40000000e+01,   1.00000000e+00]])
    * Plot SVM superwine, linear superwine
    * It's a bit of a "faff"!
    
* 4. Human perception
    * In theory, tasting is blind avoiding _some_ of the proposed issues, but ultimately we are measuring and trying to approximate _expert_ opinion, rather than the opinion "of the masses", maketability, profitability, sellability -- these are other ways to measure wine that are objective ("amount sold"), etc., etc.
    * But, given that caveat, we are able to match expert opinion with relatively low error
    * Again, we lack knowledge of how different experts were from one another -- if you had a sense of their distributions, you could re-fit the data on augmented, noise-added data
    * Future work: could bring in non-experts, do a randomized controlled trial, etc, double-blind, look at sales figures. (Who is the customer and audience for this app? -- the AI's taste will be 'expert', could do multiple models with multiple 'experts' -- they'll all be biased in some way, but at the least you want to understand what you're model's trying to do. Understand the product)
* Big caveat: learning _expert_ preferences, rather than that of general public, etc.
* In report, have section on further work
    * With additional data, 
        * could make better assessments about human variabiltiy
        * could be more predictive (say, yearly rainfall)


### 
```python
# wine-simulator 3000
In [30]: i = 0
    ...: while pred < 9.0:
    ...:     try:
    ...:         new_wine = np.array([np.random.uniform(*s) for s in spread]).reshape(1, -1)
    ...:         new_wine[-1] = np.random.choice((0,1))
    ...:         pred = svm.predict(new_wine)
    ...:     except KeyboardInterrupt:
    ...:         raise KeyboardInterrupt
    ...:     i += 1
    ...:     print('iter {}, pred {}'.format(i, pred), end='\r')
```