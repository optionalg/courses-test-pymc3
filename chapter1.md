---
title       : Basic Bayesian Inference
description : Insert the chapter description here
attachments :
  slides_link : https://s3.amazonaws.com/assets.datacamp.com/course/teach/slides_example.pdf



--- type:NormalExercise lang:python xp:100 skills:2 key:c85026b09d
## Example 1


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}
from pymc3 import Poisson

x = Poisson.dist(mu=1)
samples = x.random(size=10000)

print(samples.mean())

plt.hist(samples, bins=len(set(samples)))
plt.show()


```

*** =solution
```{python}
from pymc3 import Poisson

x = Poisson.dist(mu=1)
samples = x.random(size=10000)

print(samples.mean())

plt.hist(samples, bins=len(set(samples)))
plt.show()

```

*** =sct
```{python}

```




--- type:NormalExercise lang:python xp:100 skills:2 key:0571398f4c
## Estimation for one group: EDA


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}

```

*** =sample_code
```{python}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

radon = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_5306/datasets/radon.csv', index_col=0)

print(radon.head())

hennepin_radon = radon.query('county=="HENNEPIN"').log_radon
sns.distplot(hennepin_radon)
plt.show()

print(hennepin_radon.shape)

```

*** =solution
```{python}

```

*** =sct
```{python}

```

--- type:NormalExercise lang:python xp:100 skills:2 key:88bcef139d
## Estimation for one group: The model


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

radon = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_5306/datasets/radon.csv', index_col=0)

hennepin_radon = radon.query('county=="HENNEPIN"').log_radon


```

*** =sample_code
```{python}
from pymc3 import Model, Uniform

with Model() as radon_model:
    
    μ = Normal('μ', mu=0, sd=10)
    σ = Uniform('σ', 0, 10)

with radon_model:
    
    y = Normal('y', mu=μ, sd=σ, observed=hennepin_radon)

```

*** =solution
```{python}

```

*** =sct
```{python}

```
