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
import matplotlib.pyplot as plt

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
from pymc3 import Normal, Model, Uniform

with Model() as radon_model:
    
    μ = Normal('μ', mu=0, sd=10)
    σ = Uniform('σ', 0, 10)

with radon_model:
    
    y = Normal('y', mu=μ, sd=σ, observed=hennepin_radon)

```

*** =solution
```{python}
from pymc3 import Normal, Model, Uniform

with Model() as radon_model:
    
    μ = Normal('μ', mu=0, sd=10)
    σ = Uniform('σ', 0, 10)

with radon_model:
    
    y = Normal('y', mu=μ, sd=σ, observed=hennepin_radon)

```

*** =sct
```{python}

```

--- type:NormalExercise lang:python xp:100 skills:2 key:23862bbfe8
## Fitting the model

Pre exercise code, for context:

```python
import pandas as pd

radon = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_5306/datasets/radon.csv', index_col=0)

hennepin_radon = radon.query('county=="HENNEPIN"').log_radon

from pymc3 import Normal, Model, Uniform

with Model() as radon_model:
    
    μ = Normal('μ', mu=0, sd=10)
    σ = Uniform('σ', 0, 10)

with radon_model:
    
    y = Normal('y', mu=μ, sd=σ, observed=hennepin_radon)
```

*** =instructions

*** =hint

*** =pre_exercise_code
```{python}
import pandas as pd

radon = pd.read_csv('https://s3.amazonaws.com/assets.datacamp.com/production/course_5306/datasets/radon.csv', index_col=0)

hennepin_radon = radon.query('county=="HENNEPIN"').log_radon

from pymc3 import Normal, Model, Uniform

with Model() as radon_model:
    
    μ = Normal('μ', mu=0, sd=10)
    σ = Uniform('σ', 0, 10)

with radon_model:
    
    y = Normal('y', mu=μ, sd=σ, observed=hennepin_radon)


```

*** =sample_code
```{python}
from pymc3 import fit

with radon_model:

    samples = fit(random_seed=42).sample(10)
```

*** =solution
```{python}

```

*** =sct
```{python}

```

--- type:NormalExercise lang:python xp:100 skills:2 key:ccce1a1b67
## Getting started w/ PyMC3


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}
import numpy as np
import matplotlib.pyplot as plt

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

```

*** =sample_code
```{python}
import pymc3 as pm

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)


```

*** =solution
```{python}

```

*** =sct
```{python}

```

--- type:NormalExercise lang:python xp:100 skills:2 key:dd62e9fca6
## Maximum a posteriori methods

Pre-exercise code, for context:

```python
import numpy as np
import matplotlib.pyplot as plt

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

import pymc3 as pm

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)
```


*** =instructions

*** =hint

*** =pre_exercise_code
```{python}
import numpy as np
import matplotlib.pyplot as plt

# Initialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

import pymc3 as pm

basic_model = pm.Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

```

*** =sample_code
```{python}
map_estimate = pm.find_MAP(model=basic_model)

```

*** =solution
```{python}

```

*** =sct
```{python}

```
