A scikit-learn like generic Implementation of Hidden Markov Models as described in :
    "Pattern recognition and machine learning, Bishop".

For now It implements gaussian observations and bernoulli (boolean) observations.

### Installation

```
git clone git://github.com/mehdidc/hiddenmarkovmodel
cd hiddenmarkovmodel
python setup.py install
```


## Example Usage
```python
import numpy as np
from hmm import BernoulliHMM
X = [0] * 20 + [1] * 20 # build a simple sequence of booleans
X = np.array(X)[:, np.newaxis] # make it a matrix of 1 column because it expects a matrix
hmm = BernoulliHMM(n_iters=40, n_states=2, verbose=1)
hmm.fit(X) # learn the parameters of the model with EM
print(hmm.generate(40)[:, 0]) # generate a sequence of 40 booleans from the model
```
