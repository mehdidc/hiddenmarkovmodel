A scikit-learn like generic Implementation of Hidden Markov Models as described in :
    "Pattern recognition and machine learning, Bishop".

For now It implements gaussian observations, bernoulli (boolean) and multinomial (categorical) observations.

### Installation

```
git clone git://github.com/mehdidc/hiddenmarkovmodel
cd hiddenmarkovmodel
python setup.py install
```

## Example Usage
```python
import numpy as np
from hmm import MultinomialHMM
hmm = MultinomialHMM(n_states=3, verbose=0, n_repeats=20)
X = ['a'] * 5 + ['b'] * 5 + ['c'] * 5 # build a simple sequence of characters
X = np.array(X)
hmm.fit(X) # learn the parameters of the model with EM
for i in range(3):
	print(''.join(hmm.generate(15))) # generate a sequence of 15 characters from the model
```
