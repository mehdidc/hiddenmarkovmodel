import numpy as np

from hmm import MultinomialHMM

np.random.seed(1)
hmm = MultinomialHMM(n_states=5, verbose=0, n_repeats=20)
X = 'aa bb aaab bb aaaa bbb aaaaa bbbbb aaaaaaaaa bbbbbbb'
X = map(lambda c:c, X)
X = np.array(X)
hmm.fit(X) # learn the parameters of the model with EM
for i in range(3):
	print(''.join(hmm.generate(100))) # generate a sequence of 15 characters from the model