import numpy as np
import matplotlib.pyplot as plt
from hmm import GaussianHMM

np.random.seed(1)
nb = 20
X = np.sin(np.linspace(0, 2 * np.pi, nb))
hmm = GaussianHMM(n_states=20, verbose=1, n_repeats=10, cov_type='diagonal', random_state=2)
hmm.fit(X) # learn the parameters of the model with EM
X_gen = hmm.generate(300)
plt.plot(X, c='blue', label='true', linewidth=3, alpha=0.5)
plt.plot(X_gen, c='red', label='generated', linewidth=3, alpha=0.5)
plt.axvline(nb - 1, c='green', linestyle='dashed', label='extrapolation starting')
plt.legend()
plt.title('Gaussian HMM extrapolation')
plt.show()