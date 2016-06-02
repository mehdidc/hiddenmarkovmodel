import numpy as np
from scipy.stats import multivariate_normal


def close_to(a, b):
    return np.abs(a - b).max() <= 1e-8


class HMM(object):
    """
    A generic Implementation of Hidden Markov Models as described in :
      "Pattern recognition and machine learning, Bishop".

    This class leaves the calculation of the conditionals p(x|z) and 
    the M-step (updates) of the conditionals left to the user to be
    implemented in a class which inherits HMM.

    the equations in the comments refer to the book  : 
        "Pattern recognition and machine learning, Bishop".
    
    Parameters
    ----------

    n_states : int, nb of HMM states

    n_iters : int, max number of iterations
        
    verbose : int, 0 (no verbose print), 1 (show log-likelihood in each iteration), 2 (debug mode)
        
    tol : float, stop when difference between prev log-likelihood and current is smaller than tol
        
    random_state : int, seed integer for initialization of HMM parameters (not for generation)

    Attributes
    ----------
    
    pi_ : vector of floats of size n_states.
        the prior probabilities of the first state
    trans_matrix_ : matrix with shape (n_states, n_states) of floats. 
        the resulting transition matrix of HMM where rows describe current states and cols
        next state probabilities.

    """

    def __init__(self, n_states=10, n_iters=100, verbose=0, tol=0.001, random_state=None):
        self.K = n_states
        self.n_iters = n_iters
        self.verbose = verbose
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

        self.pi_ = np.array([1./ self.K] * self.K)
        self.trans_matrix_ = self.rng.uniform(size=(self.K, self.K)) # rows = current state, cols = new state probas
        self.trans_matrix_ /= self.trans_matrix_.sum(axis=1, keepdims=True) # normalize rows


    def generate(self, seq_length=10, random_state=None):
        """
        Generate samples according to the model.

        Parameters
        ----------

        seq_length : int, length of the sequence to generate
        random_state : int, seed integer for the randon number generator

        Returns
        -------
        
        samples : matrix of shape (seq_length, nb_features) where nb_features
            are the number of features of observed data.
        """
        X = np.zeros((seq_length, self._nb_feats))
        rng = np.random.RandomState(random_state)
        z = rng.choice(range(self.K), p=self.pi_)
        for i in range(1, seq_length):
            X[i, :] = self.generate_one_step(z, rng)
            z = rng.choice(range(self.K), p=self.trans_matrix_[z])
        return X

    def generate_one_step(self, z, rng):
        """
        Generate observations based on hidden states.

        Parameters
        ----------

        z : int, hidden state component index
        rng : random number generator to use

        Returns
        -------

        sample : vector of size nb_features where nb_features are
            the number of features of observed data
        """
        raise NotImplementedError()

    def _get_alphas(self, X):
        if self.verbose > 1:
            assert close_to(self.pi_.sum(), 1)

        p_x_z = self.p_x_given_z(X)  # shape (N, K)
        Z = self.trans_matrix_  # shape (K, K)
        alphas = np.zeros((X.shape[0], self.K))
        alphas[0] = (self.pi_ * p_x_z[0])  # eq 13.37
        if self.verbose > 1:
            alphas_check = alphas.copy()
        n = len(X)
        for i in range(1, n):
            alphas[i] = (p_x_z[i] * (alphas[i - 1][:, np.newaxis] * Z).sum(axis=0))  # eq 13.36
            #alphas[i] /= alphas[i].sum()
            #c = alphas[i].sum()
            #alphas[i] *= c 
            if self.verbose > 1:
                for k in range(self.K):
                    alphas_check[i, k] = p_x_z[i, k] * sum(alphas_check[i - 1, kp] * Z[kp, k] for kp in range(self.K))
                #c = alphas_check[i].sum()
                #alphas_check[i] *= c
        if self.verbose > 1:
            assert close_to(alphas, alphas_check)
            assert np.any(alphas[0]) > 0, alphas[0]
        return alphas

    def _get_betas(self, X):
        p_x_z = self.p_x_given_z(X)
        Z = self.trans_matrix_
        betas = np.zeros((X.shape[0], self.K))
        betas[-1] = 1  # eq just before 13.39
        n = len(X)

        if self.verbose > 1:
            betas_check = betas.copy()
        for i in range(n - 2, -1, -1):
            betas[i] = (betas[i + 1] * p_x_z[i + 1] * Z).sum(axis=1)  # eq 13.38
            if self.verbose > 1:
                for k in range(self.K):
                    betas_check[i, k] = sum(betas_check[i + 1, kp] * p_x_z[i + 1, kp] * Z[k, kp] for kp in range(self.K))
        if self.verbose > 1:
            assert close_to(betas, betas_check)
        assert np.any(betas[0]) > 0, betas[0]
        return betas

    def _get_gammas(self, X):  # used in E-step
        # eq 13.33
        alphas = self._get_alphas(X)
        betas = self._get_betas(X)
        return alphas * betas

    def _get_epsilons(self, X):  # used in E-step
        alphas = self._get_alphas(X)
        betas = self._get_betas(X)
        p_x_z = self.p_x_given_z(X)
        Z = self.trans_matrix_
        # eq 13.43
        px = alphas[-1, :].sum()
        eps = (alphas[0:-1, :, np.newaxis] *  # shape (N - 1, K, 1)
               p_x_z[1:, np.newaxis, :] *  # shape (N - 1, K, 1)
               Z[np.newaxis, :, :] *  # shape (1, K, K)
               betas[1:, np.newaxis, :]) / px # shape (N - 1, K, 1)
        N = len(X)

        if self.verbose > 1:
            eps_check = np.zeros((N - 1, self.K, self.K))
            for i in range(1, N):
                #d = sum(alphas[i - 1, kp] * p_x_z[i, k] * Z[kp, k] * betas[i, k] for k in range(self.K) for kp in range(self.K))
                for k in range(self.K):
                    for kp in range(self.K):
                        eps_check[i - 1, kp, k] = alphas[i - 1, kp] * p_x_z[i, k] * Z[kp, k] * betas[i, k] / px
            assert close_to(eps, eps_check)
        return eps

    def p_x_given_z(self, X):
        """
        Return the conditional probabilities p(X|z) for all possible z.

        Parameters
        ----------

        X : matrix of shape (seq_length, nb_features).
            Observed data.

        Returns
        -------

        probas : matrix of shape (seq_length, n_states).
        """
        raise NotImplementedError()

    def _p_z_given_prev_z(self):
        return self.trans_matrix_

    def fit(self, X):
        """
        Fit an HMM model.

        Parameters
        ----------

        X : matrix of shape (seq_length, nb_features).
            Observed data.

        """
        prev_ll = None
        ll = None
        self._nb_feats = X.shape[1]
        for i in range(self.n_iters):
            # E-step
            gammas, eps = self.e_step(X)
            # M-step
            self.m_step_latent(X, gammas, eps)
            self.m_step_conditionals(X, gammas, eps)
            # evaluate
            prev_ll = ll
            ll = self.loglikelihood(X)
            if prev_ll is not None:
                change = np.abs(prev_ll - ll)
                if change <= self.tol:
                    if self.verbose:
                        print('Delta log-likelihood reached the tolerance, stop.')
                    return
            if self.verbose:
                print('log-likelihood : {}'.format(ll))

    def e_step(self, X):
        """
        Perform the E-step.
        """
        gammas = self._get_gammas(X)
        eps = self._get_epsilons(X)
        return gammas, eps

    def m_step_latent(self, X, gammas, eps):
        """
        Performs the M-step to update the transition matrix and the prior
        probabilities of the first state.
        """
        if self.verbose > 1:
            assert close_to(self.pi_.sum(), 1)
            assert gammas[0].sum() > 0
        # eq 13.18
        newpi_ = gammas[0] / gammas[0].sum()
        self.pi_ = newpi_
        # eq 13.19
        new_Z = eps.sum(axis=0)
        new_Z /= new_Z.sum(axis=1, keepdims=True)

        if self.verbose > 1:
            new_Z_check = np.zeros_like(new_Z)
            for j in range(self.K):
                s = eps[:, j, :].sum()
                for k in range(self.K):
                    new_Z_check[j, k] = eps[:, j, k].sum() / s
            assert close_to(new_Z, new_Z_check)
            assert close_to(self.pi_.sum(), 1)
            assert close_to(self.trans_matrix_.sum(axis=1), np.ones((self.K)))
        self.trans_matrix_ = new_Z
           

    def m_step_conditionals(self, X, gammas, eps):
        """
        Perform the M-step to update the parameters of the conditionals
        """
        raise NotImplementedError()

    def loglikelihood(self, X):
        """
        Return the log-likelihood of data
        """
        # eq 13.42
        alphas = self._get_alphas(X)
        return np.log(alphas[-1, :].sum())


class GaussianHMM(HMM):

    def __init__(self, **kwargs):
        super(GaussianHMM, self).__init__(**kwargs)
        self._mu = None
        self._cov = None

    def m_step_conditionals(self, X, gammas, eps):
        # eq  13.20 : 
        x = X[:, np.newaxis, :] # shape : (N, K, F)
        w = gammas[:, :, np.newaxis] # shape : (N, K, F)
        self._mu = (x*w).sum(axis=0) / w.sum(axis=0)
        # eq  13.21 :
        cov = np.zeros((self.K, X.shape[1], X.shape[1]))
        for k in range(self.K):
            cov[k] = np.cov(weighted_X[:, k, :].T)
        self._cov = cov # shape (K, F, F)
        

    def p_x_given_z(self, X):
        if self._mu is None:
            self._mu = self.rng.uniform(size=(self.K, X.shape[1]))
        if self._cov is None:
            self._cov = np.repeat(np.eye(X.shape[1])[np.newaxis, :, :], self.K, axis=0)
        p = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            densities = multivariate_normal(self._mu[k], self._cov[k]).pdf(X)
            p[:, k] = densities
        return p


class BernoulliHMM(HMM):

    def __init__(self, **kwargs):
        super(BernoulliHMM, self).__init__(**kwargs)
        self._mu = None

    def m_step_conditionals(self, X, gammas, eps):
        # eq 13.20
        gammas = gammas / gammas.sum(axis=1, keepdims=True)
        x = X[:, np.newaxis, :]  # shape : (N, K, F)
        l = gammas[:, :, np.newaxis]  # shape : (N, K, F)
        self._mu = (x*l).sum(axis=0) / l.sum(axis=0) # shape : (K, F)

        if self.verbose > 1:
            mu_check = np.zeros_like(self._mu)
            for k in range(self.K):
                s = sum(gammas[x, k] for x in range(X.shape[0]))
                for f in range(X.shape[1]):
                    mu_check[k, f] = sum(gammas[x, k] * X[x, f] for x in range(X.shape[0])) / s
            assert close_to(self._mu, mu_check)

    def p_x_given_z(self, X):
        if self._mu is None:
            self._mu = np.ones((self.K, X.shape[1])) * 0.5
        p = self._mu[np.newaxis, :, :]
        x = X[:, np.newaxis, :]
        return (p ** x * (1 - p) ** (1 - x)).sum(axis=2)

    def generate_one_step(self, z, rng):
        return rng.uniform(size=self._mu.shape[1]) <= self._mu[z]

if __name__ == '__main__':
    hmm = BernoulliHMM(n_iters=40, n_states=2, verbose=1)
    X = [0] * 20 + [1] * 20
    X = np.array(X)[:, np.newaxis]
    hmm.fit(X)
    print(hmm.trans_matrix_)
    print(hmm._mu)
    print(hmm.generate(40)[:, 0])
    print(hmm.generate(40)[:, 0])
    print(hmm.generate(40)[:, 0])
    print(hmm.generate(40)[:, 0])
