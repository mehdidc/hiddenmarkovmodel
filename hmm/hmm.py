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

    n_repeats : int, number of repetitions with different random initializations.
        
    random_state : int, seed integer for initialization of HMM parameters (not for generation)

    Attributes
    ----------
    
    pi_ : vector of floats of size n_states.
        the prior probabilities of the first state
    trans_matrix_ : matrix with shape (n_states, n_states) of floats. 
        the resulting transition matrix of HMM where rows describe current states and cols
        next state probabilities.

    """

    def __init__(self, n_states=10, n_iters=100, verbose=0, tol=0.001, n_repeats=10, random_state=None):        
        self.K = n_states
        self.n_iters = n_iters
        self.verbose = verbose
        self.tol = tol
        self.rng = np.random.RandomState(random_state)
        self.n_repeats = n_repeats

        self.pi_ = None
        self.trans_matrix_ = None

        self._params_list = ['pi_', 'trans_matrix_']

    def _init_from_data(self, X):
        """
        Init parameters from the data.

        this can be overloaded to do some desired initialization.
        """
        self._X_type = X.dtype
        self.pi_ = np.array([1./ self.K] * self.K)
        self.trans_matrix_ = self.rng.uniform(size=(self.K, self.K)) # rows = current state, cols = new state probas
        self.trans_matrix_ /= self.trans_matrix_.sum(axis=1, keepdims=True) # normalize rows

    def _get_params(self):
        return {p: getattr(self, p).copy() for p in self._params_list}

    def _set_params(self, params):
        for k, v in params.items():
            assert k in self._params_list
            setattr(self, k, v)

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
        X = np.empty((seq_length, self._nb_feats), dtype=self._X_type)
        rng = np.random.RandomState(random_state)
        z = rng.choice(range(self.K), p=self.pi_)
        for i in range(seq_length):
            X[i, :] = self._generate_one_step(z, rng)
            z = rng.choice(range(self.K), p=self.trans_matrix_[z])
        return X

    def _generate_one_step(self, z, rng):
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

        p_x_z = self._p_x_given_z(X)  # shape (N, K)
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
        p_x_z = self._p_x_given_z(X)
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
        p_x_z = self._p_x_given_z(X)
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

    def _p_x_given_z(self, X):
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
        ll = -np.inf
        best_params = None
        for i in range(self.n_repeats):
            ll_new = self._fit_once(X)
            if ll_new > ll:
                ll = ll_new
                best_params = self._get_params()
        if best_params is None:
            print('HMM failed, log-likelihood is {}'.format(ll))
            return self
        self._set_params(best_params)
        if self.verbose:
            print('Best log-likelihood of all repetitions : {}'.format(ll))
        return self

    def _fit_once(self, X):
        self._init_from_data(X)
        prev_ll = None
        ll = None
        self._nb_feats = X.shape[1]
        for i in range(self.n_iters):
            # E-step
            gammas, eps = self._e_step(X)
            # M-step
            self._m_step_latent(X, gammas, eps)
            self._m_step_conditionals(X, gammas, eps)
            # evaluate
            prev_ll = ll
            ll = self.loglikelihood(X)
            if prev_ll is not None:
                change = np.abs(prev_ll - ll)
                if change <= self.tol:
                    if self.verbose:
                        print('Delta log-likelihood reached the tolerance at iter {}, stop.'.format(i + 1))
                    break
            if self.verbose:
                print('log-likelihood at iter {} : {}'.format(i, ll))
        return ll

    def _e_step(self, X):
        """
        Perform the E-step.
        """
        gammas = self._get_gammas(X)
        eps = self._get_epsilons(X)
        return gammas, eps

    def _m_step_latent(self, X, gammas, eps):
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
           

    def _m_step_conditionals(self, X, gammas, eps):
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
        self.mu_ = None
        self._cov = None

        self._params_list.append('mu_')
        self._params_list.append('cov_')


    def _init_from_data(self, X):
        super(GaussianHMM, self)._init_from_data(X)
        self.mu_ = self.rng.uniform(size=(self.K, X.shape[1]))
        self.cov_ = np.repeat(np.eye(X.shape[1])[np.newaxis, :, :], self.K, axis=0)

    def _m_step_conditionals(self, X, gammas, eps):
        # eq  13.20 : 
        x = X[:, np.newaxis, :] # shape : (N, K, F)
        w = gammas[:, :, np.newaxis] # shape : (N, K, F)
        self.mu_ = (x*w).sum(axis=0) / w.sum(axis=0)
        # eq  13.21 :
        cov = np.zeros((self.K, X.shape[1], X.shape[1]))
        for k in range(self.K):
            cov[k] = np.cov(weighted_X[:, k, :].T)
        self.cov_ = cov # shape (K, F, F)
        

    def _p_x_given_z(self, X):
        p = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            densities = multivariate_normal(self.mu_[k], self._cov[k]).pdf(X)
            p[:, k] = densities
        return p


class BernoulliHMM(HMM):
    """
    Bernoulli version of HMM.

    Observations can only have values of 0 and 1.

    """

    def __init__(self, **kwargs):
        super(BernoulliHMM, self).__init__(**kwargs)
        self.mu_ = None
        self._params_list.append('mu_')

    def _init_from_data(self, X):
        super(BernoulliHMM, self)._init_from_data(X)
        self.mu_ = np.ones((self.K, X.shape[1])) * 0.5

    def _m_step_conditionals(self, X, gammas, eps):
        # eq 13.20
        gammas = gammas / gammas.sum(axis=1, keepdims=True)
        x = X[:, np.newaxis, :]  # shape : (N, K, F)
        g = gammas[:, :, np.newaxis]  # shape : (N, K, F)
        self.mu_ = (x*g).sum(axis=0) / g.sum(axis=0) # shape : (K, F)

        if self.verbose > 1:
            mu_check = np.zeros_like(self.mu_)
            for k in range(self.K):
                s = sum(gammas[x, k] for x in range(X.shape[0]))
                for f in range(X.shape[1]):
                    mu_check[k, f] = sum(gammas[x, k] * X[x, f] for x in range(X.shape[0])) / s
            assert close_to(self.mu_, mu_check)

    def _p_x_given_z(self, X):
        p = self.mu_[np.newaxis, :, :]
        x = X[:, np.newaxis, :]
        return (p ** x * (1 - p) ** (1 - x)).prod(axis=2)

    def _generate_one_step(self, z, rng):
        return rng.uniform(size=self.mu_.shape[1]) <= self.mu_[z]


class MultinomialHMM(HMM):
    """
    Multinomial version of HMM.

    Observations can be integers or even strings.
    """

    def __init__(self, **kwargs):
        super(MultinomialHMM, self).__init__(**kwargs)
        self.mu_ = None
        self._cats = None
        self._int2cat = None
        self._params_list.append('mu_')


    def _init_from_data(self, X):
        super(MultinomialHMM, self)._init_from_data(X)
        self._cats = list(set(X.flatten()))
        self._cats_int = range(len(self._cats))
        self._cat2int = {c: i for i, c in enumerate(self._cats)}
        self._int2cat = {i: c for i, c in enumerate(self._cats)}
        self.mu_ = np.ones((len(self._cats), self.K, X.shape[1]))
        self.mu_ /= self.mu_.sum(axis=1, keepdims=True)

    def _to_int(self, X):
        X_int = np.empty((X.shape[0], X.shape[1]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_int[i, j] = self._cat2int[X[i, j]]
        return X_int

    def _m_step_conditionals(self, X, gammas, eps):
        gammas = gammas / gammas.sum(axis=1, keepdims=True)
        mu = np.empty((len(self._cats), self.K, X.shape[1])) # shape : (C, K, F)
        X = self._to_int(X)
        # TODO : vectorize
        for cat in self._cats_int:
            for f in range(X.shape[1]):
                x_ = X[:, f]
                ex = (x_==cat)[:, np.newaxis]
                g_ = gammas
                mu[cat, :, f] = (ex * g_).sum(axis=0) / g_.sum(axis=0)
        self.mu_ = mu

    def _p_x_given_z(self, X):
        mu = self.mu_# (C, K, F)
        X = self._to_int(X) # (N, F)
        # TODO : vectorize
        p = np.ones((X.shape[0], self.K))
        for n in range(X.shape[0]):
            for f in range(X.shape[1]):
                for k in range(self.K):
                    p[n, k] *= mu[X[n, f], k, f]
        return p

    def _generate_one_step(self, z, rng):
        p = self.mu_[:, z, :]
        sample = np.empty((p.shape[1],), dtype=self._X_type)
        for f in range(p.shape[1]):
            sample[f] = rng.choice(self._cats, p=p[:, f])
        return sample

if __name__ == '__main__':
    hmm = MultinomialHMM(n_states=3, verbose=0, n_repeats=50)
    X = ['a'] * 5 + ['b'] * 5 + ['c'] * 5
    X = np.array(X)[:, np.newaxis]
    hmm.fit(X)
    print(''.join(hmm.generate(15)[:, 0]))
    print(''.join(hmm.generate(15)[:, 0]))
    print(''.join(hmm.generate(15)[:, 0]))