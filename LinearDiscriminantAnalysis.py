""" Linear Discriminant Analysis
"""

# Author: René Chenard <rene.chenard.1@ulaval.ca>


import random
import numpy as np
from numpy.random import randn
from numpy.linalg import pinv, eig
from scipy.linalg import orth
from sklearn.mixture import GaussianMixture


class LinearDiscriminantAnalysis(object):
    """Linear Discriminant Analysis

    A basis transformation tool using Linear discriminant analysis (LDA) in
    order to find a more informative representation.

    This tool finds the projection that maximizes the between class scatter to
    within class scatter ratio in order to maximize class separability along
    the first axes.

    If no labels are provided, Gaussian Mixture Model (GMM) is used in order to
    identify clusters that are more likely to belong to different classes.

    If n_components is specified, this tool can be used as a dimensionality
    reduction tool.

    The transformation matrix produced is a full rank matrix and is, therefore,
    invertible. In other words, this is an invertible LDA transformation. If no
    dimensionality reduction was applied to the transformed data, the original
    data should be recoverable by doing an inverse transformation.

    Parameters
    ----------
    n_components : int, default=None
        Number of components (<= n_features) for dimensionality reduction.
        If None, will be set to n_features. This parameter only affects the
        `transform` method.

    n_classes : int, default=None
        Number of classes (< 1) contained in the dataset. If no labels are
        given, unsupervised clustering via GMM will be used with this amount of
        clusters. If None, Akaike Information Criterion (AIC) is used to find
        the optimal amount of clusters within a specified range.

    eps : float, default=0.01
        Absolute threshold (<= 0) for a singular value of X to be considered
        significant. Eigenvectors whose singular values are non-significant are
        discarded and the rank of the transformation matrix is filled with
        non collinear randomly generated vectors.

    random_state : int, RandomState instance, default=None
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    mu_ : array, shape (1, n_features)
        Centroid of the whole dataset. Essentially, the mean of all samples
        along each feature.

    mu_c_ : array, shape (n_classes, n_features)
        Centroid of every classes (or clusters, when unsupervised).

    S_within_ : array, shape (n_features, n_features)
        Within-class scatter matrix.

    S_between_ : array, shape (n_features, n_features)
        Between-class scatter matrix.

    W_ : array, shape (n_features, n_features)
        Transformation matrix.

    W_inverse_ : array, shape (n_features, n_features)
        Inverse transformation matrix.

    eig_pairs_ : tuple list
        Pairs of eigenvalues and eigenvectors from the linear discriminant
        analysis.

    """

    def __init__(self, n_components=None, n_classes=None,
                 eps=0.01, random_state=None):
        # Parameters:
        self.n_components = n_components
        self.n_classes = n_classes
        self.eps = eps
        self.random_state = random_state

        # Attributes:
        self.mu_ = None
        self.mu_c_ = None
        self.S_within_ = None
        self.S_between_ = None
        self.W_ = None
        self.W_inverse_ = None
        self.eig_pairs_ = None

        # Random seed:
        np.random.seed(random_state)
        random.seed(random_state)

    def fit(self, X, y=None, min_clusters=2, max_clusters=40, verbose=False):
        assert 0 < min_clusters <= max_clusters < X.shape[0] * X.shape[1]
        assert 0 <= self.eps

        if y is None:
            y = self._clusters(X=X, min_clusters=min_clusters,
                               max_clusters=max_clusters, verbose=verbose)
        else:
            if verbose:
                print("Targets were provided: using the labeled data.\n")

        self.mu_, self.mu_c_ = self._means(X=X, y=y, verbose=verbose)

        N_c, self.S_within_ = self._scatter_within(X=X, y=y, mu_c=self.mu_c_,
                                                   verbose=verbose)

        self.S_between_ = self._scatter_between(mu=self.mu_, mu_c=self.mu_c_,
                                                N_c=N_c, verbose=verbose)

        self.eig_pairs_ = self._eig_pairs(S_within=self.S_within_,
                                          S_between=self.S_between_)

        P = self._filter_eig_pairs(eig_pairs=self.eig_pairs_, eps=self.eps,
                                   verbose=verbose)

        self.W_ = self._fill_rank(P=P, verbose=verbose)

    def transform(self, X):
        if self.n_components is not None:
            assert 0 < self.n_components < X.shape[1]
            W = self.W_[:, :self.n_components]
        else:
            W = self.W_

        return X @ W

    def fit_transform(self, X, y=None, min_clusters=2, max_clusters=40,
                      verbose=False):
        self.fit(X=X, y=y, min_clusters=min_clusters,
                 max_clusters=max_clusters, verbose=verbose)
        return self.transform(X)

    def inverse_transform(self, X, verbose=False):
        if self.W_inverse_ is None:
            self.W_inverse_ = pinv(self.W_)
        if X.shape[1] != self.W_.shape[1]:
            if verbose:
                print(f"Reverse tranformation after dimensionality reduction "
                      f"may yield unexpected results: "
                      f"{X.shape[1]} -> {self.W_.shape[1]}")

            X_ = np.repeat(self.mu_.reshape(1, -1), X.shape[0], 0)
            X_[:, :X.shape[1]] = X[:, :X.shape[1]]
            X = X_
        return X @ self.W_inverse_

    def _clusters(self, X, min_clusters, max_clusters, verbose=False):
        if verbose:
            print("No target is provided: using unsupervised clustering.\n")

        if self.n_classes is None:
            if verbose:
                print(f"Searching for an optimal number of clusters "
                      f"between {min_clusters} and {max_clusters}...\n")

            range_n = np.arange(min_clusters, max_clusters + 1)
            models = [GaussianMixture(n, covariance_type='full',
                                      random_state=self.random_state).fit(
                X)
                for n in range_n]
            index = np.argmin(np.array([m.aic(X) for m in models]))
            model = models[index]
            self.n_classes = index + min_clusters
            print(f"Optimal number of clusters found: "
                  f"{self.n_classes}\n")
        else:
            assert 0 < self.n_classes < X.shape[0] * X.shape[1]

            if verbose:
                print(f"Using the provided number of classes "
                      f"({self.n_classes}) for unsupervised clustering.\n")
            model = GaussianMixture(self.n_classes, covariance_type='full',
                                    random_state=self.random_state).fit(X)

        if verbose:
            print("Predicting the classes from the clusters...\n")
        return model.predict(X)

    @staticmethod
    def _means(X, y, verbose=False):
        mu = np.mean(X, axis=0).reshape(-1, 1)
        if verbose:
            print(f"Mu:\n{mu.T}\n")

        mu_c = np.zeros((X.shape[1], len(np.unique(y))))
        for i, target in enumerate(np.unique(y)):
            mu_c[:, i] = np.mean(X[y == target], axis=0)
            if verbose:
                print(f"Mu_c[{i}]:\n{mu_c[:, i]}\n")

        return mu, mu_c

    @staticmethod
    def _scatter_within(X, y, mu_c, verbose=False):
        data = []
        N_c = np.zeros(len(np.unique(y)))
        for i, target in enumerate(np.unique(y)):
            delta = X[y == target].T - mu_c[:, i].reshape(-1, 1)
            data.append(delta @ delta.T)
            N_c[i] = np.sum(y == target)

        S_within = np.sum(data, axis=0)
        if verbose:
            print(f"S_intra:\n{S_within}\n")

        return N_c, S_within

    @staticmethod
    def _scatter_between(mu, mu_c, N_c, verbose=False):
        delta = np.array(mu_c - mu)
        S_between = N_c * delta @ delta.T
        if verbose:
            print(f"S_inter:\n{S_between}\n")

        return S_between

    @staticmethod
    def _eig_pairs(S_within, S_between):
        A = pinv(S_within) @ S_between
        eig_val, eig_vec = eig(A)
        eig_val = np.abs(eig_val)

        return sorted(zip(eig_val, eig_vec.T), key=lambda k: k[0],
                      reverse=True)

    @staticmethod
    def _filter_eig_pairs(eig_pairs, eps, verbose=False):
        eig_vals, eig_vecs = zip(*eig_pairs)
        total = sum(eig_vals)
        eigenvectors = []

        if verbose:
            print("Singular values:")
        for i, v in enumerate(eig_pairs):
            if v[0] / total >= eps / 100:
                verdict = 'Accepted'
                eigenvectors.append(v[1])
            else:
                verdict = 'Rejected'
            if verbose:
                percentage = v[0] / total
                eigenvalue = v[0]
                print(f"Singular value {i + 1:}: "
                      f"\t{percentage:<8.2%} "
                      f"\t{eigenvalue:<8.6} \t {verdict}")
        if verbose:
            print()

        return np.vstack(eigenvectors).T.real

    @staticmethod
    def _fill_rank(P, verbose=False):
        n_features = P.shape[0]

        while True:
            W = randn(n_features, n_features)
            W = orth(W)
            W[:P.shape[0], :P.shape[1]] = P
            if orth(W).shape == (n_features, n_features):
                break

        if verbose:
            print(f"W:\n{W}\n")

        return W
