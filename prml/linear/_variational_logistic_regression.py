import typing as tp

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from prml.linear._logistic_regression import LogisticRegression


class VariationalLogisticRegression(LogisticRegression, BaseEstimator):
    """Variational logistic regression model.

    Graphical Model
    ---------------

    ```txt
    *----------------*
    |                |               ****  alpha
    |     phi_n      |             **    **
    |       **       |            *        *
    |       **       |            *        *
    |       |        |             **    **
    |       |        |               ****
    |       |        |                |
    |       |        |                |
    |       |        |                |
    |       |        |                |
    |       |        |                |
    |       v        |                v
    |      ****      |               ****  w
    |    **    **    |             **    **
    |   *        *   |            *        *
    |   *        *<--|------------*        *
    |    **    **    |             **    **
    |  t_n ****      |               ****
    |             N  |
    *----------------*
    ```
    """

    def __init__(
            self,
            a0: float = 1.,
            b0: float = 1.,
    ):
        """Construct variational logistic regression model.

        Parameters
        ----------
        alpha : tp.Optional[float]
            precision parameter of the prior
            if None, this is also the subject to estimate
        a0 : float
            a parameter of hyper prior Gamma dist.
            Gamma(alpha|a0,b0)
            if alpha is not None, this argument will be ignored
        b0 : float
            another parameter of hyper prior Gamma dist.
            Gamma(alpha|a0,b0)
            if alpha is not None, this argument will be ignored
        """
        self.a0 = a0
        self.b0 = b0
        self.mapping = list()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def lambdaf(self, eps):
        return (self.sigmoid(eps) - 1 / 2) / (2 * eps)

    def fit(self, x_train: np.ndarray, t: np.ndarray, feature_names: tp.Optional[np.ndarray] = None,
            iter_max: int = 3000):
        """Variational bayesian estimation of the parameter.

        Parameters
        ----------
        x_train : np.ndarray
            training independent variable (N, D)
        t : np.ndarray
            training dependent variable (N,)
        iter_max : int, optional
            maximum number of iteration (the default is 1000)
        """
        self.classes_ = np.unique(t)

        n, d = x_train.shape
        a = self.a0 + 0.5 * d

        def alpha():
            try:
                self.b = self.b0 + 0.5 * (np.sum(self.w_mean ** 2) + np.trace(self.w_var))
            except AttributeError:
                self.b = self.b0
            return a / self.b

        epsilon = np.random.uniform(-1, 1, size=n)
        eye = np.eye(d)
        m0 = np.zeros(d)
        param = np.copy(epsilon)
        for _ in range(iter_max):
            lambda_ = self.lambdaf(epsilon)
            self.w_var = np.linalg.inv(eye * alpha() + 2 * (lambda_ * x_train.T) @ x_train)
            self.w_mean = self.w_var @ (
                    np.linalg.inv(eye * alpha()) @ m0 + np.sum(x_train.T * (t - 0.5), axis=1))

            epsilon = np.sqrt(np.sum(
                (
                        x_train
                        @ (self.w_var + self.w_mean * self.w_mean[:, None])
                        * x_train
                ),
                axis=-1,
            ))
            if np.allclose(epsilon, param):
                try:
                    index = list(np.argsort(-np.abs(self.w_mean)))
                    map = [(feature_names[i], i, f"{round(self.w_mean[i], 4)} ± {round(np.sqrt(self.w_var[i][i]), 4)}")
                           for i in index]
                    print("all positive", len(map), map)
                    self.mapping = map
                except:
                    pass
                break
            else:
                param = np.copy(epsilon)

        if not len(self.mapping):
            try:
                index = list(np.argsort(-np.abs(self.w_mean)))
                map = [(feature_names[i], i, f"{round(self.w_mean[i], 4)} ± {round(np.sqrt(self.w_var[i][i]), 4)}")
                       for i in index]
                print("all positive", len(map), map)
                self.mapping = map
            except:
                pass

    # @property
    # def alpha(self) -> float:
    #     """Return expectation of variational distribution of alpha.
    #
    #     Returns
    #     -------
    #     float
    #         Expectation of variational distribution of alpha.
    #     """
    #     try:
    #         self.b = self.b0 + 0.5 * (np.sum(self.w_mean ** 2) + np.trace(self.w_var))
    #     except AttributeError:
    #         self.b = self.b0
    #     return self.a / self.b

    def proba(self, x: np.ndarray):
        """Return probability of input belonging class 1.

        Parameters
        ----------
        x : np.ndarray
            Input independent variable (N, D)

        Returns
        -------
        np.ndarray
            probability of positive (N,)
        """
        mu_a = x @ self.w_mean
        try:
            var_a = np.sum(x @ self.w_var * x, axis=1)
        except:
            var_a = np.sum(x @ self.w_var * x)
        y = self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
        return y

    def predict(self, x: np.ndarray):
        return self.classify(x)

    def predict_proba(self, x: np.ndarray):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        # 目标仅测试了二分类，未对其他类进行测试
        prob = self.proba(x)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            prob /= prob.sum(axis=1).reshape((prob.shape[0], - 1))
            return prob

    def score(self, X, y, sample_weight=None):
        y_pred = self.classify(X)
        return accuracy_score(y, y_pred)

    def bound(self, mean: float, st: float, n: int, z: float = 1.96) -> tuple:
        # lower_bound, upper_bound
        return mean - (z * st) / np.sqrt(n), mean + (z * st) / np.sqrt(n)

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))
