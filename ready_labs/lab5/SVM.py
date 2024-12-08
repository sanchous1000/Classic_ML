import numpy as np
from scipy.optimize import minimize

def optimizer(y_train, C, kernel):
    def _optim_func(lam):
        N= lam.shape[0]
        N1 = y_train.shape[0]
        lam1 = lam.reshape(N , 1)
        lam2 = lam.reshape(1,N)
        y_train1 = lam.reshape(N1 , 1)
        y_train2 = lam.reshape( 1,N1)

        return -np.sum(lam) + 0.5 * np.sum(
            lam1 * lam2 * y_train1 * y_train2 * kernel
        )
    min_it = minimize(
        _optim_func, np.zeros(len(y_train)),
        constraints=[
            {'type': 'ineq', 'fun': lambda x: x},  
            {'type': 'ineq', 'fun': lambda x: C - x},  
            {'type': 'eq', 'fun': lambda x: np.dot(x, y_train)}  
        ]
    )
    return min_it.x

class SVC:
    def __init__(self, C):
        self.w0 = None
        self.C = C

    def kernel_calc(self, X, X_):
        if self.kernel_type == 'linear':
            return np.dot(X, X_.T)
        if self.kernel_type == 'rbf':
            X1 = X ** 2
            X2 = X_ ** 2
            X1 = np.sum(X1 , axis = 1).reshape(X1.shape[0], 1)
            X2 = np.sum(X2 , axis = 1).reshape( 1, X2.shape[0])
            return np.exp(-self.gamma * (X1 - 2 * np.dot(X , X_.T) + X2))
        if self.kernel_type == 'polynom':
            return np.power(np.dot(X, X_.T) + 1, self.d)


    def fit(self, X_train, y_train, kernel_type, gamma=None, d=None):
        self.kernel_type = kernel_type
        N = y_train.shape[0]
        self.X = X_train
        self.gamma = gamma
        self.d = d
        kernel = self.kernel_calc(X_train, X_train)
        self.lam = optimizer(y_train, self.C, kernel)
        
        

        indices = self.lam > 0
        self.X_ind = X_train[indices]
        self.Y_ind = y_train[indices]
        self.lam = self.lam[indices]

        # Вычисляем w0
        kernel = kernel[indices][:, indices]
        self.w0 = np.mean(
            self.Y_ind - np.sum(
                self.lam.reshape(self.lam.shape[0] , 1) * self.Y_ind.reshape(self.Y_ind.shape[0] , 1) * kernel, axis=0
            )
        )

    def predict(self, X):
        N, _ = X.shape
        kernel = self.kernel_calc(X, self.X_ind)
        decision = np.sum(
            self.lam.reshape(self.lam.shape[0] , 1) * self.Y_ind.reshape(self.Y_ind.shape[0] , 1) * kernel.T, axis=0
        ) + self.w0
        return np.sign(decision)





'''import numpy as np
from scipy.optimize import minimize

class SVC:
    def __init__(self, C=1.0, kernel="linear", gamma=None, degree=3):
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.lam = None 
        self.support_vectors = None
        self.support_labels = None
        self.w0 = None

    def _kernel(self, X, X_):
        if self.kernel_type == "linear":
            return np.dot(X, X_.T)
        elif self.kernel_type == "rbf":
            if self.gamma is None:
                self.gamma = 1 / X.shape[1]
            pairwise_sq_dists = (
                np.sum(X**2, axis=1)[:, None] +
                np.sum(X_**2, axis=1)[None, :] -
                2 * np.dot(X, X_.T)
            )
            return np.exp(-self.gamma * pairwise_sq_dists)
        elif self.kernel_type == "polynom":
            return (np.dot(X, X_.T) + 1) ** self.degree
       

    def fit(self, X, y):
        n_samples = X.shape[0]
        K = self._kernel(X, X)  

        def objective(lam):
            return -np.sum(lam) + 0.5 * np.sum(
                lam[:, None] * lam[None, :] * y[:, None] * y[None, :] * K
            )

       
        constraints = [
            {"type": "eq", "fun": lambda lam: np.dot(lam, y)},  
            {"type": "ineq", "fun": lambda lam: lam},          
            {"type": "ineq", "fun": lambda lam: self.C - lam}   # λ <= C
        ]

        # Инициализация
        initial_lambda = np.zeros(n_samples)

        # Оптимизация
        result = minimize(objective, initial_lambda, constraints=constraints)

       
        self.lam = result.x

        # Выбираем опорные векторы (λ > 1e-5)
        support_indices = self.lam > 1e-5
        self.support_vectors = X[support_indices]
        self.support_labels = y[support_indices]
        self.lam = self.lam[support_indices]

        # Вычисляем w0
        K_support = K[support_indices][:, support_indices]
        self.w0 = np.mean(
            self.support_labels - np.sum(
                self.lam[:, None] * self.support_labels[:, None] * K_support, axis=0
            )
        )

    def predict(self, X):
        K = self._kernel(X, self.support_vectors)
        decision = np.sum(
            self.lam[:, None] * self.support_labels[:, None] * K.T, axis=0
        ) + self.w0
        return np.sign(decision)
'''