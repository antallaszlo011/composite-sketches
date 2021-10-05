# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./regression_train_data2.csv', index_col=0)
x = df.iloc[:, : 512].to_numpy().transpose()
features = df.iloc[:, 512 :].columns
y = df.iloc[:, 512 :].to_numpy().transpose()

y = y * 2 - 1


class LinearRegression:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    lambda : float
        Effectiveness of orthonormalization process
    eta : float
        Learning rate
    n_iterations : int
        No of passes over the training set
    Attributes
    ----------
    w_ : weights/ after fitting the model
    cost_ : total error of the model after each iteration
    """

    def __init__(self, lambda_=0.25, eta=0.005, n_iterations=5000):
        self.lambda_ = lambda_
        self.eta = eta
        self.n_iterations = n_iterations
        # esetleg be lehet vezetni egy epszilon parametert is, konvergenciaig iteralunk amig > eps

    def fit(self, x, y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_features, n_samples]
            Training samples
        y : array-like, shape = [n_target_values, n_samples]
            Target values
        Returns
        -------
        self : object
        """

        self.w_ = np.zeros((y.shape[0], x.shape[0]))
        self.b_ = np.zeros((y.shape[0]))
        self.scores = []
        m = x.shape[1]

        for k in range(self.n_iterations):
            if k % 500 == 0 and k > 0:
                print('Iteration:', k, '/', self.n_iterations)
                self.plotLearningGraph(it_num=k)
                self.saveWeightsToCSV(it_num=k)

            y_pred = np.matmul(self.w_, x) + self.b_[:, np.newaxis]
            residuals = y_pred - y

            self.b_ -= (self.eta / m) * np.sum(residuals, axis=1)
            self.w_ -= (self.eta) * ((1/m) * np.matmul(residuals, x.transpose()) + 4 * self.lambda_ * (np.matmul(np.matmul(self.w_, self.w_.transpose()), self.w_) - self.w_))


            cost = np.sum((self.predict(x) - y) ** 2, axis=1) / y.shape[1]
            score = sum(cost) / cost.shape[0]
            self.scores.append(score)

        return self, self.w_, self.b_

    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        y_pred = np.sign(np.matmul(self.w_, x) + self.b_[:, np.newaxis])
        return y_pred
        # y_pred[y_pred  > 0.5] = 1
        # y_pred[y_pred <= 0.5] = 0
        # return y_pred

    def plotLearningGraph(self, it_num=0):
        plt.clf()
        plt.plot(self.scores)
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy score')
        # plt.show()
        if it_num > 0:
            plt.savefig('model_accuracy_%d.png' % it_num)
        else:
            plt.savefig('model_accuracy.png')

    def saveWeightsToCSV(self, it_num=0):
        W = self.w_.copy()
        for i in range(40):
            v_i = W[i, :]
            norm = np.sqrt(np.dot(v_i, v_i))
            W[i, :] = W[i, :] / norm

            # print(np.average(np.dot(v_i, v_i)))

        df = pd.DataFrame(W, index=features, columns=range(512))
        # print(df.head())
        if it_num > 0:
            df.to_csv('feature_axis_ortonorm_%d.csv' % it_num)
        else:
            df.to_csv('feature_axis_ortonorm.csv')


lin_reg = LinearRegression()
lin_reg, W, b = lin_reg.fit(x, y)
y_pred = lin_reg.predict(x)
lin_reg.plotLearningGraph()

mse = np.sum((y_pred - y)**2, axis=1) / y.shape[1]

score = sum(mse) / mse.shape[0]

print((y - y_pred) ** 2) 
print(((y - y_pred) ** 2).shape)
print(list(sorted(zip(mse, features))))
print(score)

for i in range(5):
    for j in range(5):
        v_i = W[i, :]
        v_j = W[j, :]
        print('<v%d, v%d> = %f'%(i, j, np.dot(v_i, v_j)))


for i in range(40):
    v_i = W[i, :]
    norm = np.sqrt(np.dot(v_i, v_i))
    W[i, :] = W[i, :] / norm

    print(np.dot(v_i, v_i))

df = pd.DataFrame(W, index=features, columns=range(512))
print(df.head())
df.to_csv('feature_axis_ortonorm.csv')




# y_pred =  np.zeros((40, m))
# for i in range(m):
#     y_i = y[:, i]
#     x_i = x[:, i]
#     y_i_pred = np.matmul(self.w_, x_i) + self.b_
#     y_pred[:, i] = y_i_pred
# y_pred = np.dot(x, self.w_) + self.b_       # y_pred = W * x + b
# residuals = y_pred - y                      # err = y_pred - y
# ortho_reg = np.matmul(np.matmul(self.w_, self.w_.transpose()), self.w_) - self.w_       # (W * W' * W - W)
# self.w_ -= (2 * self.eta / m) * np.dot(x.T, residuals) - 4 * self.lambda_ * ortho_reg   # 
# self.b_ -= (2 * self.eta / m) * np.sum(residuals, axis=0)
# cost = np.sum((residuals ** 2)) / (2 * m)
# self.cost_.append(cost)
