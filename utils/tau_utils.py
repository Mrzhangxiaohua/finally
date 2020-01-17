import numpy as np
from sklearn.svm import OneClassSVM


def tau_to_npy(a, domain='xs'):
    tau_a = np.array([])
    for i in range(a.shape[1]):
        # print(i)
        temp1 = a[:, i].reshape(-1, 1)
        y_pred = OneClassSVM(nu=0.1).fit(temp1).predict(temp1)
        index = np.where(y_pred == 1)[0].tolist()
        length = len(index)
        average = np.sum(temp1[index]) / length
        tau_a = np.append(tau_a, average)
    tau_a = tau_a.reshape(1, -1)
    if domain == 'xs':
        np.save("Xs.npy", tau_a)
    elif domain == 'xt':
        np.save("Xt.npy", tau_a)
    elif domain == 'xs_add':
        np.save("Xs_add.npy", tau_a)
    elif domain == 'xt_add':
        np.save("Xt_add.npy", tau_a)
    else:
        print("data save error")
