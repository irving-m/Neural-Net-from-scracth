import numpy as np

w_record = [np.array([[10, 11, 12], [13, 14, 25]]), np.array([20, 21, 22]), np.array([30, 31, 32])]
dw_record = [np.array([[1, 1, 1], [1, 1, 1]]), np.array([2, 2, 2]), np.array([3, 3, 3])]
w_opt = []

for w, dw in zip(w_record, dw_record):
    w_opt.append(w - 0.1 * dw)

print(w_opt)

print([w - 0.1 * dw for w, dw in zip(w_record, dw_record)])



print(np.random.rand(4) - 0.5)