from sklearn.metrics import mean_absolute_error 
import numpy as np

"""predict -> Mean absolute error"""

a = np.array([2, 4, 4, 5])
b = np.array([1, 2, 3, 4])

mae = mean_absolute_error(a, b)
print("computer predict: ", mae)
mp = 1/len(a)*sum((abs(a - b)))
print("my predict: ", mp)

# >>> computer predict:  1.25
# >>> my predict:  1.25

from sklearn.metrics import r2_score

"""predict -> R2_score"""

com_r2 = r2_score(a,b)

x = sum((b-a)**2)
y = sum((a-np.mean(a))**2)
my_r2 = 1-x/y

print("computer predict: ", com_r2)
print("my predict: ", my_r2)

# >>> computer predict:  -0.4736842105263157
# >>> my predict:  -0.4736842105263157
