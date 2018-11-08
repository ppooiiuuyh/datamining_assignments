import numpy as np
import matplotlib.pyplot as plt

num     = 201
std     = 20
a       = 2 
b       = 10


n       = np.random.rand(num)
nn      = n - np.mean(n)
x       = np.linspace(-100,100,num)

y1      = a * x + nn * std + b 



#cal a and b
a_ =0
b_ =0



minimum = 999999999
for i in np.arange(-10,10,0.1):
    temp = np.sqrt(np.sum(np.square(y1 - i*x)))
    if temp < minimum :
        minimum = temp
        a_ = i
print(a_)

minimum = 999999
for i in np.arange(-10,10,0.1):
    temp = np.sqrt(np.sum(np.square(y1 - (a_*x+i))))
    if temp < minimum :
        minimum = temp
        b_ = i
print(b_)

y2      = a_ * x + b_


plt.plot(x, y1, 'b.', x, y2, 'k--')
plt.show()

# x  : x-coordinate data
# y1 : (noisy) y-coordinate data
# y2 : (clean) y-coordinate data 
# y = f(x) = a * x + b
