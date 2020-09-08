import numpy as np
from sympy import *
from sympy.plotting import plot

# Generate function.
x = symbols('x')
y = sin(x)
for a in range(3, 10, 2):
    y += sin(a * x) / a

# Plot function.
p1 = plot(y, title="Plot of original function.")

# `n` in an and bn.
N = 10

# Find a0.
y_ = (1/(2*pi)) * integrate(y, (x, -pi, pi))
print(y_)

# Find an.
for n in range(1, N):
    y_ += (1/pi) * integrate(y * cos(n*x), (x, -pi, pi)) * cos(n*x)

# Find bn.
for n in range(1, N):
    y_ += (1/pi) * integrate(y * sin(n*x), (x, -pi, pi)) * sin(n*x)

print(y_)
p2 = plot(y_, title="Plot of estimated function.")
