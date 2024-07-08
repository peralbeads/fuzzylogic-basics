# task 1
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(10)
plt.plot(x, 4 * x, label='y=x', color='#90EE90')
plt.plot(x, 3 * x, label='y=2x', color='#00008B')
plt.plot(x, 2 * x, label='y=3x', color='green')
plt.plot(x, x, label='y=4x', color='red')
plt.legend()
plt.show()

#task2

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

x = np.linspace(-3, 3, 100)
y = x**2  # Parabola function: y = x^2

# Plot the parabola
plt.plot(x, y, label='y=x^2', color='purple')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#task3

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin, label='sin(x)', color='blue')
plt.plot(x, y_cos, label='cos(x)', color='red')

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#task4

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 100)
y_square = x ** 2
y_cube = x ** 3

plt.plot(x, y_square, label='$y=\\alpha^2$', color='blue')
plt.plot(x, y_cube, label='$y=\\alpha^3$', color='green')

plt.xlabel("$\\alpha$")
plt.ylabel("$y$")
plt.legend()
plt.show()
