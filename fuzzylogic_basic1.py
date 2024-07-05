
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

'''
# eg 1
# Define load range

# Linear Membership Functions (Triangular)

warm_linear = fuzz.trimf(x_temp, [10, 30, 30])
hot_linear = fuzz.trimf(x_temp, [10, 40, 40])

# Gaussian Membership Functions
cold_gauss = fuzz.gaussmf(x_temp, 0, 2)
warm_gauss = fuzz.gaussmf(x_temp, 30, 3)
hot_gauss = fuzz.gaussmf(x_temp, 40, 5)

# Plotting Linear Membership Functions
plt.figure(figsize=(12, 6))

plt.subplot(121)

plt.plot(x_temp, warm_linear, 'y', linewidth=2, label='Warm (Linear)')
plt.plot(x_temp, hot_linear, 'r', linewidth=2, label='Hot (Linear)')
plt.title('Linear Membership Functions')
plt.xlabel('Temperature (F)')
plt.ylabel('Membership Degree')
plt.legend(loc='upper right', title='Temperature States', fontsize='small')


# Plotting Gaussian Membership Functions
plt.subplot(122)
plt.plot(x_temp, cold_gauss, 'b', linewidth=2, label='Cold (Gaussian)')
plt.plot(x_temp, warm_gauss, 'y', linewidth=2, label='Warm (Gaussian)')
plt.plot(x_temp, hot_gauss, 'r', linewidth=2, label='Hot (Gaussian)')
plt.title('Gaussian Membership Functions')
plt.xlabel('Temperature (F)')
plt.ylabel('Membership Degree')
plt.legend(loc='upper right', title='Temperature States', fontsize='small')
plt.tight_layout()
plt.show()




#eg 2

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.arange(0, 40, 0.1)
y = fuzz.trapmf(x, [0, 2, 5, 7])

cold_linear = fuzz.trapmf(x, [0, 5, 10, 15])
warm_linear = fuzz.trapmf(x, [10, 20, 25, 30])
hot_linear = fuzz.trapmf(x, [20, 30, 35 , 40])

plt.plot(x, cold_linear, 'b', linewidth=2, label='Cold (Linear)')
plt.plot(x, warm_linear, 'y', linewidth=2, label='Cold (Linear)')
plt.plot(x, hot_linear, 'g', linewidth=2, label='Cold (Linear)')

plt.ylabel('Membership value')
plt.xlabel('Temperature range')
plt.legend(loc='upper right', title='Temperature States', fontsize='small')
plt.show()




#eg3

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.arange(0, 40, 0.1)
# gaussian curve for [5.122, 10.72]
curve1_gauss = fuzz.gaussmf(x, 10.72, 5.122)
# gaussian curve for [4.481, 19.21]
curve2_gauss = fuzz.gaussmf(x, 19.21, 4.481)
# gaussian curve for  [4.768, 25.52]
curve3_gauss = fuzz.gaussmf(x, 25.52, 4.768)
plt.plot(x, curve1_gauss, 'b', linewidth=2, label='[sigma 5.122, mean 10.72] (Gaussian)')
plt.plot(x, curve2_gauss, 'y', linewidth=2, label='[sigma 4.481, mean 19.21] (Gaussian)')
plt.plot(x, curve3_gauss, 'g', linewidth=2, label='[sigma 4.768, mean 25.52] (Gaussian)')
plt.ylabel('Membership value')
plt.xlabel('Temperature range')
plt.legend(loc='upper right', title='Temperature States', fontsize='small')
plt.show()


# eg 4

def gaussimf(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

def trianmf(x, a, b, c):
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0)

x_temp = np.linspace(0, 40, 400)

# Linear Membership Functions (Triangular)
cold_linear = trianmf(x_temp, 0, 0, 30)

# Gaussian Membership Functions
cold_gauss = gaussimf(x_temp, 0, 2)

# Plotting Linear Membership Functions
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(x_temp, cold_linear, 'b', linewidth=2, label='Cold (Linear)')
plt.title('Linear Membership Functions')
plt.xlabel('Temperature (F)')
plt.ylabel('Membership Degree')
plt.legend(loc='upper right', title='Temperature States', fontsize='small')
# Plotting Gaussian Membership Functions
plt.subplot(122)
plt.plot(x_temp, cold_gauss, 'b', linewidth=2, label='Cold (Gaussian)')
plt.title('Gaussian Membership Functions')
plt.xlabel('Temperature (F)')
plt.ylabel('Membership Degree')
plt.legend(loc='upper right', title='Temperature States', fontsize='small')
plt.tight_layout()
plt.show()



# eg 5

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Generate trapezoidal membership function on range [0, 1]
x_temp = np.linspace(0, 40, 400)
# mfx = fuzz.trapmf(x, [0, 2, 5, 5])

# Linear Membership Functions (Triangular)
cold_linear = fuzz.trimf(x_temp, [0, 0, 30])
warm_linear = fuzz.trimf(x_temp, [10, 30, 30])
hot_linear = fuzz.trimf(x_temp, [10, 40, 40])

# Gaussian Membership Functions
cold_gauss = fuzz.gaussmf(x_temp, 0, 2)
warm_gauss = fuzz.gaussmf(x_temp, 30, 3)
hot_gauss = fuzz.gaussmf(x_temp, 40, 5)

# Aggregate the fuzzy sets
aggregated_lin = np.fmax(cold_linear, np.fmax(warm_linear, hot_linear))
# Aggregate the fuzzy sets
aggregated_gaus = np.fmax(cold_gauss, np.fmax(warm_gauss, hot_gauss))



# Defuzzify this membership function five ways
defuzz_centroid = fuzz.defuzz(x_temp, cold_linear, 'centroid')  # Same as skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x_temp, cold_linear, 'bisector')
defuzz_mom = fuzz.defuzz(x_temp, cold_linear, 'mom')
defuzz_som = fuzz.defuzz(x_temp, cold_linear, 'som')
defuzz_lom = fuzz.defuzz(x_temp, cold_linear, 'lom')

defuzz_centroid = fuzz.defuzz(x_temp, warm_linear, 'centroid')  # Same as skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x_temp, warm_linear, 'bisector')
defuzz_mom = fuzz.defuzz(x_temp, warm_linear, 'mom')
defuzz_som = fuzz.defuzz(x_temp, warm_linear, 'som')
defuzz_lom = fuzz.defuzz(x_temp, warm_linear, 'lom')

defuzz_centroid = fuzz.defuzz(x_temp, hot_linear, 'centroid')  # Same as skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x_temp, hot_linear, 'bisector')
defuzz_mom = fuzz.defuzz(x_temp, hot_linear, 'mom')
defuzz_som = fuzz.defuzz(x_temp, hot_linear, 'som')
defuzz_lom = fuzz.defuzz(x_temp, hot_linear, 'lom')

# defuzz gaussian

defuzz_centroid = fuzz.defuzz(x_temp, cold_gauss, 'centroid')  # Same as skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x_temp, cold_gauss, 'bisector')
defuzz_mom = fuzz.defuzz(x_temp, cold_gauss, 'mom')
defuzz_som = fuzz.defuzz(x_temp, cold_gauss, 'som')
defuzz_lom = fuzz.defuzz(x_temp, cold_gauss, 'lom')

defuzz_centroid = fuzz.defuzz(x_temp, warm_gauss, 'centroid')  # Same as skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x_temp, warm_gauss, 'bisector')
defuzz_mom = fuzz.defuzz(x_temp, warm_gauss, 'mom')
defuzz_som = fuzz.defuzz(x_temp, warm_gauss, 'som')
defuzz_lom = fuzz.defuzz(x_temp, warm_gauss, 'lom')

defuzz_centroid = fuzz.defuzz(x_temp, hot_gauss, 'centroid')  # Same as skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x_temp, hot_gauss, 'bisector')
defuzz_mom = fuzz.defuzz(x_temp, hot_gauss, 'mom')
defuzz_som = fuzz.defuzz(x_temp, hot_gauss, 'som')
defuzz_lom = fuzz.defuzz(x_temp, hot_gauss, 'lom')


# Collect info for vertical lines
labels = ['centroid', 'bisector', 'mean of maximum', 'min of maximum', 'max of maximum']
xvals = [defuzz_centroid, defuzz_bisector, defuzz_mom, defuzz_som, defuzz_lom]
colors = ['r', 'b', 'g', 'c', 'm']
ymax = [fuzz.interp_membership(x_temp, cold_linear, i) for i in xvals]
ymax = [fuzz.interp_membership(x_temp, warm_linear, i) for i in xvals]
ymax = [fuzz.interp_membership(x_temp, hot_linear, i) for i in xvals]
ymax = [fuzz.interp_membership(x_temp, cold_gauss, i) for i in xvals]
ymax = [fuzz.interp_membership(x_temp, warm_gauss, i) for i in xvals]
ymax = [fuzz.interp_membership(x_temp, hot_gauss, i) for i in xvals]


# Display and compare defuzzification results against membership function
plt.figure(figsize=(8, 5))

plt.plot(x_temp, cold_linear, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)

plt.plot(x_temp, warm_linear, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)

plt.plot(x_temp, hot_linear, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)

plt.plot(x_temp, cold_gauss, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)

plt.plot(x_temp, warm_gauss, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)

plt.plot(x_temp, hot_gauss, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)


plt.ylabel('Fuzzy membership')
plt.xlabel('Universe variable (arb)')
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()



import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Generate load range
x_temp = np.linspace(0, 40, 400)

# Linear Membership Functions (Triangular)
cold_linear = fuzz.trimf(x_temp, [0, 0, 30])
warm_linear = fuzz.trimf(x_temp, [10, 30, 30])
hot_linear = fuzz.trimf(x_temp, [10, 40, 40])

# Gaussian Membership Functions
cold_gauss = fuzz.gaussmf(x_temp, 0, 2)
warm_gauss = fuzz.gaussmf(x_temp, 30, 3)
hot_gauss = fuzz.gaussmf(x_temp, 40, 5)

# Aggregate the fuzzy sets
aggregated_lin = np.fmax(cold_linear, np.fmax(warm_linear, hot_linear))
aggregated_gaus = np.fmax(cold_gauss, np.fmax(warm_gauss, hot_gauss))

# Defuzzify using the centroid method on the aggregated fuzzy sets
defuzz_centroid_lin = fuzz.defuzz(x_temp, aggregated_lin, 'centroid')
defuzz_centroid_gaus = fuzz.defuzz(x_temp, aggregated_gaus, 'centroid')

# Collect info for vertical lines
labels = ['centroid']
xvals_lin = [defuzz_centroid_lin]
xvals_gaus = [defuzz_centroid_gaus]
colors = ['r']
ymax_lin = [fuzz.interp_membership(x_temp, aggregated_lin, i) for i in xvals_lin]
ymax_gaus = [fuzz.interp_membership(x_temp, aggregated_gaus, i) for i in xvals_gaus]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plot aggregated linear
axs[0].plot(x_temp, aggregated_lin, 'k', label='Aggregated Linear')
for xv, y, label, color in zip(xvals_lin, ymax_lin, labels, colors):
    axs[0].vlines(xv, 0, y, label=label, color=color)
axs[0].set_title('Aggregated Linear')
axs[0].legend()

# Plot aggregated Gaussian
axs[1].plot(x_temp, aggregated_gaus, 'k', label='Aggregated Gaussian')
for xv, y, label, color in zip(xvals_gaus, ymax_gaus, labels, colors):
    axs[1].vlines(xv, 0, y, label=label, color=color)
axs[1].set_title('Aggregated Gaussian')
axs[1].legend()

plt.tight_layout()
plt.show()

# Print the centroid values
print(f"Centroid of aggregated linear fuzzy set: {defuzz_centroid_lin}")
print(f"Centroid of aggregated Gaussian fuzzy set: {defuzz_centroid_gaus}")
'''
from skfuzzy import control as ctrl

load = np.arange(0, 31, 1)
fabric = np.arange(0, 1.1, 0.1)
spin_period = np.arange(0, 21, 1)
wash_time = np.arange(0, 121, 10)


load_light = fuzz.trimf(load, [0, 10, 10])
load_medium = fuzz.trimf(load, [10, 20, 20])
load_heavy = fuzz.trimf(load, [20, 30, 30])


fabric_rough = fuzz.trimf(fabric, [0, 0.3, 0.3])
fabric_normal = fuzz.trimf(fabric, [0.3, 0.7, 0.7])
fabric_fine = fuzz.trimf(fabric, [0.7, 1, 1])


spin_period_short = fuzz.trimf(spin_period, [0, 10, 10])
spin_period_long = fuzz.trimf(spin_period, [10, 20, 20])

wash_time_short = fuzz.trimf(wash_time, [0, 60, 60])
wash_time_long = fuzz.trimf(wash_time, [60, 120, 120])

plt.figure(figsize=(17, 4))

# First subplot for load
plt.subplot(141)
plt.plot(load, load_light, 'b', linewidth=2, label='Cold')
plt.plot(load, load_medium, 'g', linewidth=2, label='Moderate')
plt.plot(load, load_heavy, 'r', linewidth=2, label='Hot')
plt.title('Amount of cloths Membership Functions')
plt.xlabel('load')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)

# Second subplot for fabric
plt.subplot(142)
plt.plot(fabric, fabric_rough, 'b', linewidth=2, label='Dry')
plt.plot(fabric, fabric_normal, 'g', linewidth=2, label='Moderate')
plt.plot(fabric, fabric_fine, 'r', linewidth=2, label='Wet')
plt.title('fabric Membership Functions')
plt.xlabel('fabric quality')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)

# Third subplot for fan speed
plt.subplot(143)
plt.plot(spin_period, spin_period_short, 'b', linewidth=2, label='Slow')
plt.plot(spin_period, spin_period_long, 'g', linewidth=2, label='Medium')

plt.title('spin period Membership Functions')
plt.xlabel('spin time')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)

plt.subplot(144)
plt.plot(wash_time, wash_time_short, 'b', linewidth=2, label='Slow')
plt.plot(wash_time, wash_time_long, 'g', linewidth=2, label='Medium')

plt.title('wash time Membership Functions')
plt.xlabel('wash time')
plt.ylabel('Membership Degree')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()