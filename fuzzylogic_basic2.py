import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
# New Antecedent/Consequent objects hold universe variables and membership
# functions
temperature = np.arange(0, 41, 1)
humidity = np.arange(0, 1.1, 0.1)
fan_speed = np.arange(0, 101, 10)

temp_uni = ctrl.Antecedent(temperature, 'temperature')
humidity_uni = ctrl.Antecedent(humidity, 'humidity')
fan_speed_uni = ctrl.Consequent(fan_speed, 'fan speed')

# Auto-membership function population is possible with .automf(3, 5, or 7)

temp_uni['cold'] = fuzz.trimf(temp_uni.universe, [10, 10, 15])
temp_uni['moderate'] = fuzz.trimf(temp_uni.universe, [12, 20, 27])
temp_uni['hot'] = fuzz.trimf(temp_uni.universe, [25, 40, 40])

humidity_uni['dry'] = fuzz.trimf(humidity_uni.universe, [0, 0, 0.4])
humidity_uni['moderate'] = fuzz.trimf(humidity_uni.universe, [0.3, 0.45, 0.65])
humidity_uni['wet'] = fuzz.trimf(humidity_uni.universe, [0.5, 1, 1])

fan_speed_uni['slow'] = fuzz.trimf(fan_speed_uni.universe, [0, 10, 30])
fan_speed_uni['medium'] = fuzz.trimf(fan_speed_uni.universe, [25, 45, 60])
fan_speed_uni['fast'] = fuzz.trimf(fan_speed_uni.universe, [50, 70, 100])

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
temp_uni.view()
humidity_uni.view()
fan_speed_uni.view()

rule1 = ctrl.Rule((temp_uni['cold'] & humidity_uni['dry']) | (temp_uni['cold'] & humidity_uni['wet']),
                  fan_speed_uni['slow'])
rule2 = ctrl.Rule(temp_uni['moderate'] & humidity_uni['wet'], fan_speed_uni['slow'])
rule3 = ctrl.Rule(temp_uni['moderate'] & humidity_uni['dry'], fan_speed_uni['medium'])
rule4 = ctrl.Rule(temp_uni['moderate'] & humidity_uni['wet'], fan_speed_uni['fast'])
rule5 = ctrl.Rule((temp_uni['hot'] & humidity_uni['wet']) | (temp_uni['hot'] & humidity_uni['dry']),
                  fan_speed_uni['fast'])

rule1.view()
rule2.view()
rule3.view()
rule4.view()
rule5.view()

fan_speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4, rule5])
fspeed = ctrl.ControlSystemSimulation(fan_speed_ctrl)
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
fspeed.input['temperature'] = 18  # Adjusted input value
fspeed.input['humidity'] = 0.3    # Adjusted input value
# Crunch the numbers
fspeed.compute()
# Print the output
print(f"Fan Speed: {fspeed.output['fan speed']:.2f}")
# Visualize the result (optional)
fan_speed_uni.view(sim=fspeed)
plt.show()
