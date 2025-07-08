axis_keys = [
    "Age", "Sex", "Chest pain", "Blood pressure", "Cholesterol [mg/dl]", "Fasting blood sugar",
    "Resting electrocardiographic", "Maximum heart rate", "Exercise induced angina",
    "Oldpeak", "Peak slopee", "#Major vessels", "Thal",
]

input_size = 13
lower_bounds = [29.0, 0.0, 0.0, 94.0, 126.0, 0.0, 0.0, 71.0, 0.0, 0.0, 0.0, 0.0, 0.0]
upper_bounds = [77.0, 1.0, 3.0, 200.0, 564.0, 1.0, 2.0, 202.0, 1.0, 6.2, 2.0, 4.0, 3.0]
bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))]