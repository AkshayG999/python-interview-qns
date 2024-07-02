# Task 1: Data Manipulation with Pandas
Objective:
Filter and transform a DataFrame.

Instructions:
Given the following DataFrame, filter rows where the value in the "age" column is greater than 25 and create a new column "is_adult" that indicates if the age is 18 or older.

import pandas as pd

data = {'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'age': [24, 27, 22, 32, 29],
        'salary': [48000, 54000, 35000, 58000, 62000]}

df = pd.DataFrame(data)

 Task: Write code to filter and transform the DataFrame as described
Expected Output:

# Filtered and transformed DataFrame
# Expected Output:
#       name  age  salary  is_adult
# 1      Bob   27   54000      True
# 3    David   32   58000      True
# 4      Eva   29   62000      True


# -----ANSWER-------

import pandas as pd

data = {'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
        'age': [24, 27, 22, 32, 29],
        'salary': [48000, 54000, 35000, 58000, 62000]}

df = pd.DataFrame(data)

# Filter rows where age > 25
filtered_df = df[df['age'] > 25]

# Create a new column 'is_adult' indicating if the age is 18 or older
filtered_df['is_adult'] = filtered_df['age'] >= 18

print(filtered_df)

Expected Output:

     name  age  salary  is_adult
1     Bob   27   54000      True
3   David   32   58000      True
4     Eva   29   62000      True

# ------------------------------------------------------------------------------------

# Task 2: Basic Machine Learning with scikit-learn
Objective:
Fit a simple linear regression model and predict a value.

Instructions:
Given the following dataset, fit a linear regression model to predict y based on x and predict the y value when x = 5.

from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 5, 7, 11])

# Task: Write code to fit the model and predict y for x = 5
Expected Output:

# Prediction for x = 5
# Expected Output: Approximately 11 (depending on the fit)

# -----ANSWER-------
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 5, 7, 11])

# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(x, y)

# Predict y for x = 5
y_pred = model.predict(np.array([[5]]))

print("Predicted value for x = 5:", y_pred[0])

Predicted value for x = 5: 10.999999999999998

# ------------------------------------------------------------------------------------


# Task 3: Numpy Array Operations
Objective:
Perform specific operations on a NumPy array.

Instructions:
Given the following array, normalize the values to be between 0 and 1.


import numpy as np

arr = np.array([3, 6, 9, 12, 15])

# Task: Write code to normalize the array values between 0 and 1
Expected Output:

# Normalized array
# Expected Output: array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])

# -----ANSWER-------

import numpy as np

arr = np.array([3, 6, 9, 12, 15])

# Normalize the array values between 0 and 1
normalized_arr = (arr - arr.min()) / (arr.max() - arr.min())

print("Normalized array:", normalized_arr)

Normalized array: [0.   0.2  0.4  0.6  0.8]

# ------------------------------------------------------------------------------------


# Task 4: Simple Text Processing
Objective:
Count the number of words in a string.

Instructions:
Given the following string, count the number of words.

text = "Machine learning is fascinating and challenging."

# Task: Write code to count the number of words in the string
Expected Output:

# Word count
# Expected Output: 6

# -----ANSWER-------
text = "Machine learning is fascinating and challenging."

# Count the number of words
word_count = len(text.split())

print("Word count:", word_count)

Word count: 6


# ------------------------------------------------------------------------------------

# Task 5: Dictionary Operations
Objective:
Find the key with the highest value in a dictionary.

Instructions:
Given the following dictionary, find the key associated with the maximum value.

scores = {'Alice': 82, 'Bob': 91, 'Charlie': 88, 'David': 79}

# Task: Write code to find the key with the highest value
Expected Output:

# Key with the highest value
# Expected Output: 'Bob'

# -----ANSWER-------
scores = {'Alice': 82, 'Bob': 91, 'Charlie': 88, 'David': 79}

# Find the key with the highest value
max_key = max(scores, key=scores.get)

print("Key with the highest value:", max_key)

Key with the highest value: Bob

# ------------------------------------------------------------------------------------

Task 6: Simple Plotting with Matplotlib
Objective:
Plot a basic line graph.

Instructions:
Given the following lists, plot a line graph.

import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]

# Task: Write code to plot a line graph with x and y values
Expected Output:

# A line plot showing y = x^2

# -----ANSWER-------
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]

# Plot a line graph
plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = x^2')
plt.grid(True)
plt.show()


Expected Output:
A plot showing the line y = x^2 with points marked at each x, displayed in a grid.

# ------------------------------------------------------------------------------------

Task 7: List Comprehensions
Objective:
Generate a list of squares.

Instructions:
Given the range 1 to 10, generate a list of squares of these numbers using a list comprehension.

# Task: Write code to generate the list of squares
Expected Output:

# List of squares
# Expected Output: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# -----ANSWER-------
# Generate the list of squares from 1 to 10
squares = [x**2 for x in range(1, 11)]

print("List of squares:", squares)

List of squares: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# ------------------------------------------------------------------------------------

Task 8: Dictionary Comprehensions
Given a sorted list of positive integers with some entries being None, write a function to return a new list where all None values are replaced with the most recent non-None value in the list.

input_list = [1,2,None,None,4,5,None]

# -----ANSWER-------
 
import pandas as pd

def fill_none(input_list):
    # Create a pandas Series from the input list
    series = pd.Series(input_list)
    
    # Replace None values with 0 for the first element if it is None
    if series.iloc[0] is None:
        series.iloc[0] = 0
    
    # Use forward fill to replace None values with the most recent non-None value
    filled_series = series.fillna(method='ffill')
    
    # Convert the series back to a list and return
    return filled_series.tolist()

# Example usage
input_list = [1, 2, None, None, 4, 5, None]
output_list = fill_none(input_list)
print(output_list)


# ------------------------------------------------------------------------------------


class A:
    def greet(self):
        return "Hello from A"

class B:
    def greet(self):
        return "Hello from B"

class C(A, B):
    pass

obj = C()
print(obj.greet())
