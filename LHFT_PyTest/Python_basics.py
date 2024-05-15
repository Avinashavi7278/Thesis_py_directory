##  The f before the quotation marks tells Python to look at the contents and replace any expression inside {}





################################################################################################################

x = 10
y = "Hello"
# Here, x is an integer, and y is a string. Python automatically detects the data type based on the value assigned.


################################################################################################################


# Control Structures
# Control structures in Python include if, elif, else for conditional operations, and loops like for and while for iterating over data. 
# These structures help manage the flow of a program and are crucial for decision making and repeating tasks.

#Example:


if x > 5:
    print("x is greater than 5")
elif x == 5:
    print("x is 5")
else:
    print("x is less than 5")

for i in range(5):
    print(i)

################################################################################################################

# Classes and Objects
# Python supports Object-Oriented Programming (OOP). Classes provide a means of bundling data and functionality together. 
# Creating a new class creates a new type of object, allowing new instances of that type to be made.

# Example:


class Dog:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name} says Woof!"

dog = Dog("Buddy")
print(dog.speak())

# the self keyword is used in object-oriented programming to refer to the instance of the class. It helps differentiate between instance attributes (and methods)
#  and local variables (or methods defined outside the class), allowing the class's methods to access attributes or other methods that belong to the same object



# One more example for self 

class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # grade is a number between 0 and 100

    def is_passing(self):
        return self.grade >= 60

    def birthday(self):
        self.age += 1
        return f"Happy {self.age}th Birthday, {self.name}!"

# Create an instance of Student
student1 = Student("Alice", 15, 85)
print(student1.is_passing())  # Output: True
print(student1.birthday())    # Output: Happy 16th Birthday, Alice!


################################################################################################################
# Modules and Packages
# Modules in Python are simply Python files with the .py extension containing Python definitions and statements. Packages are a way of structuring Python’s module namespace by using “dotted module names”. They help in organizing and reusing code effectively across different projects.

# Example:

# Assuming we have a module named 'mathematics' with a function 'add'
from mathematics import add

print(add(2, 3))

################################################################################################################

# Exception Handling
# Exception handling allows a programmer to handle errors gracefully and without crashing the program. It is done via the try, except blocks.

Example:

try:
    # Try to execute this block
    result = 10 / 0
except ZeroDivisionError:
    # Handle the error
    print("Divided by zero!")
################################################################################################################

# List Comprehensions
# List comprehensions provide a concise way to create lists. They consist of brackets containing an expression followed by a for clause, 
#then zero or more for or if clauses. They are a more readable and expressive way to create lists.

Example:


squares = [x**2 for x in range(10)]
print(squares)


################################################################################################################

# Create an empty list
my_list = []

# Append elements to the list
my_list.append(1)
my_list.append(2)
my_list.append(3)

print(my_list)  # Output: [1, 2, 3]

#we start with an empty list my_list, and then we use the append() method to add the integers 1, 2, and 3 to the list. Finally, we print the list to see the result.
################################################################################################################

# Example 1: Using abs() with integers
num = -10
abs_num = abs(num)
print("Absolute value of", num, "is", abs_num)  # Output: Absolute value of -10 is 10

# The absolute value of a number is its distance from zero on the number line, regardless of its direction. In other words, it is the non-negative 
# value of a real number without regard to its sign. For example:

# The absolute value of 5 is 5, because it is 5 units away from zero on the number line.
# The absolute value of -7 is 7, because it is also 7 units away from zero on the number line, just in the opposite direction.
# So, no matter whether the number is positive or negative, its absolute value is always positive or zero. The abs() function in Python calculates this absolute value.






################################################################################################################


arr1 = np.arange(10)
print("Array 1:", arr1)
# Result:
# Array 1: [0 1 2 3 4 5 6 7 8 9]

arr2 = np.arange(1, 11, 2)
print("Array 2:", arr2)
# Result:
# Array 2: [1 3 5 7 9]
################################################################################################################

# You can specify an axis to find the index of the maximum value along that axis. For example, axis=0 will find the index of the maximum value in each column, and axis=1 will do so for each row.

# python
# Copy code
# Find index of maximum value along each column
max_index_col = np.argmax(array2, axis=0)
print("Index of maximum value along each column:", max_index_col)

# Find index of maximum value along each row
max_index_row = np.argmax(array2, axis=1)
print("Index of maximum value along each row:", max_index_row)
# Output:

# sql
# Copy code
# Index of maximum value along each column: [1 0]
# Index of maximum value along each row: [1 0]
################################################################################################################
################################################################################################################
################################################################################################################

################################################################################################################
################################################################################################################################################################################################################################
################################################################################################################

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

