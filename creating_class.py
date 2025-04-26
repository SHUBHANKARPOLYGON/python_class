# Define a class
class Person:
    # Constructor to initialize object
    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Method to display details
    def display(self):
        print(f"Name: {self.name}")
        print(f"Age: {self.age}")

# Create an object of the class
person1 = Person("Shubhankar Tiwary", 21)

# Call the method
person1.display()
