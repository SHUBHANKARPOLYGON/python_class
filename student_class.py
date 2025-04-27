# Define the Student class
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display_info(self):
        print(f"Student Name: {self.name}")
        print(f"Student Age: {self.age}")

# Create an object of the Student class
student1 = Student("Shubhankar", 21)

# Call the method to display student information
student1.display_info()
