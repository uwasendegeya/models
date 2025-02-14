def greet_decorator(funct):
    def wrapper(name):
        print("Good Afternoon")
        funct(name)
    return wrapper
@greet_decorator
def display(name):
    print(f"{name}")
display("Nina")
