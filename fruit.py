# def displayNames(name):
#     print(f"{name}")
# a=("apple","strawberry","mango","pineapple","lemon")
# x=input("enter fruit")
# if(x in a):
#     print("fruit exist")
from datetime import datetime

class car:
    def __init__(self,brand,model,year):
        self.brand=brand
        self.model=model
        self.year=year

    def get_info(self):
        return(f"my car:\n{self.brand}\n{self.model}\n{self.year}")
    def isVintage(self):
        now=datetime.now().year
        # print(now)
        if((now-self.year)>24):
            print("true")
        else:
            print("false")

a=car("TOYOTA","V8",1000)
print(a.get_info())
print(a.isVintage())
    