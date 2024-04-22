class Animal:
    def __init__(self, name):
        self.name = name

    def talk(self):
        print("I am an animal.")

# 子类（Derived Class）
class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # 调用父类的初始化方法
        self.breed = breed

    def speak(self):
        print(f"I am a {self.breed} dog.")

# 创建子类实例
dog = Dog("Buddy", "Golden Retriever")

# 调用子类的方法
dog.speak()  # 输出: I am a Golden Retriever dog.
dog.talk()  # 输出：I am an animal.
# 注意，如果这里talk的命名为speak，则输出仍为 I am a Golden Retriever dog.（有先调用自己的函数）