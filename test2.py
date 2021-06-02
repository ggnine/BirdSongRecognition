import random
f = open("./myfile"+str(random.randint(1,10000000))+".txt", "w")
f.write("hi")