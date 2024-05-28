import math
import matplotlib.pyplot as plt
import random

angle = float(input("Podaj kąt w stopniach: "))

speed = 50  # m/s
height = 100 # meters

rads = math.radians(angle)
vspeed = speed * math.sin(rads)
hspeed = speed * math.cos(rads)

distance = (speed * math.sin(rads) + math.sqrt(speed**2 * math.sin(rads) + 2 * 9.81 * height))*speed*math.cos(rads)/9.81

distance_hit = random.randint(50, 340)

i = 0

while(math.fabs(distance - distance_hit) > 20):
    i+=1
    distance_hit = random.randint(50, 340)
    distance = (speed * math.sin(rads) + math.sqrt(speed**2 * math.sin(rads) + 2 * 9.81 * height))*speed*math.cos(rads)/9.81
    
print("Liczba iteracji: ", i)
print("Dystans: ", distance)
print("Dystans trafienia: ", distance_hit)
print("Roznica dystansu trafienia: ", distance - distance_hit)

x = []
y = []

for i in range(0,distance_hit+10):
    x.append(i)
    y.append( (-1)*(9.81/2)/(hspeed**2 * math.cos(rads)**2) * i**2 + i*math.sin(rads)/math.cos(rads) + height)


plt.plot(x, y, 'ro-')
plt.show()