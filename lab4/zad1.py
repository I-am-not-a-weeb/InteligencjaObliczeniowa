import math

arr = [
    [23, 75, 176,True],
    [25, 67, 180,True],
    [28, 120, 175,False],
    [22, 65, 165, True],
    [46, 70, 187, True],
    [50, 68, 180, False],
    [48, 97, 178, False],
    [22,80,183,False]
]

def summ(x):
    return 1/(1+math.e**(-x))

def classify(age,weight,height):
  first_node = summ(age * -0.46122 + weight * 0.97314 + height * -0.39203 + 0.80109)
  second_node = summ(age * 0.78548 + weight * 2.10584 + height * - 0.57847 + 0.43529)
  third_node = summ(first_node * -0.81546 + second_node * 1.03775 - 0.2368)
  return round(third_node, 0)

def test_network():
    for i in arr:
        age = i[0]
        weight = i[1]
        height = i[2]
        expected = i[3]
        result = classify(age, weight, height)
        print(f'Expected: {expected}, result of classify: {bool(result)}')
        
test_network()