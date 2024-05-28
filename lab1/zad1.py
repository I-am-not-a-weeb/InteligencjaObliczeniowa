import datetime
import math

# Prompt the user for their name
name = input("Podaj swoje imie: ")

# Prompt the user for their date of birth
dob = input("Wpisz swoja date urodzenia (format YYYY-MM-DD): ")

# Convert the date of birth string to a datetime object
dob_date = datetime.datetime.strptime(dob, "%Y-%m-%d")

# Calculate the number of days since the date of birth
current_date = datetime.datetime.now()
days_passed = (current_date - dob_date).days


phisical_phase = math.sin(2*math.pi*days_passed/23)
emotional_phase = math.sin(2*math.pi*days_passed/28)
intellectual_phase = math.sin(2*math.pi*days_passed/33)

print("faza fizyczna: ", phisical_phase)
print("faza emocjonalna: ", emotional_phase)
print("faza intelektualna: ", intellectual_phase)

if(phisical_phase < 0 or emotional_phase < 0 or intellectual_phase < 0):
    phisical_phase = math.sin(2*math.pi*(days_passed+1)/23)
    emotional_phase = math.sin(2*math.pi*(days_passed+1)/28)
    intellectual_phase = math.sin(2*math.pi*(days_passed+1)/33)

print("jutrzejsza faza fizyczna: ", phisical_phase)
print("jutrzejsza faza emocjonalna: ", emotional_phase)
print("jutrzejsza faza intelektualna ", intellectual_phase)