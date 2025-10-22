from safetycheck import SafetyCheck

# initialize
checker = SafetyCheck()

# test examples
texts = [
    "I feel like nothing matters anymore.",
    "Today was a wonderful day with my family.",
    "im so happy i could die on the spot right now",
    "laugh die me", 
    "when will this end", 
    "im feeling very anxious lately",
    "im so tired i want to sleep forever", 
    "why is life so hard, i cant keep doing this", 
    "im so tired",
    "i just ate a bottle of panadol, goodbye world", 
    " i think ill jump off a bridge tomorrow", 
    "i want to kill myself", 
    " my friend killed herself", 
    "my wife died, i want to join her", 
    "im had such a great day today!", 
    "its my birthday today!", 
    "i just dont see a point in living", 
    "i don't really enjoy life, it kinda just passes me by", 
    "i want to off myself",
    "goodbye"
]

for t in texts:
    print("\n---")
    result = checker(t)
    print(f"Returned: {result}")
