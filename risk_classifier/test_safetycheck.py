from SafetyCheck import SafetyCheck

# initialize
checker = SafetyCheck()

# test examples
texts = [
    "I feel hopeless and want to give up.",
    "Today was a beautiful day with my family.",
]

for t in texts:
    print("\n---")
    result = checker(t)
    print(f"Returned: {result}")
