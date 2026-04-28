import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

with open("ch02/edith-wharton.txt", "r") as f:
    text = f.read()

print("Total number of character:", len(text))
print(text[:99])
