import json

outpath = r"C:\Users\prabh\testMLflow\andor.json"
data = {
    "title": "Andor",
    "review": (
        "In an era filled with danger, deception, and intrigue, Cassian Andor embarks on a path that is destined to turn him into a Rebel hero."
    ),
}

# Create and write to the JSON file
with open(outpath, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("JSON file successfully created!")
