from loading_helpers import compute_class_list
import json

filename = 'descriptors/gpt3/descriptors_imagenet.json'

with open(filename, 'r') as f:
    data = json.load(f)

class_list = compute_class_list(data, sort_config=False)

# example_class_list = ["tiger shark", "night snake", "hermit crab"]

from openai import OpenAI
client = OpenAI()

subcategories = {}
count = 0
for class_name in class_list:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "The ImageNet dataset is constructed from 1000 classes. I will need you to create at minimum 40 subcategories and assign at maximum 25 of the ImageNet classes to each. For example of a subcategory and its classes, a subcategory \"dog\" may have the classes \"golden retriever\", \"greyhound\", and \"daschund\" assigned to it. Every class must be assigned to a subcategory, none can be missed. First, create the list of subcategories to assign these classes to, in the form of a Python list, and stop there before assigning the classes."
                }
            ]
            },
            {
            "role": "assistant",
            "content": [
                {
                "type": "text",
                "text": f'Here is a list of 40 subcategories for the ImageNet classes:\n\nsubcategories = ["fish","birds","amphibians","reptiles","mammals","insects","arachnids","crustaceans","coral and sea life","dogs","cats","wild animals","domestic animals","primates","ungulates","carnivores","herbivores","rodents","marsupials","food","beverages","vegetables","fruits","grains","dairy products","meats","snacks","sweets","utensils","furniture","vehicles","musical instruments","clothing","electronics","kitchen appliances","office supplies","toys","sports equipment","outdoor gear","home decor","tools","building materials","art supplies","games","stationery"]'
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"Which of the subcategories in the above Python list should {class_name} be assigned to? It must be one of the subcategories in the list, not a new one. Respond with only the subcategory name."
                }
            ]
            }
        ],
        temperature=0,
        max_tokens=40,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={
            "type": "text"
        }
    )

    subcategory = str(response.choices[0].message.content).replace('\"', '').replace('\'', '')
    count += 1
    print(f"Class number {count} of {len(class_list)}: ", subcategory)

    if subcategory in subcategories:
        subcategories[subcategory].append(class_name)
    else:
        subcategories[subcategory] = [class_name]

print(subcategories)

class_filename = 'class_analysis/json/openai_class_analysis_imagenet.json'

with open(class_filename, 'w') as f:
    json.dump(subcategories, f, indent=4)