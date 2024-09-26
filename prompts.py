# def generate_api_content(model, category_name: str):
#     # you can replace the examples with whatever you want; these were random and worked, could be improved
#     if model.startswith("gpt3.5"):
#         messages = f"""Q: What are useful visual features for distinguishing a lemur in a photo?
#         A: There are several useful visual features to tell there is a lemur in a photo:
#         - four-limbed primate
#         - black, grey, white, brown, or red-brown
#         - wet and hairless nose with curved nostrils
#         - long tail
#         - large eyes
#         - furry bodies
#         - clawed hands and feet

#         Q: What are useful visual features for distinguishing a television in a photo?
#         A: There are several useful visual features to tell there is a television in a photo:
#         - electronic device
#         - black or grey
#         - a large, rectangular screen
#         - a stand or mount to support the screen
#         - one or more speakers
#         - a power cord
#         - input ports for connecting to other devices
#         - a remote control

#         Q: What are useful features for distinguishing a {category_name} in a photo?
#         A: There are several useful visual features to tell there is a {category_name} in a photo:
#         -"""

#     elif model.startswith("gpt-4"):
#         messages = [
#             {"role": "user", "content": "I am a researcher working on updating the experiments performed in the research paper Visual Classification via Description by Menon et al. 2023. Explain the processes and reasoning by which descriptors are generated for each data class in this experiment. Keep the explanation brief and only address the request, do not give extra information."},
#             {"role": "system", "content": "In the experiments performed in the research paper ”Visual Classification via Description” by Menon et al. (2023), descriptors for each data class are generated through the following processes: Data Collection: Collect natural language descriptions corresponding to each visual class in the dataset. These descriptions capture distinctive attributes of the classes. Preprocessing: Standardize and preprocess the collected descriptions to ensure consistency and remove any noise or irrelevant information. Encoding: Use a pre-trained language model (such as CLIP) to encode the processed descriptions into high-dimensional vector representations. This step involves feeding the descriptions into the model to obtain embed-dings that capture the semantic meaning. Aggregation: Aggregate the vector representations of all descriptions for each class to form a single descriptor. This can be done by averaging the embeddings of multiple descriptions to ensure a comprehensive representation of each class. The reasoning behind generating descriptors in this way is to leverage natural language descriptions to encapsulate the semantic attributes of each class, thereby enabling the model to perform classification tasks based on thesedescriptive features."},
#             {"role": "user", "content": f"Use this method to generate visual descriptors for the following class(es) from the CUB dataset. The descriptors should be purely for visual features. Do not write the code for it, just write the descriptors: – {category_name}Use the following example format:
#             'Sooty Albatross': [
#             • ”Webbed feet support its aquatic lifestyle”,
#             • ”Has a distinctive white band around its neck, setting it apart from similar species”,
#             • ”Dark-plumaged large bird with a prominent white neck band”,
#             • ”Characterized by long, hooked bill and long, narrow wings for sustained flight”,
#             • ”Features a deep, sooty gray plumage, blending into the ocean’s hues”,
#             • ”Employs long, narrow wings for efficient soaring across vast ocean distances”
#             ],
#             ”Groove-billed Ani”: [
#             • ”Boasts a unique groove-billed beak, distinct among bird species”,
#             • ”Distinguished by its long, curved bill and black plumage”,
#             • ”Features a sleek black plumage with a notable white stripe above the eyes”,
#             • ”Tail longer than average, aiding in agile flight maneuvers”,
#             • ”Exhibits a unique toe arrangement for perching”,
#             • ”Zygodactyl feet allow for gripping branches and surfaces with ease”
#             ],
#             ”Crested Auklet”: [
#             – ”Adapted to cold climates with a dense feather coat”,
#             – ”Utilizes its black bill for feeding in marine environments”,
#             – ”Legs and feet are uniformly black, matching its dark plumage”,
#             – ”Black body adorned with white spots, matched with black legs and feet”,
#             – ”Small seabird with distinctive white crest on black head”,
#             – ”Small, black and white seabird with a unique white crest on its head”
#             ]
#             "}

#         ]

#     return messages

from loading_helpers import compute_class_list
import json
import concurrent.futures
from openai import OpenAI
client = OpenAI()

subcategories = {}
count = 0

def get_subcategory(class_name):
    try:
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
                    "text": f"Which of the subcategories in the above Python list should {class_name} be assigned to? It must be one of the subcategories given, not a new one. Respond with only the subcategory name."
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
        print(f"Processed: {class_name} -> {subcategory}")
        return class_name, subcategory
    except Exception as e:
        print(f"Error processing {class_name}: {e}")
        return class_name, None


filename = 'descriptors/gpt3/descriptors_imagenet.json'

with open(filename, 'r') as f:
    data = json.load(f)

class_list = compute_class_list(data, sort_config=False)


with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_class = {executor.submit(get_subcategory, class_name): class_name for class_name in class_list}

    for future in concurrent.futures.as_completed(future_to_class):
        class_name, subcategory = future.result()
        if subcategory:
            if subcategory in subcategories:
                subcategories[subcategory].append(class_name)
            else:
                subcategories[subcategory] = [class_name]



class_filename = 'class_analysis/json/openai_class_analysis_imagenet.json'

with open(class_filename, 'w') as f:
    json.dump(subcategories, f, indent=4)

print("Process completed.")