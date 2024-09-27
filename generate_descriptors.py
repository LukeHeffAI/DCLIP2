import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import json
import time
import itertools
from descriptor_strings import stringtolist
from load import hparams
from loading_helpers import compute_class_list, load_json

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_api_content(model, category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    if model.startswith("gpt3.5"):
        messages = f"""Q: What are useful visual features for distinguishing a lemur in a photo?
        A: There are several useful visual features to tell there is a lemur in a photo:
        - four-limbed primate
        - black, grey, white, brown, or red-brown
        - wet and hairless nose with curved nostrils
        - long tail
        - large eyes
        - furry bodies
        - clawed hands and feet

        Q: What are useful visual features for distinguishing a television in a photo?
        A: There are several useful visual features to tell there is a television in a photo:
        - electronic device
        - black or grey
        - a large, rectangular screen
        - a stand or mount to support the screen
        - one or more speakers
        - a power cord
        - input ports for connecting to other devices
        - a remote control

        Q: What are useful features for distinguishing a {category_name} in a photo?
        A: There are several useful visual features to tell there is a {category_name} in a photo:
        -"""

    elif model.startswith("gpt-4"):
        messages = [
            {
                "role": "user", "content":
"""I am a researcher working on updating the experiments performed in the research paper Visual Classification via Description by Menon et al. 2023. Explain the processes and reasoning by which descriptors are generated for each data class in this experiment. Keep the explanation brief and only address the request, do not give extra information."""
            },
            {
                "role": "assistant", "content":
f"""In the experiments performed in the research paper "Visual Classification via Description" by Menon et al. (2023), descriptors for each data class are generated through the following processes:
- Data Collection: Collect natural language descriptions corresponding to each visual class in the dataset. These descriptions capture distinctive attributes of the classes.
- Preprocessing: Standardize and preprocess the collected descriptions to ensure consistency and remove any noise or irrelevant information.
- Encoding: Use a pre-trained language model (such as CLIP) to encode the processed descriptions into high-dimensional vector representations. This step involves feeding the descriptions into the model to obtain embed-dings that capture the semantic meaning.
- Aggregation: Aggregate the vector representations of all descriptions for each class to form a single descriptor. This can be done by averaging the embeddings of multiple descriptions to ensure a comprehensive representation of each class. The reasoning behind generating descriptors in this way is to leverage natural language descriptions to encapsulate the semantic attributes of each class, thereby enabling the model to perform classification tasks based on these descriptive features."""
            },
            {
                "role": "user", "content": 
f"""Use this method to generate visual descriptors for the following class from the {hparams['dataset_name']} dataset. The descriptors should be purely for visual features, focusing on what makes this class distinct from others in the dataset. Do not write the code or methods for generating the descriptors, just generate the descriptors. Generate them in the following example JSON format, without specifying file type in your output, where the keys are classes and values are descriptors:

"Class 1 Name": ["Descriptor 1", "Descriptor 2", "Descriptor 3", "Descriptor 4", "Descriptor 5", "Descriptor 6"],
"Class 2 Name": ["Descriptor 1", "Descriptor 2", "Descriptor 3", "Descriptor 4", "Descriptor 5", "Descriptor 6"],
"Class 3 Name": ["Descriptor 1", "Descriptor 2", "Descriptor 3", "Descriptor 4", "Descriptor 5", "Descriptor 6"]

Generate descriptors for the following class, preserving the class name exactly: \"{category_name}\""""
            }
        ]

    return messages, category_name


# Generator function
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))

def obtain_descriptors_and_save(filename, model="gpt-4o"): 
    try:
        with open(filename, 'r') as fp:
            descriptors = json.load(fp)  # Load existing data
    except FileNotFoundError:
        print(f"File not found: {filename}, creating new file")
        descriptors = {}
    
    dataset = load_json(hparams['descriptor_fname'])
    class_list = compute_class_list(dataset)

    # Generate prompts along with corresponding class names
    prompts = [generate_api_content(model=model, category_name=category.replace('_', ' ')) for category in class_list]

    count = 0
    for prompt_partition in partition(prompts, 20):
        for prompt, category_name in prompt_partition:  # Unpacking the tuple (messages, category_name)
            for attempt in range(4):  # Retry up to 4 times
                try:
                    # Send API request
                    response = client.chat.completions.create(
                        model=model,
                        messages=prompt,
                        temperature=0.0,
                        max_tokens=220,
                        response_format={
                            "type": "json_object"
                        }
                    )

                    response_content = str(response.choices[0].message.content)
                    count += 1
                    try:
                        # Convert response string to JSON
                        json_response = json.loads(response_content)
                        # print(str(json_response.keys()).split("'")[1], "\n\t", json_response[f'{category_name}'])

                        # Append the response to the descriptors dictionary
                        descriptors[category_name] = [descriptor.lower() for descriptor in json_response[f'{category_name}']]
                        print(f'Class {count}: ', category_name, descriptors[category_name])
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                    break  # Exit retry loop if successful

                except openai.RateLimitError:
                    if attempt < 3:  # Retry with exponential backoff
                        time.sleep((2 ** attempt) * 3)
                    else:
                        print(f"Request failed after {attempt + 1} attempts")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    break  # Exit retry loop for non-rate-limit errors

    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp, indent=4)

    return descriptors

filename = f'descriptors/{hparams['desc_type']}/descriptors_{hparams['dataset']}.json'

obtain_descriptors_and_save(filename=filename, model="davinci-002")