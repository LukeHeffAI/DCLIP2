import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from load import set_hparams
from loading_helpers import compute_class_list, load_json
import itertools


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
# openai.api_key = os.getenv("OPENAI_API_KEY")
# # Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

hparams, _, _, _, _, gpt_descriptions, unmodify_dict, label_to_classname, n_classes = set_hparams(model_size='ViT-B/32', desc_type='gpt-3', dataset='food101', method='clip')

with open(hparams['descriptor_fname'], 'r') as f:
    descriptor_data = json.load(f)

# Load the class list
class_list = compute_class_list(descriptor_data, sort_config=True)


def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
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
-
"""

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))

def obtain_descriptors_and_save(filename, class_list):
    responses = {}
    descriptors = {}
    
    prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You will respond in the style of a completion model, completing the prompt by continuing in the style given."
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Q: What are useful visual features for distinguishing a lemur in a photo? \nA: There are several useful visual features to tell there is a lemur in a photo: \n- four-limbed primate \n- black, grey, white, brown, or red-brown \n- wet and hairless nose with curved nostrils \n- long tail \n- large eyes \n- furry bodies \n- clawed hands and feet \n \nQ: What are useful visual features for distinguishing a television in a photo? \nA: There are several useful visual features to tell there is a television in a photo: \n- electronic device \n- black or grey \n- a large, rectangular screen \n- a stand or mount to support the screen \n- one or more speakers \n- a power cord \n- input ports for connecting to other devices \n- a remote control \n \nQ: What are useful features for distinguishing a oyster in a photo? \nA: There are several useful visual features to tell there is a oyster in a photo: \n"
                }
            ]
            }
        ],
        response_format={
            "type": "text"
        },
        temperature=0,
        max_tokens=2048
        )
    
    descriptor_set = response.choices[0].message.content.split('\n')

    output_filename = f"{filename}_test.json"

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)
    
    print(f"Descriptors saved to {filename}")

# obtain_descriptors_and_save('example', ["bird", "dog", "cat"])