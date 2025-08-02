import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from load import set_hparams
from loading_helpers import compute_class_list, load_json
import itertools
from descriptor_strings import stringtolist


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
# openai.api_key = os.getenv("OPENAI_API_KEY")
# # Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

hparams, _, _, _, _, gpt_descriptions, unmodify_dict, label_to_classname, n_classes = set_hparams(model_size='ViT-B/32', desc_type='gpt-3', dataset='food101', method='clip')
filename = hparams['descriptor_fname'] + '.json'

def generate_prompt(class_name: str):

    class_name = class_name.replace('_', ' ')

    prompt = f"""Q: What are useful visual features for distinguishing a lemur in a photo?
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

Q: What are useful features for distinguishing a {class_name} in a photo?
A: There are several useful visual features to tell there is a {class_name} in a photo:
-
"""

    messages = [
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
                    "text": prompt
                    }
                ]
                }
            ]
    
    return messages

# generator 
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))

def obtain_descriptors_and_save(filename):
    responses = {}
    descriptors = {}

    with open(filename, 'r') as f:
        descriptor_data = json.load(f)

    # Load the class list
    class_list = compute_class_list(descriptor_data, sort_config=True)

    prompts = [generate_prompt(class_name) for class_name in class_list]

    responses = [ client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=prompt_partition,
        response_format={
            "type": "text"
        },
        temperature=0,
        max_tokens=100
        ) for prompt_partition in partition(prompts, 20) ]

    response_texts = [r["choices"][0]["message"]["content"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]

    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    output_filename = f"{filename}_test.json"

    # save descriptors to json file
    if not output_filename.endswith('.json'):
        output_filename += '.json'
    with open(output_filename, 'w') as fp:
        json.dump(descriptors, fp)

    print(f"Descriptors saved to {output_filename}")

    return descriptors

descriptor_test = obtain_descriptors_and_save(filename)
print("Generated Descriptors:", descriptor_test)