import os
import openai
from dotenv import load_dotenv
import json
import time

import itertools

from descriptor_strings import stringtolist

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

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
-"""

# generator
def partition(lst, size):
    for i in range(0, len(lst), size):
        yield list(itertools.islice(lst, i, i + size))
        
def obtain_descriptors_and_save(filename, class_list):
    responses = []
    descriptors = {}

    prompts = [generate_prompt(category.replace('_', ' ')) for category in class_list]
    
    # responses = [openai.Completion.create(model="gpt3.5-turbo-instruct",
    #                                         prompt=prompt_partition,
    #                                         temperature=0.,
    #                                         max_tokens=100,
    #                                         ) for prompt_partition in partition(prompts, 20)]
    # response_texts = [r["text"] for resp in responses for r in resp['choices']]
    # descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    # descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    for prompt_partition in partition(prompts, 20):
        for attempt in range(4):  # Try up to four times (initial + three retries)
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {
                        "role": "user",
                        "content": str(prompt_partition[0])
                    }
                    ],
                    temperature=0,
                    max_tokens=125,
                )
                responses.append(response.choices[0].message.content)

                print(responses[-1])
                break  # Exit the retry loop if the request was successful
            except openai.error.RateLimitError:
                if attempt < 3:  # Don't wait after the last attempt
                    time.sleep((2 ** attempt) * 3)  # Exponential backoff: 1s, 2s, 4s
                else:
                    print(f"Request failed after {attempt + 1} attempts")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break  # Exit retry loop if a non-rate-limit error occurs

        # time.sleep(5)  # Wait for 1 second between different prompt partitions to avoid rate limits

    response_texts = [r["text"] for resp in responses for r in resp['choices']]
    descriptors_list = [stringtolist(response_text) for response_text in response_texts]
    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)
    
# # Generate a list of all the classes in the CUB dataset from descriptors_cub.json
# with open('descriptors/descriptors_cub_gpt4.json', 'r') as fp:
#     descriptors = json.load(fp)

# obtain_descriptors_and_save('/home/luke/Documents/GitHub/data/CUB/CUB_200_2011/', list(descriptors.keys())[0:3])