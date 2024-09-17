import os
from openai import OpenAI
from dotenv import load_dotenv
import time


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

start_time = time.time()


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
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
f"""Use this method to generate visual descriptors for the following class from the CUB dataset. The descriptors should be purely for visual features. Do not write the code or methods for generating the descriptors, just generate the descriptors in JSON format (without stating the format type, as this will be directly saved into a JSON file). Generate them in the following example JSON format, where the keys are classes and values are descriptors:
"Sooty Albatross": [
\t- "Webbed feet support its aquatic lifestyle",
\t- "Has a distinctive white band around its neck, setting it apart from similar species",
\t- "Dark-plumaged large bird with a prominent white neck band",
\t- "Characterized by long, hooked bill and long, narrow wings for sustained flight",
\t- "Features a deep, sooty gray plumage, blending into the ocean's hues",
\t- "Employs long, narrow wings for efficient soaring across vast ocean distances"
],
"Groove-billed Ani": [
\t- "Boasts a unique groove-billed beak, distinct among bird species",
\t- "Distinguished by its long, curved bill and black plumage",
\t- "Features a sleek black plumage with a notable white stripe above the eyes",
\t- "Tail longer than average, aiding in agile flight maneuvers",
\t- "Exhibits a unique toe arrangement for perching",
\t- "Zygodactyl feet allow for gripping branches and surfaces with ease"
],
"Crested Auklet": [
\t- "Adapted to cold climates with a dense feather coat",
\t- "Utilizes its black bill for feeding in marine environments",
\t- "Legs and feet are uniformly black, matching its dark plumage",
\t- "Black body adorned with white spots, matched with black legs and feet",
\t- "Small seabird with distinctive white crest on black head",
\t- "Small, black and white seabird with a unique white crest on its head"
]

Generate descriptors for the following class: "Black-footed Albatross"
"""
            }
        ],
    temperature=0.0,
    max_tokens=200
)


print(response.choices[0].message.content)




end_time = time.time()

print(f"Time taken: {end_time - start_time:.2f} seconds")