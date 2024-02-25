import sys
import yaml

# Reading input from command line
input_string = sys.argv[1]

# Splitting the input string by '|' to get individual components
components = input_string.split('|')

# Assigning the components to the respective keys in a dictionary
output_dict = [{
    "target": components[0],
    "positive": components[1],
    "unconditional": components[2],
    "neutral": components[0],  # Assuming neutral is the same as target
    "action": "enhance",
    "guidance_scale": 3,
    "resolution": 512,
    "dynamic_resolution": False,
    "batch_size": 12
}]

# Writing the dictionary to 'data/prompts-xl.yaml'
with open('data/prompts-xl.yaml', 'w') as file:
    yaml.dump(output_dict, file)

print("Data saved to 'data/prompts-xl.yaml'")

