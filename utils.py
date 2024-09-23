import re
import json
import random


def generate_random_distribution(k):
    random_numbers = [random.random() for _ in range(k)]
    sum_of_numbers = sum(random_numbers)

    # Normalize the numbers to make their sum equal to 1
    return [x / sum_of_numbers for x in random_numbers]


def find_answer_from_string(s):
    if s in ['1', '2', '3', '4', '5', 1, 2, 3, 4, 5]:
        return int(s)

    # Define the pattern for special characters
    pattern = re.compile(r'[12345]')
    # Search for the first special character
    match = pattern.search(s)
    # Return the special character if found, otherwise return None

    return int(match.group()) if match else None


def save_to_json(data, output_fn):
    with open(output_fn, 'w') as F:
        json.dump(data, F, indent=4)
    print(f'Data of size {len(data)} saved to file: {output_fn}')
