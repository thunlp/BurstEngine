import re
import csv

# Function to parse the input text
def parse_input_text(text):
    data = {}
    lines = text.strip().split('\n')
    entry = {}
    
    for line in lines:
        if line.startswith('******************'):
            if len(entry) > 0:
                print(entry)
                key = (entry['method'], entry['seqlen'])
                if key not in data:
                    data[key] = {'method': entry['method'], 'seqlen': entry['seqlen'], 'toks_8': '', 'toks_16': '', 'toks_32': ''}
                data[key]['toks_' + entry['sp_size']] = entry['toks']
            entry = {}
        else:
            match = re.match(r'(\w+):\s+(\S+)', line)
            if match:
                key = match.group(1)
                value = match.group(2)
                entry[key] = value
    
    if entry:
        key = (entry['method'], entry['seqlen'])
        if key not in data:
            data[key] = {'method': entry['method'], 'seqlen': entry['seqlen'], 'toks_8': '', 'toks_16': '', 'toks_32': ''}
        data[key]['toks_' + entry['sp_size']] = entry['toks']
    
    return list(data.values())


# Read the input text from a file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='Path to the input file')
args = parser.parse_args()

with open(args.file, 'r') as file:
    print("Reading input file...", args.file)
    input_text = file.read()

# Parse the input text
parsed_data = parse_input_text(input_text)

# Write the parsed data to a CSV file
file = "70b_output.csv"
with open(file, 'w', newline='') as csvfile:
    fieldnames = ['method', 'seqlen', 'toks_8', 'toks_16', 'toks_32']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for entry in parsed_data:
        writer.writerow(entry)

print(f"Data has been successfully written to {file}")
