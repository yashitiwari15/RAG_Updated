import requests
import pandas as pd

# List of files
files = ['2.pdf', '3.pdf', '4.pdf', '5.pdf', 
         '6.pdf', '7.pdf', '8.pdf', '9.pdf', '10.pdf']

# List of questions (one at a time, as per your API structure)
questions = [
    "Q1.What is the type of this contract Document?",
    "Q2.What is the name of the landlord? Which party is the landlord?",
    "Q3.What is the name of the tenant? Which party is the tenant?",
    "Q4.What is the asset leased in this contract?",
    "Q.5What is the address of the asset leased?",
    "Q6.Area (Square Metres)",
    "Q7.What is the contract start date?",
    "Q8.What is the contract end date?",
    "Q9.What is the rental amount?",
    "Q.10What is the rent free period?",
    "Q11.What is the security deposit",
    "Q12.What is the break clause?"
]

# Initialize a list to store the results
results = []

# API endpoint URL
url = 'http://127.0.0.1:5001/query'  # Replace with your API URL
headers = {
    'Content-Type': 'application/json'
}
print("Starting")
# Loop through each question
for question in questions:
    # Loop through each file for the current question
    print(question)
    for file in files:
        # Payload containing the question and file
        payload = {
            "query": question,
            "filenames": [file]  # Sending a single file at a time
        }

        # Make the API request
        response = requests.post(url, json=payload, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()  # Parse the JSON response
            answer = result.get("ans", "No answer provided")  # Extract the 'ans' field
            
            # Store the result (file name, question, and answer)
            results.append({
                'file_name': file,
                'question': question,
                'answer': answer
            })
        else:
            print(f"Error with {file}: {response.status_code}")

# Convert the results to a DataFrame for easier handling
df = pd.DataFrame(results)

# Save the results to a CSV or Excel file
df.to_csv('api_answers.csv', index=False)
# Alternatively, save as an Excel file
# df.to_excel('api_answers.xlsx', index=False)

print("API calls completed and answers saved!")
