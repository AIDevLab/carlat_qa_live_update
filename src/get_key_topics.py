from fastapi import FastAPI
import json 
import requests

from openai import OpenAI


few_shots_samples = open("few_shots_samples/few_shots_samples.txt",'r', encoding='utf-8')
system_prompt = "prompts\system_prompt.txt"

client = OpenAI()

print(completion.choices[0].message)

def chunk_transcript(transcript, delimiter, nbr_tokens):
    chunks = []
    current_chunk = ""
    tokens_count = 0

    # Split the transcript into tokens using the delimiter
    tokens = transcript.split(delimiter)

    for token in tokens:
        # Check if adding the current token exceeds the maximum number of tokens
        if tokens_count + len(token.split()) <= nbr_tokens:
            # Add the token to the current chunk
            current_chunk += token + delimiter
            # Update the tokens count
            tokens_count += len(token.split())
        else:
            # Add the current chunk to the list of chunks
            chunks.append(current_chunk.strip())
            # Reset the current chunk and tokens count
            current_chunk = token + delimiter
            tokens_count = len(token.split())

    # Add the last chunk to the list of chunks
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def prepare_input(few_shots_samples, transcript_chunk):

few_shots_path = "few_shots_samples/"
prompts_path = "prompts/"
app = fastapi()
@app.post('/get_key_topics')
def get_key_topcis(data):
    transcript = data.get(transcript)

    # get the trqnscript chunks 
    nbr_tokens = 2000
    chunks = chunk_transcript(transcript, "\n", nbr_tokens)

    draft =""
    for chunk in chunks:
        # summarize each chunk using few shots learning 
        llm_input = prepare_input(few_shots_samples, chunk)

        #call llm with the processed input 
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": llm_input}
        ]
        )

        # concat the sub-results
        draft = draft +"\n" + completion.choices[0].message 

    return draft 


