import time
from anthropic import AnthropicBedrock
import numpy as np

CLAUDE_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"

def transform2list(errors): # used to transform the mistake string to a list
    llist = []
    split_errors = errors.split("\n")
    n = 1
    for ex in split_errors:
        if len(ex) > 1:
            try:
                sample = ex.split(f"{n}. ")[1]
                llist.append(sample)
                n += 1
            except:
                print(f"{ex} not splittable")
    return llist

def api_call(prompt, msgs, model, retries=10):
    region = "us-east-1"
    client = AnthropicBedrock(
        aws_region=region,
        # aws_access_key=access_key,
        # aws_secret_key=secret_key,
        # aws_session_token=session_token,
    )
    for attempt in range(retries):
        try:
            message = client.messages.create(model=model,max_tokens=2156,system=prompt,messages=msgs)
            return message
        except Exception as e:
            print(f"Attempt {attempt + 1} failed:{e}")
            if attempt < retries - 1:
                wait_time = np.random.randint(20, 40)
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Exiting.")
    return None