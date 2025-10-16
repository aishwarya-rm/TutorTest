'''
Goal: get a dictionary of tutor action classification to student response to get few shot examples.
'''
import json
from tkinter.ttk import Label

import numpy as np
import pandas as pd
import os, sys
import pickle
from utils import *
import re

def label_tutor_action(tutor_utterance, sc): # Should use the actual labeled responses as few shot examples
    tutor_example_dict = pickle.load(open("offline_data/tutor_example_dict_cima.pkl", "rb"))
    options = (f'1. Question\nPurpose: To guide the student towards a solution without directly providing an answer.'
               f'\nExamples:\n{tutor_example_dict[1]}\n\n2. Hint/Information Revel\nPurpose: To help a student who is struggling with the concept to progress to the next stage of translation. '
               f'\nExamples:\n{tutor_example_dict[2]}\n\n3. Correction '
               f'\nPurpose: To help the student identify and learn from mistakes.\nExamples:{tutor_example_dict[3]}\n\n4. Confirmation \nPurpose: Affirm that the student is on the right track, encouraging'
               f'them to continue while reinforcing their understanding. \nExamples:\n{tutor_example_dict[4]}')
    q_prompt = (
        f"We are evaluating a dialogue between an AI tutor and a human student. They are working to solve a math problem. "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the student’s utterances are "
        f"prefaced by ”Student:”. Here is the prior conversation they have had: {sc}\n\n."
        f"To respond to the student's last utterance, the tutor makes the following comment: \n[Tutor Response]: {tutor_utterance}\n\n"
        f"Based on the prior conversation and the tutor utterance, can you classify the tutor's utterance as one of the five following question types? "
        f"Your options are:\n{options}. Let's take a deep breath and think carefully.")
    msgs = [{"role": "user",
             "content": "If you think the tutor is asking a question, respond <<1>>. "
                        "If you think the tutor is giving the student a hint or revealing helpful information, respond <<2>>. "
                        "If you think the tutor is correcting the student, respond <<3>>. "
                        "If you think the tutor is confirming the student's procedure, respond <<4>>. "
                        "Choose the closest answer based on the examples for each question type, and your "
                        "observation of the tutor's utterance. Make sure to state your answer in the format << >>."}]
    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    response = message.content[0].text
    pattern = r'<<([^>]*)>>'
    number = -1
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        try:
            number = int(matches[0])
        except ValueError:
            print("Input was invalid; defaulting to -1 for MA...")
    else:
        # label not found
        print("Label not found .... replace with -1 for NA")
    return number

def relabel_conversation(past_conversation):
    labeled_conversation = []
    for i in range(len(past_conversation)):
        utterance = past_conversation[i]
        if i % 2 == 0:
            labeled_conversation.append("Student: " + utterance)
        else:
            labeled_conversation.append("Tutor: " + utterance)
    return labeled_conversation

if __name__ == '__main__':

    with open('offline_data/cima_dataset.json') as f:
        d = json.load(f)
        example_dict = {1: [], 2: [], 3: [], 4: []} # Student responses after a tutor action with each of these categories
        for i in range(1134):
            past_conversation = d['prepDataset'][str(i)]['past_convo'] # Full past conversation
            # Reformat the past conversation to add labels:
            sc = relabel_conversation(past_conversation)
            for i in range(len(past_conversation)-1):
                if i % 2 == 1: # Tutor utterance
                    conversation_until_now = "".join(sc)
                    action = label_tutor_action(past_conversation[i], conversation_until_now)
                    if action != -1:
                        student_utterance = past_conversation[i+1]
                        example_dict[action].append(student_utterance) # How students respond to an utterance with the class of actions.
            pickle.dump(example_dict, open("offline_data/student_example_dict_cima.pkl", 'wb'))
        import ipdb; ipdb.set_trace()






