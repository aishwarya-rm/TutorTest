'''
Process the conversation, and for each turn of the conversation, classify which state a student is in.
'''
import pickle
import numpy as np
from utils import *
import tqdm
import re

def label_conversation_states(sc):
    turns = sc.split('Student: ')[1:] # The first split is a space
    dialogue = ""
    states = [] # 0 is distracted, 1 is focused
    for t in turns:
        dialogue += "Student: "
        dialogue += t
        q_prompt = (
            f"We are evaluating a dialogue between an AI tutor and a human student. They are working to solve a math problem. "
            f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the student’s utterances are "
            f"prefaced by ”Student:”. Here is the dialogue: {dialogue}\n\nBased on the most recent exchange"
            f" between the student and the tutor, does the student appear to be focused? Note that a student can be focused"
            f"earlier in the conversation and then become distracted later in the conversation and likewise. "
            f"Answer <<yes>> or <<no>> and explain your reason. "
            f"Make sure to format your answer as <<yes>> or <<no>>. If you're unsure, answer"
            f" <<unknown>>. Let's take a deep breath and think carefully.")
        msgs = [{"role": "user", "content": q_prompt}, {"role": "assistant", "content": "Student:"}]
        message = api_call(q_prompt, msgs, CLAUDE_SONNET)
        response = message.content[0].text
        # print("Dialogue: ", dialogue)
        # print(response)
        label = 'NA'
        if "<<no>>" in response:
            label = 'd' # student is distracted
        elif "<<yes>>" in response:
            label = 'f' # student is focused
        states.append(label)
    return states
def learn_P(all_states):
    '''
    @param all_states: all labeled states in the conversations from comta dataset
    @return: P, which is a probability matrix.
    '''
    count_P = {'f':{'d':0, 'f':0, 'NA': 0}, 'd':{'d':0, 'f':0, 'NA':0}, 'NA':{'d':0, 'f':0, 'NA':0}}
    nC = len(all_states)
    for c in range(nC):
        states = all_states[c]
        nS = len(states)
        for i in range(1, nS):
            curr_state = states[i]
            prev_state = states[i-1]
            count_P[prev_state][curr_state] += 1

    # Calculate the probability
    P = np.zeros((3, 3)) # There are no examples of distracted/focused/NA after an NA state? It might be only the last state (there's only 1 possible case of this)

    # p(distracted|focused)
    P[0][0] = count_P['f']['d'] / (count_P['f']['d'] + count_P['f']['f'] + count_P['f']['NA'])
    # p(focused|focused)
    P[0][1] = count_P['f']['f'] / (count_P['f']['d'] + count_P['f']['f'] + count_P['f']['NA'])
    # p(NA|focused)
    P[0][2] = count_P['f']['NA'] / (count_P['f']['d'] + count_P['f']['f'] + count_P['f']['NA'])

    # p(distracted|distracted)
    P[1][0] = count_P['d']['d'] / (count_P['d']['d'] + count_P['d']['f'] + count_P['d']['NA'])
    # p(focused|distracted)
    P[1][1] = count_P['d']['f'] / (count_P['d']['d'] + count_P['d']['f'] + count_P['d']['NA'])
    # p(NA|distracted)
    P[1][2] = count_P['d']['NA'] / (count_P['d']['d'] + count_P['d']['f'] + count_P['d']['NA'])

    # p(distracted|NA) --> all very unlikely states to reach, but we have an uniformative prior on focused and distracted.
    P[2][0] = 0.5 #count_P['NA']['d'] / (count_P['NA']['d'] + count_P['NA']['f'] + count_P['NA']['NA'])
    # p(focused |NA)
    P[2][1] = 0.5 #count_P['NA']['f'] / (count_P['NA']['d'] + count_P['NA']['f'] + count_P['NA']['NA'])
    # p(NA|NA)
    P[2][2] = 0. #count_P['NA']['NA'] / (count_P['NA']['d'] + count_P['NA']['f'] + count_P['NA']['NA'])
    return P

def label_conversation_actions(sc):
    turns = sc.split('Student: ')[1:]
    actions = []
    options = ('1. Question\nPurpose: To guide the student towards a solution without directly providing an answer.'
                   '\nExamples:\n“What do you remember about [related topic]?”\n“What do you think would happen if you tried using the distributive property here?'
               '    ”\n\n2. Hint/Information Revel\nPurpose: To help a student who is struggling with the concept to progress to the next step of solving the problem. '
                   '\nExamples:\n"Here is a suggestion for a next step: [suggestion]"\n"Try focusing on the units in this problem. What do the units tell you about what you\'re solving for?"\n\n3. Correction '
                    '\nPurpose: To help the student identify and learn from mistakes.\nExamples:\n“Let’s '
                   'review this step—where do you think we might have gone wrong?”\n“You\'ve got the right idea, but the way you factored this'
               '     quadratic is incorrect. Let\'s revisit the factoring process.”\n\n4. Confirmation \nPurpose: Affirm that the student is on the right track, encouraging'
               'them to continue while reinforcing their understanding. \nExamples:\n“Yes, that’s correct! Well done.”\n“That’s right! You understood the difference'
               's between velocity and acceleration perfectly. Now, let’s move on to applying this concept.”')
    for t in turns:
        dialogue = "Student: " + t # We don't consider any long-term observations for the tutor action, so we just need to use the existing turn.
        q_prompt = (
            f"We are evaluating a dialogue between an AI tutor and a human student. They are working to solve a math problem. "
            f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the student’s utterances are "
            f"prefaced by ”Student:”. Here is one turn of their dialogue: {dialogue}\n\n."
            f"Based on the dialogue, can you classify the tutor's utterance as one of the five following question types? "
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
        # print("Response: " + str(response) + "\n\n\n")
        pattern = r'<<([^>]*)>>'
        matches = re.findall(pattern, response)
        if len(matches) > 0:
            try:
                number = int(matches[0])
            except ValueError:
                number = -1
                print("Input was invalid; defaulting to -1 for MA...")
            actions.append(number)
        else:
            # label not found
            print("Label not found .... replace with -1 for NA")
            actions.append(-1)
    return actions

if __name__ == '__main__':
    examples = pickle.load(open("offline_data/offline_few_shot_exs_10282024.pkl", "rb")) # Offline data of khan academy
    all_states = []
    all_actions = []
    for kk, sc in enumerate(tqdm.tqdm(examples)): # There are only 11 examples of conversations here if we restrict ourselves to just algebra questions
        states = label_conversation_states(sc)
        actions = label_conversation_actions(sc)
        if len(states) != len(actions):
            print("States and actions not the same length")
        all_states.append(states)
        all_actions.append(actions)
        pickle.dump(all_states, open("offline_data/labeled_states_10282024.pkl", "wb"))
        pickle.dump(all_actions, open("offline_data/labeled_actions_10282024.pkl", "wb"))

    import ipdb; ipdb.set_trace()

