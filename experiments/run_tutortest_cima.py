'''CIMA dataset '''
from distutils.errors import LinkError

import tqdm
import re
import json
import numpy as np
import random
from generate_practice_questions_khan_academy import transform_practice_problems
from utils import *
import pickle


first_object = ['dog', 'cat', 'plant', 'bunny', 'ball', 'bag', 'box']
prepositional_phrases = ['is in front of', 'is next to', 'is inside of', 'is under', 'is behind', 'is on top of']
colors = ['green', 'pink', 'blue', 'yellow', 'red', 'purple']
second_object = ['tree', 'box', 'plant', 'cat', 'bunny', 'dog', 'bed', 'table', 'bag']
italian_first_object = ['il cane', 'il gatto', 'la pianta', 'il coniglietto', 'la palla', 'la borsa', 'la scatola']

def generate_practice_problems(prepositional_phrase, noun_english, noun_italian):
    random_second_object = random.choice(second_object)
    random_color = random.choice(colors)
    preface = f"You are given an image of a {noun_english} that {prepositional_phrase} a {random_color} {random_second_object}. "
    goal = "Your goal is to describe this image in Italian. Specifically, use the correct proposition and translation of the second object to complete the following sentence: \n"
    translation_sentence = f"[Sentence]:  {noun_italian} (blank) \n\n Include the full sentence in your answer. For example, your answer should start with `{noun_italian}`.  "
    practice_problem = preface + goal + translation_sentence
    correct_answer = "the " + noun_english + " " + prepositional_phrase + " the " + random_color + " " + random_second_object
    return practice_problem, correct_answer

def generate_task(prepositional_phrase, noun_english, second_noun, color, noun_italian):
    preface = f"You are given an image of a {noun_english} that {prepositional_phrase} a {color} {second_noun}. "
    goal = "Your goal is to describe this image in Italian. Specifically, use the correct proposition and translation of the second object to complete the following sentence: \n"
    translation_sentence = f"[Sentence]:  {noun_italian} (blank) \n\n Include the full sentence in your answer. For example, your answer should start with `{noun_italian}`.  "
    task = preface + goal + translation_sentence
    return task

def decode_action_cima(action_idx):
    idx_to_text = {1: "ask a question", 2: "give the student a hint or reveal important information about the problem",
                   3: "correct the student", 4: "encourage the student in their progress", -1: "respond without a specific goal"}
    return idx_to_text[action_idx]

def adaptive_good_policy(sc, s):
    if s == 'e':
        cognitive_error_sentence = "Currently, the student has a cognitive error."
    elif s == 'ne':
        cognitive_error_sentence = "Currently, the student does not have a cognitive error. "
    q_prompt = (f'You are an online tutor working with student that is learning Italian. In the dialogue'
                f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                f'``Student:". \n\n[Dialogue]\n{sc}. {cognitive_error_sentence} Based on the student\'s '
                f'state, what is the best action to take? Your options are: 1. ask a question either 2. give the student a hint, 3. correct a '
                f'prior step of the problem derivation, and 4. encourage the student and confirm their process. ')

    msgs = [{"role": "user",
             "content": "If you think you should ask a question, respond <<1>>. "
                        "If you think you should give a hint, respond <<2>>. "
                        "If you think you should provide a correction <<3>>. "
                        "If you think you should encourage the student, respond <<4>>. "
                        "Make sure to state your answer in the format << >>."}]

    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    response = message.content[0].text
    pattern = r'<<([^>]*)>>'
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        try:
            number = int(matches[0])
        except ValueError:
            number = -1
            print("Input was invalid; defaulting to -1 for MA...")
    else:
        number = -1
        print("Label not found .... replace with -1 for NA")
    return number

def filter_examples_tutor(dataset):
    example_dict = {1: [], 2: [], 3: [], 4: [], -1: []}
    for i in range(1134):
        tutor_responses = dataset['prepDataset'][str(i)]['tutorResponses']
        response_classification = dataset['prepDataset'][str(i)]['tutorActions']
        for j in range(len(response_classification)):
            idx = response_classification[j].index(True) + 1
            if idx == 5:
                idx = -1
            example_dict[idx].append(tutor_responses[j])
    return example_dict

def bad_tutor_response(a, curr_conv, tutor_example_dict):
    example_responses = tutor_example_dict[a]
    a = decode_action_cima(a)
    q_prompt = (f'You are an online tutor working with a student that is learning italian. The student\'s first language is English, '
                f'so most of the instruction should be in English. In the dialogue'
                f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                f'``Student:". \n\n[Dialogue]\n{curr_conv}. Generate a response'
                f'to the student\'s last utterance in the role of an online tutor. Specifically, {a}. Here are examples'
                f'of how an online tutor will interact with a student to {a}. \n\n[Examples]\n{example_responses}.Keep'
                f'your response brief and do not reveal the answer to the problem that the student is trying to work on. Additionally, try to be goal oriented in your response. For example, '
                f'if you said in a prior utterance that you want to give the student another practice problem, actually provide a practice problem.')
    msgs = [{"role": "user", "content": q_prompt}, {"role": "assistant", "content": "Tutor:"}]
    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    response = message.content[0].text
    return response

def good_tutor_response(a, curr_conv, solving_strategy, tutor_example_dict):
    example_responses = tutor_example_dict[a]
    a = decode_action_cima(a)
    q_prompt = (f'You are an online tutor working with a student that is learning italian. The student\'s first language is English. In the dialogue'
                f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                f'``Student:". \n\n[Dialogue]\n{curr_conv}. Respond '
                f'to the student\'s last utterance in the role of an online tutor. Keep in mind that you are in the role of a teacher and that the instruction'
                f'should be primarily in English since that is the student\'s first language. Now, respond to the student\'s last utterance. Specifically, {a}. Here are examples'
                f'of how an online tutor will interact with a student to {a}. \n\n[Examples]\n{example_responses}. Keep'
                f'your response brief and do not reveal the answer to the problem that the student is trying to work on. Additionally, try to be goal oriented in your response. For example, '
                f'if you said in a prior utterance that you want to give the student another practice problem, actually provide a practice problem.'
                f'Finally, the student is likely to have a cognitive error when they begin'
                f'their interaction with you. Your goal is to help them overcome this cognitive error, without knowing the actual cognitive error. Ask the student questions to help you understand'
                f'their cognitive error, and use well-known strategies to help them overcome this cognitive error. This may involve explaining other ways to view the translation problem at hand'
                f'and using alternative examples. For example, this student requires one of the following strategies to help them overcome their cognitive error: \n[Strategy]: \n {solving_strategy} ')
    msgs = [{"role": "user", "content": q_prompt}, {"role": "assistant", "content": "Tutor:"}]
    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    response = message.content[0].text
    return response

def generate_mistakes(task, verbose=False): # For right now, ask an LLM to produce possible mistakes for each of these initial conversations
    mistake_prompt = (f"We are designing a task when an AI tutor is helping a student learn the Italian language. "
                      f"The student is currently working on the following problem. \n [Problem]: {task} \n\n"
                      f"Can you provide a list of 2 different cognitive errors that the student could have where "
                      f"the tutor would need to change their teaching content for different mistakes? Reference the "
                      f"appropriate literature that cites these cognitive errors as relevant to the problem domain. "
                      f"Begin your generation with `List:`.")
    msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "List:"}]
    message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
    mistakes = message.content[0].text
    if verbose:
        print("Problem: " + str(task))
        print("Mistakes: " + str(mistakes) + "\n")
    mistake_list = transform2list(mistakes)
    return mistake_list

def generate_solution(cognitive_error, task):
    mistake_prompt = (
        f"We are designing a task when an AI tutor is helping a student learn the Italian language. "
        f"The student is currently working on the following problem. \n [Problem]: {task} \n\n"
        f"The student may have cognitive errors that require the tutor to personalize their teaching content. Here is one of "
        f"the cognitive errors that a student may have: \n[Cognitive Error]:\n {cognitive_error}\n. If a student had this"
        f"cognitive error, can you identify 2 ways that an online tutor could help the student overcome this cognitive"
        f"error? This can be a suggestion about a way that the online tutor can change their teaching content or "
        f"key information that should be communicated to the student. Make your suggestions very specific to "
        f"the cognitive error from earlier and the problem that the student is working on. "
        f"Recall that the setting is online tutoring, which "
        f"means that visual cues are difficult to employ. Begin your generation with `List:`")
    msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "List:"}]
    message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
    solution = message.content[0].text
    solution_list = transform2list(solution)
    return solution_list

def tutor_analysis(curr_conv, cognitive_error, solving_strategy):
    prompt = (
        f"We are evaluating a conversation between an online tutor and a student learning Italian. "
        f"In the dialogue below, the the tutor's are prefaced by ``Tutor:'' and the student\'s utterances are "
        f"prefaced by ``Student:''. \n\n[Dialogue]\n{curr_conv}. The student currently has the following cognitive error: "
        f"[Cognitive error]\n{cognitive_error}\n. "
        f"According to experts, to help the student overcome this error, the tutor must use at least one of the following "
        f"strategies: [Strategies]\n{solving_strategy}. Based"
        f"on the dialogue from earlier, has the tutor employed any of these strategies to help the student overcome their "
        f"cognitive error? Respond with a <<yes>> or <<no>> depending on whether or not the tutor has used the strategies provided, "
        f"and provide a justification. Use the following format: [Format]\n <<answer>> [Justification]\n."
        f"In the formatted response, answer is either yes or no, and Justification is a space to provide reasoning for your answer.  ")
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": "Judge:"}]
    message = api_call(prompt, msgs, model=CLAUDE_SONNET)
    response = message.content[0].text
    if verbose:
        print(str(response))
    tutor_identified_error = False
    pattern = r'<<\s*(yes|no)\s*>>'
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        try:
            tutor_identified_error = (matches[0] == 'yes')
        except:
            print("Matches: " + str(matches))
    else:
        print("no matches, defaulting to False")
    return tutor_identified_error

def student_response_cognitive_error(a, curr_conv, cognitive_error, student_example_dict):
    if a != -1:
        examples = student_example_dict[a]
    else:
        examples = []
    cog_sentence = (f"Since you are still learning, you have a cognitive error that causes you to make mistakes and ask questions as you are learning."
    f"You have the following cognitive error: {cognitive_error}. This cognitive error affects how you respond to practice problems and your general"
    f"understanding of the Italian language. Respond to the tutor's last utterance while conditioning on having this cognitive error. Keep in mind that you should maintain this"
    f"cognitive error throughout your interactions with the tutor. This cognitive error may cause you to incorrectly conjugate verbs, or incorrectly translate words"
    f"into Italian. For example, be consistent across responses as you continue this conversation. You might also be slow to pick up new material. "
    f"Do not include details about this cognitive error in the response.")
    mistake_prompt = (
        f"You are a student that is asking an online tutor for help with a problem. Your first language is English"
        f"and you should respond primarily in English to the tutor. The following is an excerpt of your dialogue so far.  "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are "
        f"prefaced by ”Student:”. Here is the dialogue: {curr_conv} \n\n {cog_sentence} "
        f"Occasionally, your tone may be informal or abrupt. Here are examples of how other"
        f"students respond to the same category of tutor response:  \n\n[Examples]\n{examples}. Remember to respond "
        f"primarily in English since you do not have a good grasp of the Italian language yet. Begin your response "
        f"with Student:. ")
    msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "Student:"}]
    message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
    mistake = message.content[0].text
    return mistake

def student_response_no_error(a, curr_conv, cognitive_error, student_example_dict):
    if a != -1:
        examples = student_example_dict[a]
    else:
        examples = []
    cog_sentence = (f"Since you are still learning, you had a cognitive error at the beginning of your interaction with "
                    f"the tutor. This error was: {cognitive_error}.")
    mistake_prompt = (
        f"You are a student that is asking an online tutor for help with a problem. Your first language is English"
        f"and you should respond primarily in English to the tutor. The following is an excerpt of your dialogue so far.  "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are "
        f"prefaced by ”Student:”. Here is the dialogue: {curr_conv}\n\n {cog_sentence}"
        f"Respond to the tutor's last utterance conditioning on overcoming any cognitive error if you had errors. "
        f"Make sure to reference the tutor's prior statements and provide a productive"
        f"response. Do not include details about your cognitive error in the response. "
        f"Occasionally, your tone may be informal or abrupt. Here are examples of how other"
        f"students respond to the same category of tutor response:  \n\n[Examples]\n{examples}. Remember to respond "
        f"primarily in English. Preface the response with Student:. ")
    msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "Student:"}]
    message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
    mistake = message.content[0].text
    return mistake

def ask_practice_problem(curr_conv, practice_problems, cognitive_error, s, correct_english_answer):
    new_conv = curr_conv
    practice_problem_prompt = (
        f"You are an online tutor that is interacting with a student to help them learn Italian. The following is an excerpt of your dialogue so far. \n\n[Dialogue]\n{curr_conv}. "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are prefaced by ”Student:” To "
        f"better understand the student's grasp of the topic, you are going to ask a couple of practice problems. These practice problems will tell you what the student knows. Ask the following "
        f"two practice problems to the student. \n\n[Practice Problems]\n{practice_problems}. Provide your question in the following format:\n\n[Format]:\n Here are a few practice problems"
        f"to help me understand how much you have learned so far. Can you try to solve these problems? Here are the problems: [problems]\n In this sentence, problems are practice problems. Make sure to stick to the required format. ")
    msgs = [{"role": "user", "content": practice_problem_prompt}, {"role": "assistant", "content": "Tutor:"}]
    message = api_call(practice_problem_prompt, msgs, model=CLAUDE_SONNET)
    tutor_dialogue = "Tutor: " + message.content[0].text
    if verbose: print(tutor_dialogue)
    new_conv += tutor_dialogue
    if s == 'e':
        if cognitive_error == None:
            cog_sentence = f"Even though you are still learning, you have no cognitive errors and do not make mistakes. "
        elif len(cognitive_error) == 2:
            cog_sentence = (
                f"Since you are still learning, you have two cognitive errors that cause you to make mistakes and ask questions."
                f"You have the following two cognitive errors: {cognitive_error}. These cognitive errors affects how you respond to practice problems and your general"
                f"understanding of the problem domain. Respond to the tutor's last utterance while conditioning on having both cognitive errors. For example, if you have been making mistakes"
                f"so far, continue to make mistakes when you solve the practice problems. Do not include details about these cognitive errors in the response.")
        else:
            cog_sentence = (
                f"Since you are still learning, you have a cognitive error that causes you to make mistakes and ask questions."
                f"You have the following cognitive error: {cognitive_error}. This cognitive error affects how you respond to practice problems and your general"
                f"understanding of the problem domain. Respond to the tutor's last utterance while conditioning on having this cognitive error. For example, if you have been making"
                f"mistakes so far, continue to make mistakes when you solve the practice problems. Do not include details about this cognitive error in the response.")
    else:
        if cognitive_error == None:
            cog_sentence = f"You had no cognitive errors during your interaction with the tutor. "
        elif len(cognitive_error) == 2:
            cog_sentence = f"Since you are still learning, you had two cognitive errors at the beginning of your interaction with the tutor. These errors were: {cognitive_error}. You have now overcome this cognitive error, and learned from your mistakes. "
        else:
            cog_sentence = f"Since you are still learning, you had a cognitive error at the beginning of your interaction with the tutor. This error was: {cognitive_error}. You have now overcome this cognitive error, and learned from your mistakes. "
    student_response_prompt = (
        f"You are a student that is asking an online tutor for help with a problem. The following is an excerpt of your dialogue so far.  "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are "
        f"prefaced by ”Student:”. Here is the dialogue: {new_conv}\n\n {cog_sentence}. The online tutor just asked you to solve two practice problems. Solve these practice problems. "
        f"Preface the response with Student: and use the following format:\n"
        f"[Format]:\n [Problem solving discussion] <<answer1>> [Problem solving discussion] <<answer2>> [Problem solving discussion] \n\n In the format, use the Problem solving discussion portions for adding text justifying your answers. answer1 represents your"
        f"final answer for the first problem and answer2 represents your final answer to the second problem. Be sure to only include your final answer within << and >>. Do not include any intermediate steps."
        f"Make sure to use << and >> around answer1 and answer2, and stick to the format. Only answer1 and answer2 should be within brackets (<< and >>). Occasionally, your tone may be informal or abrupt. ")
    msgs = [{"role": "user", "content": student_response_prompt}, {"role": "assistant", "content": "Student:"}]
    message = api_call(student_response_prompt, msgs, model=CLAUDE_SONNET)
    student_response = "Student: " + message.content[0].text
    if verbose: print(student_response)
    new_conv += student_response

    # Calculate performance based on the correct english answer
    pattern = r'<<\s*([^>]+?)\s*>>'
    answer1 = 0
    answer2 = 0
    matches = re.findall(pattern, student_response)
    if len(matches) > 0:
        try:
            answer1 = matches[0]
        except:
            answer1 = 0
            print("Input: " + str(matches))
        try:
            answer2 = matches[1]
        except:
            answer2 = 0
            print("Input: " + str(matches))
    else:
        print("no matches, defaulting to 'No Answer': " + str(student_response))
    q1_correct = 0
    q1_prompt = (
        f"We are evaluating a translation from English to Italian. The English sentence to be translated is {correct_english_answer[0]}. The translation is {answer1}. Is"
        f"{answer1} a valid translation of the English sentence? Respond with a <<yes>> or <<no>> depending on whether or not the translation is correct, "
        f"and provide a justification. Use the following format: [Format]\n <<answer>> [Justification]\n."
        f"In the formatted response, answer is either yes or no, and Justification is a space to provide reasoning for your answer.  ")
    msgs = [{"role": "user", "content": q1_prompt}, {"role": "assistant", "content": "Judge:"}]
    message = api_call(q1_prompt, msgs, model=CLAUDE_SONNET)
    response = message.content[0].text
    pattern = r'<<\s*(yes|no)\s*>>'
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        try:
            q1_correct = int(matches[0] == 'yes')
        except:
            print("Matches: åå" + str(matches))
    else:
        print("no matches, defaulting to False")

    q2_correct = 0
    q2_prompt = (
        f"We are evaluating a translation from English to Italian. The English sentence to be translated is {correct_english_answer[1]}. The translation is {answer2}. Is"
        f"{answer2} a valid translation of the English sentence? Respond with a <<yes>> or <<no>> depending on whether or not the translation is correct, "
        f"and provide a justification. Use the following format: [Format]\n <<answer>> [Justification]\n."
        f"In the formatted response, answer is either yes or no, and Justification is a space to provide reasoning for your answer.  ")
    msgs = [{"role": "user", "content": q1_prompt}, {"role": "assistant", "content": "Judge:"}]
    message = api_call(q2_prompt, msgs, model=CLAUDE_SONNET)
    response = message.content[0].text
    pattern = r'<<\s*(yes|no)\s*>>'
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        try:
            q2_correct = int(matches[0] == 'yes')
        except:
            print("Matches: " + str(matches))
    else:
        print("no matches, defaulting to False")

    return ((q1_correct + q2_correct)/2) * 100

def generate_transcript_cima(task, cognitive_error, practice_problems, correct_answers, solving_strategy, policy_quality):
    student_example_dict = pickle.load(open("offline_data/student_example_dict_cima.pkl", 'rb'))
    tutor_example_dict = pickle.load(open("offline_data/tutor_example_dict_cima.pkl", 'rb'))
    tutor_identified_cognitive_error = False
    s = 'e'  # cognitive error state
    deterministic_transitions = False
    curr_conv = f"Student:  I'm struggling trying to solve the following problem:\n[Problem]: {task}"
    print(str(curr_conv))
    gamma = 0.98
    horizon = 6
    random_idx = np.random.choice([i for i in range(horizon)])
    for it in range(horizon): # Can increase if necessary
        print("student state: " + str(s))
        a = adaptive_good_policy(curr_conv, s) # The policy is the same, it's just the phrasing that's different?
        print("Chosen action: " + str(a))
        if policy_quality == 'bad':
            tutor_dialogue = "Tutor: " + bad_tutor_response(a, curr_conv, tutor_example_dict)
        elif policy_quality == 'good':
            tutor_dialogue = "Tutor: " + good_tutor_response(a, curr_conv, solving_strategy, tutor_example_dict)
        print(str(tutor_dialogue))
        curr_conv += tutor_dialogue

        # Identify if the tutor said something that can solve the cognitive error in the current conversation
        if not tutor_identified_cognitive_error:
            tutor_identified_cognitive_error = tutor_analysis(curr_conv, cognitive_error, solving_strategy)
        print("tutor identified error: " + str(tutor_identified_cognitive_error))
        # Then change the state and keep it constant
        if tutor_identified_cognitive_error:
            s = 'ne'

        if s == 'e':
            student_response = student_response_cognitive_error(a, curr_conv, cognitive_error=cognitive_error, student_example_dict=student_example_dict)
        elif s == 'ne':
            student_response = student_response_no_error(a, curr_conv, cognitive_error, student_example_dict=student_example_dict)
        response = "Student: " + student_response + "\n"
        print(response)
        curr_conv += response
        if deterministic_transitions:
            if it == random_idx:
                print("asking questions")
                performance = ask_practice_problem(curr_conv, practice_problems, cognitive_error, s, correct_answers)
                return performance * gamma, s
        gamma *= 0.98
    if not deterministic_transitions:
        performance = ask_practice_problem(curr_conv, practice_problems, cognitive_error, s, correct_answers)
        return performance * gamma, curr_conv

if __name__ == '__main__':
    # Filter the CIMA dataset for the tutor examples
    with open('offline_data/cima_dataset.json') as f:
        d = json.load(f)
        tutor_example_dict = filter_examples_tutor(d)
        pickle.dump(tutor_example_dict, open("offline_data/tutor_example_dict_cima.pkl", 'wb'))

    performance_by_policy_fname = ''
    conversations_by_policy_fname = ''

    #Generate the cognitive errors
    verbose=False
    task=None # Pick a specific problem in the CIMA dataset
    cognitive_errors = generate_mistakes(task, verbose=verbose)
    pickle.dump(cognitive_errors, open("offline_data/cognitive_errors_cima_11072024.pkl", "wb"))
    cognitive_errors = pickle.load(open("offline_data/cognitive_errors_cima_11072024.pkl", "rb"))

    performance_by_state = {'e': [], 'ne':[]}
    verbose = False
    # Generate several tasks
    for _, p in enumerate(tqdm.tqdm(prepositional_phrases)):
        for i, o1 in enumerate(first_object):
            for o2 in second_object:
                for c in colors:
                    task = generate_task(p, o1, o2, c, italian_first_object[i])
                    cognitive_errors = generate_mistakes(task, verbose=True)
                    solving_strategies = [generate_solution(cognitive_errors[0], task), generate_solution(cognitive_errors[1], task)]
                    problem1, correct1 = generate_practice_problems(p, o1, italian_first_object[i])
                    problem2, correct2 = generate_practice_problems(p, o1, italian_first_object[i])
                    practice_problems = [problem1, problem2]
                    correct_answers = [correct1, correct2]
                    performance, s = generate_transcript_cima(task=task, cognitive_error=cognitive_errors[0],
                                                              practice_problems=practice_problems,
                                                              correct_answers=correct_answers,
                                                              solving_strategy=solving_strategies[0],
                                                              policy_quality='bad') # We need to randomly select a time to send the practice problems

                    if s == 'e':
                        performance_by_state['e'].append(performance)
                    elif s == 'ne':
                        performance_by_state['ne'].append(performance)
                    pickle.dump(performance_by_state, open("results/practice_problems/cima_large_task_2.pkl", 'wb'))

    # Compare good and bad policy values.
    performance_by_policy = {'good': [], 'bad': []}
    conversations_by_policy = {'good': [], 'bad': []}
    num_turns = 3
    v_good = 0
    v_bad = 0
    verbose = False
    # Generate several tasks
    num_trajectories = 0
    for _, p in enumerate(tqdm.tqdm(prepositional_phrases)):
        for i, o1 in enumerate(first_object):
            for o2 in second_object:
                for c in colors:
                    for _ in range(num_turns):
                        task = generate_task(p, o1, o2, c, italian_first_object[i])
                        cognitive_errors = generate_mistakes(task, verbose=True)
                        solving_strategies = [generate_solution(cognitive_errors[0], task),
                                              generate_solution(cognitive_errors[1], task)]
                        problem1, correct1 = generate_practice_problems(p, o1, italian_first_object[i])
                        problem2, correct2 = generate_practice_problems(p, o1, italian_first_object[i])
                        practice_problems = [problem1, problem2]
                        correct_answers = [correct1, correct2]
                        for ce in range(len(cognitive_errors)):
                            num_trajectories += 1
                            performance_good, good_conv = generate_transcript_cima(task=task, cognitive_error=cognitive_errors[ce],
                                                                      practice_problems=practice_problems,
                                                                      correct_answers=correct_answers,
                                                                      solving_strategy=solving_strategies[ce],
                                                                      policy_quality='good')  # We need to randomly select a time to send the practice problems

                            performance_bad, bad_conv = generate_transcript_cima(task=task, cognitive_error=cognitive_errors[ce],
                                                                           practice_problems=practice_problems,
                                                                           correct_answers=correct_answers,
                                                                           solving_strategy=solving_strategies[ce],
                                                                           policy_quality='bad')  # We need to randomly select a time to send the practice proble
                            conversations_by_policy['good'].append(good_conv)
                            conversations_by_policy['bad'].append(bad_conv)
                            performance_by_policy['good'].append(performance_good)
                            performance_by_policy['bad'].append(performance_bad)
                            v_good += performance_good
                            v_bad += performance_bad
                            pickle.dump(performance_by_policy, open(performance_by_policy_fname, 'wb'))
                            pickle.dump(conversations_by_policy, open(conversations_by_policy_fname, 'wb'))

    v_good /= num_trajectories
    v_bad /= num_trajectories
    print("V(good): " + str(v_good) + " V(bad): " + str(v_bad))
    print("Files saved")
