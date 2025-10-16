'''
Figuring out how to get a good classifier for the cognitive errors within a tutor policy evaluation setting.
'''
import sys
sys.path.append('../')
from generating_cognitive_errors_khan_academy import *
import tqdm
import re
import random

def tutor_response(a, curr_conv, examples):
    example_responses = examples[a]
    a = decode_action_tutor_response(a)
    problem = "Sally has 8 apples and Bob has 10 apples. How many apples do they have together?"
    q_prompt = (f'You are an online math tutor working with a 6th grade student on a problem. In the dialogue'
                f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                f'``Student:". \n\n[Dialogue]\n{curr_conv}. Generate a response'
                f'to the student\'s last utterance in the role of an online math tutor. Specifically, {a}. Here are examples'
                f'of how an online tutor will interact with a student to {a}. \n\n[Examples]\n{example_responses}.  Keep'
                f'your response brief and do not reveal the answer to the problem that the student is trying to work on. Additionally, try to be goal oriented in your response. For example, '
                f'if you want to give the student a practice problem, actually provide a practice problem. '
                f'Additionally, when verifying the math steps taken by the student, it is important to carefully verify their work and correct them if necessary. Do not '
                f'reveal the answer in your verification process, and do not include code in your response to the student.'
                f'You can verify the student\'s work by writing an equation if appropriate, and write code to solve'
                f'that equation. For example, say we have the following addition problem. Problem: {problem}. '
                f'The corresponding equation is 8+10=X. Once you write code, run the code in a script. Once you run the code, the output will '
                f'say that X=18. The student is incorrect if their response does not match the output. Do not include any justification for your response or any of the code in your response'
                f'to the student. ')
    msgs = [{"role": "user", "content": q_prompt}, {"role": "assistant", "content": "Tutor:"}]
    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    response = message.content[0].text
    return response

def student_response_cognitive_error(a, curr_conv, example_dict, cognitive_error):
    if cognitive_error == None:
        cog_sentence = f"Even though you are still learning, you have no cognitive errors and do not make mistakes. "
    elif len(cognitive_error) == 2:
        cog_sentence = f"Since you are still learning, you have two cognitive errors that cause you to make mistakes and ask questions."
        f"You have the following two cognitive errors: {cognitive_error}. These cognitive errors affects how you respond to practice problems and your general"
        f"understanding of the problem domain. Respond to the tutor's last utterance while conditioning on both cognitive errors. Do not include details about these cognitive errors in the response."
    else:
        cog_sentence = f"Since you are still learning, you have a cognitive error that causes you to make mistakes and ask questions."
        f"You have the following cognitive error: {cognitive_error}. This cognitive error affects how you respond to practice problems and your general"
        f"understanding of the problem domain.  Respond to the tutor's last utterance while conditioning on this cognitive error. Do not include details about this cognitive error in the response."
    mistake_prompt = (
        f"You are a middle school student that is asking an online tutor for help with a problem. The following is an excerpt of your dialogue so far.  "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are "
        f"prefaced by ”Student:”. Here is the dialogue: {curr_conv}\n\n {cog_sentence} "
        f"Make sure to reference the tutor's prior statements and provide a productive response."
        f"Additionally, students of your age are usually abrupt when interacting with a tutor. Occasionally, your tone may be informal or unenthusiastic. Here are examples of how other"
        f"students respond to the same category of tutor response:  \n\n[Examples]\n{example_dict[a]}.Preface the response with Student:. ")
    msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "Student:"}]
    message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
    mistake = message.content[0].text
    return mistake

def student_response_no_error(a, curr_conv, example_dict, cognitive_error):
    if cognitive_error == None:
        cog_sentence = f"You had no cognitive errors during your interaction with the tutor. "
    elif len(cognitive_error)==2:
        cog_sentence = f"Since you are still learning, you had two cognitive errors at the beginning of your interaction with the tutor. These errors were: {cognitive_error}. You have now overcome this cognitive error, and learned from your mistakes. "
    else:
        cog_sentence = f"Since you are still learning, you had a cognitive error at the beginning of your interaction with the tutor. This error was: {cognitive_error}."
    mistake_prompt = (
        f"You are a middle school student that is asking an online tutor for help with a problem. The following is an excerpt of your dialogue so far.  "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are "
        f"prefaced by ”Student:”. Here is the dialogue: {curr_conv}\n\n {cog_sentence}"
        f"Respond to the tutor's last utterance conditioning on overcoming any cognitive error if you had errors. Make sure to reference the tutor's prior statements and provide a productive"
        f"response. Do not include details about your cognitive error in the response. "
        f"Additionally, students of your age are usually abrupt when interacting with a tutor. Occasionally, your tone may be informal or unenthusiastic. Here are examples of how other"
        f"students respond to the same category of tutor response:  \n\n[Examples]\n{example_dict[a]}.Preface the response with Student:. ")
    msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "Student:"}]
    message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
    mistake = message.content[0].text
    return mistake

def ask_practice_problem():
    pass
def generate_transcript(student_id, cognitive_error):
    examples = pickle.load(open("offline_data/offline_few_shot_exs_all.pkl", "rb"))
    all_states = pickle.load(open("offline_data/labeled_states_all.pkl", "rb"))
    all_actions = pickle.load(open("offline_data/labeled_actions_all.pkl", "rb"))
    student_example_dict = filter_examples_student(all_states, all_actions, examples)
    tutor_example_dict = filter_examples_tutor(all_states, all_actions, examples)

    sc = examples[student_id]
    s = 'e'  # cognitive error state
    curr_conv = sc.split("\n\n")[0]  # This is the dialogue until now.

    if verbose:
        print("Cognitive Error: " + str(cognitive_error))
        print("Student: " + str(student_id) + " Goal: " + str(curr_conv))

    if verbose: print(curr_conv)

    for it in range(10): # First 7 turns are state=error, last 3 turns are state=no error if you have a cog error
        if cognitive_error is not None:
            if it >=7:
                s = 'ne'
        else:
            s = 'e'
        a = adaptive_good_policy(s, curr_conv)
        if verbose: print("Student state: ", s)
        if verbose: print("Chosen action: ", a)
        # Generate a tutor response here
        tutor_dialogue = "Tutor: " + tutor_response(a, curr_conv, tutor_example_dict)
        curr_conv += tutor_dialogue
        if verbose: print(tutor_dialogue)

        # Generate the student response here, they should make a mistake when it is natural to make a mistake
        if s == 'e':
            student_response = student_response_cognitive_error(a, curr_conv, student_example_dict, cognitive_error=cognitive_error)
        elif s == 'ne':
            student_response = student_response_no_error(a, curr_conv, student_example_dict, cognitive_error)
        response = "Student: " + student_response
        curr_conv += response
        if verbose: print(response)
    return curr_conv

def generate_transcript_practice_problems(student_id, cognitive_error, practice_problems):
    examples = pickle.load(open("offline_data/offline_few_shot_exs_all.pkl", "rb"))
    all_states = pickle.load(open("offline_data/labeled_states_all.pkl", "rb"))
    all_actions = pickle.load(open("offline_data/labeled_actions_all.pkl", "rb"))
    student_example_dict = filter_examples_student(all_states, all_actions, examples)
    tutor_example_dict = filter_examples_tutor(all_states, all_actions, examples)

    sc = examples[student_id] # conversations
    s = 'e'  # cognitive error state
    curr_conv = sc.split("\n\n")[0]  # This is the dialogue until now.

    if verbose:
        print("Cognitive Error: " + str(cognitive_error))
        print("Student: " + str(student_id) + " Goal: " + str(curr_conv))

    if verbose: print(curr_conv)
    performance = []
    for it in range(10):  # First 7 turns are state=error, last 3 turns are state=no error if you have a cog error
        if cognitive_error is not None:
            if it >= 7:
                s = 'ne'
        else:
            s = 'e'
        a = adaptive_good_policy(s, curr_conv)
        if verbose: print("Student state: ", s)
        if verbose: print("Chosen action: ", a)
        # Generate a tutor response here
        tutor_dialogue = "Tutor: " + tutor_response(a, curr_conv, tutor_example_dict)
        curr_conv += tutor_dialogue
        if verbose: print(tutor_dialogue)

        # Generate the student response here, they should make a mistake when it is natural to make a mistake
        if s == 'e':
            student_response = student_response_cognitive_error(a, curr_conv, student_example_dict,
                                                                cognitive_error=cognitive_error)
        elif s == 'ne':
            student_response = student_response_no_error(a, curr_conv, student_example_dict, cognitive_error)

        response = "Student: " + student_response

        curr_conv += response

        # Generate a response to the practice problem here.
        perf_practice_problems = ask_practice_problem(curr_conv, practice_problems) # Percentage of practice problems you got right
        performance.append(perf_practice_problems)
        if verbose: print(response)
    return performance

def classify_transcript(transcript_portion, cognitive_errors, error_type, examples_one, examples_two):
    examples_one = random.sample(examples_one, 10)
    examples_two = random.sample(examples_two, 10)
    binary_classification_prompt = "Provide a binary response indicating whether the student has a cognitive error. A response of 0 means that the student does not have an error, and a response of 1 means that the student has an error. "
    probability_classification_prompt = "Respond with a probability between 0 and 1 indicating how likely the student is to have the cognitive error. A number closer to 1 means the student is more likely to have the error."
    if error_type == 1:
        cog_error_sentence = f"Based on this response, does the student appear to have the following cognitive error? Cognitive error: {cognitive_errors[0]}. Here are examples of student utterances in which the student has"
        f"the aforementioned cognitive error: {examples_one}. {binary_classification_prompt}"
    elif error_type == 2:
        cog_error_sentence = f"Based on this response, does the student appear to have the following cognitive error? Cognitive error: {cognitive_errors[1]}. Here are examples of student utterances in which the student has"
        f"the aforementioned cognitive error: {examples_two}. {binary_classification_prompt}"
    elif error_type == "Both":
        cog_error_sentence = f"Based on this response, does the student appear to have both of the following cognitive errors? Cognitive errors: {cognitive_errors}. Here are examples of student utterances in which the student has"
        f"either of the aforementioned cognitive errors: {examples_one + examples_two}. {binary_classification_prompt}"
    elif error_type == "None":
        cog_error_sentence = f"Based on this response, does the student appear to have neither of the following cognitive errors? Cognitive errors: {cognitive_errors}. Here are examples of student utterances in which the student has"
        f"either of the aformentioned cognitive errors: {examples_one + examples_two}. {binary_classification_prompt}"

    probability_classification_format = "[Format]: The probability of student having a cognitive error is << probability >>. \n\n In the formatted sentence, probability is a number between 0 and 1. Make sure to place <<  and >> around the probability. "
    binary_classification_format = "[Format]:  My binary prediction of whether the student has a cognitive error or not is <<number>>. In the formatted sentence, number is either 0 or 1. Make sure to place <<  and >> around the number,"
    q_prompt = (
        f"We are evaluating a dialogue between an AI tutor and a human student. They are working to answer the student's initial math question. "
        f"In the dialogue below, the the teacher's utterances are prefaced by ``Tutor:'' and the student\'s utterances are prefaced by"
        f"``Student:''. \n\n[Dialogue]\n{transcript_portion}. Since the student is in middle school and learning about"
        f"math, they may have a cognitive error that causes them to misunderstand the tutor. {cog_error_sentence}"
        f"Respond in the following format: \n\n {binary_classification_format}"
        f"Let's take a deep breath and think carefully. Make sure to format your answer correctly")
    msgs = [{"role": "user", "content": q_prompt}, {"role": "assistant", "content": "Number:"}]
    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    response = message.content[0].text
    number = 0
    pattern = r'<<(0(?:\.\d+)?|1(?:\.0+)?)>>'
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        try:
            number = float(matches[0])
        except ValueError:
            number = 0
            print("Input was invalid; defaulting to 0 ...")
    else:
        print("no matches, defaulting to 0.5")
    print(str(number))
    return number


if __name__ == '__main__':
    verbose=False
    n_transcripts=5
    transcripts_by_student = {}
    for _, i in enumerate(tqdm.tqdm(range(78))):
        cognitive_errors_by_student = pickle.load(open("offline_data/cognitive_errors_by_student_all.pkl", "rb"))
        cognitive_errors = cognitive_errors_by_student[i]
        transcripts_by_student[i] = {1:[], 2:[], "Both": [], "None": []}
        for _ in range(5): # Generate sufficient transcripts for training data
            # Generate transcripts no errors
            transcript_none = generate_transcript(i, None)
            # Generate transcripts both
            transcript_both = generate_transcript(i, cognitive_errors)
            # Generate transcripts one error
            transcript_one = generate_transcript(i, cognitive_errors[0])
            # Generate transcripts other error
            transcript_two = generate_transcript(i, cognitive_errors[1])

            transcripts_by_student[i][1].append(transcript_one)
            transcripts_by_student[i][2].append(transcript_two)
            transcripts_by_student[i]["Both"].append(transcript_both)
            transcripts_by_student[i]["None"].append(transcript_none)

        # Dumped for every single student.
        pickle.dump(transcripts_by_student, open("cognitive_error_results/calibration_transcripts_10102024.pkl", "wb"))

    # Does performance on the two questions correlate to the ground truth at all?
    practice_questions_by_student = pickle.load(open("cognitive_error_results/practice_problems_10102024.pkl", "rb"))
    n_turns = 5
    performance_by_student = {}
    for _, i in enumerate(tqdm.tqdm(range(78))):
        cognitive_errors_by_student = pickle.load(open("offline_data/cognitive_errors_by_student_all.pkl", "rb"))
        cognitive_errors = cognitive_errors_by_student[i]
        practice_problems = practice_questions_by_student[i]
        performance_by_student[i] = {1:[], 2:[], "Both": [], "None": []}
        for _ in range(n_turns): # Generate sufficient transcripts for training data
            # Generate transcripts no errors
            perf_none = generate_transcript_practice_problems(i, None, practice_problems)
            # Generate transcripts both
            perf_both = generate_transcript_practice_problems(i, cognitive_errors, practice_problems)
            # Generate transcripts one error
            perf_one = generate_transcript_practice_problems(i, cognitive_errors[0], practice_problems)
            # Generate transcripts other error
            perf_two = generate_transcript_practice_problems(i, cognitive_errors[1], practice_problems)

            performance_by_student[i][1].append(perf_one)
            performance_by_student[i][2].append(perf_two)
            performance_by_student[i]["Both"].append(perf_both)
            performance_by_student[i]["None"].append(perf_none)

        # Dumped for every single student.
        pickle.dump(performance_by_student, open("cognitive_error_results/practice_problems_10172024.pkl", "wb"))




