'''
Figuring out how to get a good classifier for the cognitive errors within a tutor policy evaluation setting.
'''
from idlelib.pyparse import trans

from generating_cognitive_errors_khan_academy import *
import tqdm
import re
import numpy as np
import random

def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))

def bad_tutor_response(a, curr_conv, examples): # This tutor doesn't care about solving cognitive errors, just responding
    verbose=False
    example_responses = examples[a]
    a = decode_action_tutor_response(a)
    problem = "Sally has 8 apples and Bob has 10 apples. How many apples do they have together?"
    # First make sure that the student is correct in their prior utterances, and then condition the following response on that?
    answer_prompt = (f'You are an online math tutor working with a middle school student on a problem. In the dialogue'
                f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                f'``Student:". \n\n[Dialogue]\n{curr_conv}. Using the student\'s last utterance, did the student provide a response'
                f'to a math question or work out any math? If their response contains a single number, it is likely a response'
                f'to a prior you asked and they are providing a response to a math question. Additionally, if their response includes a formula, they '
                f'are working out a math problem. In contrast, if they are asking a question about a prior step that does not contain a numerical'
                f'representation, they are not solving a problem. Respond using the following format: \n [Format]:\n\n <<answer>> '
                f'[Justification]. \n In the format, answer is either yes or no, and Justification is a place to explain your reasoning.'
                f'Make sure to place <<  and >> around the answer. Make sure that answer is either yes or no, and uncapitalized.')

    msgs = [{"role": "user", "content": answer_prompt}, {"role": "assistant", "content": "Tutor:"}]
    message = api_call(answer_prompt, msgs, model=CLAUDE_SONNET)
    answer_response = message.content[0].text
    if verbose:
        print("Student doing math: " + str(answer_response))
    student_answer = False
    pattern = r'<<\s*(yes|no)\s*>>'
    matches = re.findall(pattern, answer_response)
    if len(matches) > 0:
        try:
            student_answer = (matches[0] == 'yes')
        except:
            if verbose:
                print("Matches: " + str(matches))
    else:
        if verbose:
            print("no matches, defaulting to False")

    if student_answer:
        accuracy_prompt = (f'You are an online math tutor working with a middle school student on a problem. In the dialogue'
                f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                f'``Student:". \n\n[Dialogue]\n{curr_conv}. Based on the student\'s last utterance, it seems like the '
                f'student is attempting to solve a math problem. Is the student accurate in their problem solving? It is important to carefully '
                f'verify the student\'s work and correct them if necessary. '
                f'You can verify the student\'s work by writing an equation if appropriate, and write code to solve'
                f'that equation. For example, say the student is trying to solve following addition problem. Problem: {problem}. '
                f'The corresponding equation is 8+10=X. Once you write code, run the code in a script. Once you run the code, the output will '
                f'say that X=18. The student is incorrect if their answer to the problem does not match the output. Based on your'
                f'verification process, is the student response correct? Respond using the following format: \n [Format]:\n\n <<answer>> '
                f'[Justification]. \n In the format, answer is either yes or no, and Justification is a place to explain your reasoning.'
                f'Make sure to place <<  and >> around the answer. Make sure that answer is either yes or no, both uncapitalized.')

        msgs = [{"role": "user", "content": accuracy_prompt}, {"role": "assistant", "content": "Tutor:"}]
        message = api_call(accuracy_prompt, msgs, model=CLAUDE_SONNET)
        accurate_response = message.content[0].text
        if verbose:
            print("Student correct: " +str(accurate_response))
        student_accurate = False
        pattern = r'<<\s*(yes|no)\s*>>'
        matches = re.findall(pattern, accurate_response)
        if len(matches) > 0:
            try:
                student_accurate = (matches[0] == 'yes')
            except:
                print("Matches: " + str(matches))
        else:
            print("no matches, defaulting to False")

    if student_answer and not student_accurate:
        tutor_response_accurate = (f'Based on the student\'s last response, it seems as if the student is trying to solve a problem. Their answer is not correct. In your response, try to correct the'
                                   f'student\'s procedure')
    if student_answer and student_accurate:
        tutor_response_accurate = (
            f'Based on the student\'s last response, it seems as if the student is trying to solve a problem. Their answer is correct. In your response, encourage the student and re-affirm their process.')
    else:
        tutor_response_accurate = ''

    q_prompt = (f'You are an online math tutor working with a middle school student on a problem. In the dialogue'
                f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                f'``Student:". \n\n[Dialogue]\n{curr_conv}. Generate a response'
                f'to the student\'s last utterance in the role of an online math tutor. Specifically, {a}. Here are examples'
                f'of how an online tutor will interact with a student to {a}. \n\n[Examples]\n{example_responses}.  Keep'
                f'your response brief and do not reveal the answer to the problem that the student is trying to work on. Additionally, try to be goal oriented in your response. For example, '
                f'if you said in a prior utterance that you want to give the student a practice problem, actually provide a practice problem. {tutor_response_accurate}')
    msgs = [{"role": "user", "content": q_prompt}, {"role": "assistant", "content": "Tutor:"}]
    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    response = message.content[0].text
    return response

def good_tutor_response(a, curr_conv, examples, solving_strategy):
    verbose=False
    example_responses = examples[a]
    a = decode_action_tutor_response(a)
    problem = "Sally has 8 apples and Bob has 10 apples. How many apples do they have together?"
    # First make sure that the student is correct in their prior utterances, and then condition the following response on that?
    answer_prompt = (f'You are an online math tutor working with a middle school student on a problem. In the dialogue'
                     f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                     f'``Student:". \n\n[Dialogue]\n{curr_conv}. Using the student\'s last utterance, did the student provide a response'
                     f'to a math question or work out any math? If their response contains a single number, it is likely a response'
                     f'to a prior you asked and they are providing a response to a math question. Additionally, if their response includes a formula, they '
                     f'are working out a math problem. In contrast, if they are asking a question about a prior step that does not contain a numerical'
                     f'representation, they are not solving a problem. Respond using the following format: \n [Format]:\n\n <<answer>> '
                     f'[Justification]. \n In the format, answer is either yes or no, and Justification is a place to explain your reasoning.'
                     f'Make sure to place <<  and >> around the answer. Make sure that answer is either yes or no, and uncapitalized.')

    msgs = [{"role": "user", "content": answer_prompt}, {"role": "assistant", "content": "Tutor:"}]
    message = api_call(answer_prompt, msgs, model=CLAUDE_SONNET)
    answer_response = message.content[0].text
    if verbose:
        print("Student doing math: " + str(answer_response))
    student_answer = False
    pattern = r'<<\s*(yes|no)\s*>>'
    matches = re.findall(pattern, answer_response)
    if len(matches) > 0:
        try:
            student_answer = (matches[0] == 'yes')
        except:
            if verbose:
                print("Matches: " + str(matches))
    else:
        if verbose:
            print("no matches, defaulting to False")
    student_accurate = False
    if student_answer:
        accuracy_prompt = (
            f'You are an online math tutor working with a middle school student on a problem. In the dialogue'
            f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
            f'``Student:". \n\n[Dialogue]\n{curr_conv}. Based on the student\'s last utterance, it seems like the '
            f'student is attempting to solve a math problem. Is the student accurate in their problem solving? It is important to carefully '
            f'verify the student\'s work and correct them if necessary. '
            f'You can verify the student\'s work by writing an equation if appropriate, and write code to solve'
            f'that equation. For example, say the student is trying to solve following addition problem. Problem: {problem}. '
            f'The corresponding equation is 8+10=X. Once you write code, run the code in a script. Once you run the code, the output will '
            f'say that X=18. The student is incorrect if their answer to the problem does not match the output. Based on your'
            f'verification process, is the student response correct? Respond using the following format: \n [Format]:\n\n <<answer>> '
            f'[Justification]. \n In the format, answer is either yes or no, and Justification is a place to explain your reasoning.'
            f'Make sure to place <<  and >> around the answer. Make sure that answer is either yes or no, both uncapitalized. Finally,'
            f'make sure to condition on just the last response. Do not verify other instances of the student responding.')

        msgs = [{"role": "user", "content": accuracy_prompt}, {"role": "assistant", "content": "Tutor:"}]
        message = api_call(accuracy_prompt, msgs, model=CLAUDE_SONNET)
        accurate_response = message.content[0].text
        if verbose:
            print("Student correct: " + str(accurate_response))

        pattern = r'<<\s*(yes|no)\s*>>'
        matches = re.findall(pattern, accurate_response)
        if len(matches) > 0:
            try:
                student_accurate = (matches[0] == 'yes')
            except:
                print("Matches: " + str(matches))
        else:
            print("no matches, defaulting to False")

    if student_answer and not student_accurate:
        tutor_response_accurate = (
            f'Based on the student\'s last response, it seems as if the student is trying to solve a problem. Their answer is not correct. In your response, try to correct the'
            f'student\'s procedure')
    if student_answer and student_accurate:
        tutor_response_accurate = (
            f'Based on the student\'s last response, it seems as if the student is trying to solve a problem. Their answer is correct. In your response, encourage the student and re-affirm their process.')
    else: # Student didn't ask a question
        tutor_response_accurate = ''
    if len(solving_strategy) == 2: # There are two cognitive errors
        q_prompt = (f'You are an online math tutor working with a middle school student on a problem. In the dialogue'
                    f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                    f'``Student:". \n\n[Dialogue]\n{curr_conv}. Generate a response'
                    f'to the student\'s last utterance in the role of an online math tutor. Specifically, {a}. Here are examples'
                    f'of how an online tutor will interact with a student to {a}. \n\n[Examples]\n{example_responses}.  Keep'
                    f'your response brief and do not reveal the answer to the problem that the student is trying to work on. Additionally, try to be goal oriented in your response. For example, '
                    f'if you said in a prior utterance that you want to give the student a practice problem, actually provide a practice problem. {tutor_response_accurate}'
                    f'Finally, the student is likely to have multiple cognitive errors when they begin'
                    f'their interaction with you. Your goal is to help them overcome these cognitive errors, without knowing the actual cognitive errors. Ask the student questions to help you understand'
                    f'their cognitive errors, and use well-known strategies to help them overcome these cognitive errors. This may involve explaining broader conceptual ideas about the problem at hand'
                    f'and using real-world examples. For example, this student requires one of the following strategies to overcome one of their cognitive errors: \n[Strategy]: \n {solving_strategy[0]}. They '
                    f'will also require one of the following strategies to overcome their other cognitive error: \n[Strategy]: \n{solving_strategy[1]}. This student will require both strategies to overcome their'
                    f'cognitive errors. ')
    else: # There is one cognitive error
        q_prompt = (f'You are an online math tutor working with a middle school student on a problem. In the dialogue'
                    f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                    f'``Student:". \n\n[Dialogue]\n{curr_conv}. Generate a response'
                    f'to the student\'s last utterance in the role of an online math tutor. Specifically, {a}. Here are examples'
                    f'of how an online tutor will interact with a student to {a}. \n\n[Examples]\n{example_responses}.  Keep'
                    f'your response brief and do not reveal the answer to the problem that the student is trying to work on. Additionally, try to be goal oriented in your response. For example, '
                    f'if you said in a prior utterance that you want to give the student a practice problem, actually provide a practice problem. {tutor_response_accurate}'
                    f'Finally, the student is likely to have a cognitive error when they begin'
                    f'their interaction with you. Your goal is to help them overcome this cognitive error, without knowing the actual cognitive error. Ask the student questions to help you understand'
                    f'their cognitive error, and use well-known strategies to help them overcome this cognitive error. This may involve explaining broader conceptual ideas about the problem at hand'
                    f'and using real-world examples. For example, this student requires one of the following strategies to help them overcome their cognitive error: \n[Strategy]: \n {solving_strategy} ')
    msgs = [{"role": "user", "content": q_prompt}, {"role": "assistant", "content": "Tutor:"}]
    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    response = message.content[0].text
    return response

def verify_answer(curr_conv, practice_problems, correct_answers):
    q_prompt = (f"We are evaluating a dialogue between an online tutor and a middle school student that is learning how to do math. In the dialogue"
                f"below, the tutor's utterances are prefaced by ``Tutor:'' and the student\'s utterances are prefaced by"
                f"``Student:''. \n\n[Dialogue]\n{curr_conv}. The tutor just asked the student to solve a few practice problems. [Practice problems]:\n\n {practice_problems}. The following are the "
                f"correct answers to the practice problems. [Correct Answers]:\n\n {correct_answers}. Based on the transcript so far, did the student solve the practice problems correctly? The student's responses"
                f"have to match the correct answers exactly. Any deviation from the correct answers is incorrect including placement of decimals. Sometimes the student is confident in their response. This does not mean"
                f"that they have responded correctly. Respond in the following format:"
                f"[Format]:\n\n The student correctly responded to <<percentage>> percentage of the practice problems. [Justification]\n\n In the formatted sentence, percentage is a number between 0 and 100, and [Justification]"
                f"is a place for you to provide a justification for the percentage that you provided. Make sure to place <<  and >> around the percentage."
                f"")
    msgs = [{"role": "user", "content": q_prompt}, {"role": "assistant", "content": "Percentage:"}]
    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    if verbose: print(message)
    response = message.content[0].text
    number = 0
    pattern = r'<<\s*(\d+)\s*>>'
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        try:
            number = float(matches[0])
        except ValueError:
            number = 0
            print("Input was invalid; defaulting to 0 ...")
    else:
        print("no matches, defaulting to 0")
    print(str(number))
    return number

def student_response_cognitive_error(a, curr_conv, example_dict, cognitive_error):
    if cognitive_error == None:
        cog_sentence = f"Even though you are still learning, you have no cognitive errors and do not make mistakes. "
    elif len(cognitive_error) == 2:
        cog_sentence = f"Since you are still learning, you have two cognitive errors that cause you to make mistakes and ask questions."
        f"You have the following two cognitive errors: {cognitive_error}. These cognitive errors affects how you respond to practice problems and your general"
        f"understanding of the problem domain. Respond to the tutor's last utterance while conditioning on having both cognitive errors. Do not include details about these cognitive errors in the response."
    else:
        cog_sentence = (f"Since you are still learning, you have a cognitive error that causes you to make mistakes and ask questions as you are learning."
        f"You have the following cognitive error: {cognitive_error}. This cognitive error affects how you respond to practice problems and your general"
        f"understanding of the problem domain. Respond to the tutor's last utterance while conditioning on having this cognitive error. Keep in mind that you should maintain this"
        f"cognitive error throughout your interactions with the tutor. For example, be consistent across responses. You might also be slow to pick up new material. Do not include details about this cognitive error in the response.")
    mistake_prompt = (
        f"You are a middle school student that is asking an online tutor for help with a problem. The following is an excerpt of your dialogue so far.  "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are "
        f"prefaced by ”Student:”. Here is the dialogue: {curr_conv}\n\n {cog_sentence} "
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

def ask_practice_problem(curr_conv, practice_problems, cognitive_error, state, correct_answer):
    if verbose: print("\nAsking practice problems\n")
    new_conv = curr_conv
    practice_problem_prompt = (
        f"You are an online tutor that is interacting with a 6th grade student to help them practice math. The following is an excerpt of your dialogue so far. \n\n[Dialogue]\n{curr_conv}. "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are prefaced by ”Student:” To "
        f"better understand the student's grasp of the topic, you are going to ask a couple of practice problems. These practice problems will tell you what the student knows. Ask the following "
        f"two practice problems to the student. \n\n[Practice Problems]\n{practice_problems}. Provide your question in the following format:\n\n[Format]:\n Here are a few practice problems"
        f"to help me understand how much you have learned so far. Can you try to solve these problems? Here are the problems: [problems]\n In this sentence, problems are practice problems. Make sure to stick to the required format. ")
    msgs = [{"role": "user", "content": practice_problem_prompt}, {"role": "assistant", "content": "Tutor:"}]
    message = api_call(practice_problem_prompt, msgs, model=CLAUDE_SONNET)
    tutor_dialogue = "Tutor: " + message.content[0].text
    if verbose: print(tutor_dialogue)
    new_conv += tutor_dialogue
    if state == 'e':
        if cognitive_error == None:
            cog_sentence = f"Even though you are still learning, you have no cognitive errors and do not make mistakes. "
        elif len(cognitive_error) == 2:
            cog_sentence = (f"Since you are still learning, you have two cognitive errors that cause you to make mistakes and ask questions."
            f"You have the following two cognitive errors: {cognitive_error}. These cognitive errors affects how you respond to practice problems and your general"
            f"understanding of the problem domain. Respond to the tutor's last utterance while conditioning on having both cognitive errors. For example, if you have been making mistakes"
            f"so far, continue to make mistakes when you solve the practice problems. Do not include details about these cognitive errors in the response.")
        else:
            cog_sentence = (f"Since you are still learning, you have a cognitive error that causes you to make mistakes and ask questions."
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
        f"You are a middle school student that is asking an online tutor for help with a problem. The following is an excerpt of your dialogue so far.  "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are "
        f"prefaced by ”Student:”. Here is the dialogue: {new_conv}\n\n {cog_sentence}. The online tutor just asked you to solve two practice problems. Solve these practice problems. "
        f"Preface the response with Student: and use the following format:\n"
        f"[Format]:\n [Problem solving discussion] <<answer1>> [Problem solving discussion] <<answer2>> [Problem solving discussion] \n\n In the format, use the Problem solving discussion portions for adding text justifying your answers. answer1 represents your"
        f"final answer for the first problem and answer2 represents your final answer to the second problem. Be sure to only include your final answer within << and >>. Do not include any intermediate steps."
        f"Make sure to use << and >> around answer1 and answer2, and stick to the format. Only answer1 and answer2 should be within brackets (<< and >>). Do not include any simplifications. Additionally, students of your age are "
        f"usually abrupt when interacting with a tutor. Occasionally, your tone may be informal or unenthusiastic.")
    msgs = [{"role": "user", "content": student_response_prompt}, {"role": "assistant", "content": "Student:"}]
    message = api_call(student_response_prompt, msgs, model=CLAUDE_SONNET)
    student_response = "Student: " + message.content[0].text
    if verbose: print(student_response)
    new_conv += student_response

    # Calculate performance
    pattern = f'<<\s*(-?\d+(\.\d+)?|-\d+/\d+|\d+/\d+)\s*>>'
    answer1 = 0
    answer2 = 0
    matches = re.findall(pattern, student_response)
    if len(matches) > 0:
        try:
            answer1 = convert_to_float(matches[0][0])
        except:
            answer1 = 0
            print("Input: " + str(matches))
        try:
            answer2 = convert_to_float(matches[1][0])
        except:
            answer2 = 0
            print("Input: " + str(matches))
    else:
        print("no matches, defaulting to 0: " + str(student_response))

    points = 0
    if answer1 == correct_answer[0]:
        points += 1
    else:
        print("Correct answer: " + str(correct_answer[0]) + " Answer: " + str(answer1))
    if answer2 == correct_answer[1]:
        points += 1
    else:
        print("Correct answer: " + str(correct_answer[1]) + " Answer: " + str(answer2))
    return (points/2)*100

def tutor_analysis(sc, cognitive_error, solution):
    verbose=False
    if len(solution) == 2:
        prompt = (
            f"We are evaluating a conversation between an online math tutor and a middle school student as they work through a"
            f"math problem. In the dialogue below, the the tutor's are prefaced by ``Tutor:'' and the student\'s utterances are "
            f"prefaced by ``Student:''. \n\n[Dialogue]\n{sc}. The student currently has both of the following cognitive errors: [Cognitive errors]\n{cognitive_error}\n. "
            f"According to experts, to help the student overcome the first error, the tutor must use at least one of the following strategies: [Strategies]\n{solution[0]}. To"
            f"overcome the second error, the tutor must use at least one of the following strategies: [Strategies]\n{solution[1]}. Based"
            f"on the dialogue from earlier, has the tutor employed both sets of strategies to help the student overcome their cognitive error? Respond with a <<yes>>"
            f"or <<no>> depending on whether or not the tutor has used both of the strategies provided, and provide a justification. Respond <<yes>> only if the tutor"
            f"has used at least one strategy from both lists of strategies provided. Use the following format: [Format]\n <<answer>> [Justification]\n."
            f"In the formatted response, answer is either yes or no, and Justification is a space to provide reasoning for you answer. ")
    else:
        prompt = (f"We are evaluating a conversation between an online math tutor and a middle school student as they work through a"
                  f"math problem. In the dialogue below, the the tutor's are prefaced by ``Tutor:'' and the student\'s utterances are "
                  f"prefaced by ``Student:''. \n\n[Dialogue]\n{sc}. The student currently has the following cognitive error: [Cognitive error]\n{cognitive_error}\n. "
                  f"According to experts, to help the student overcome this error, the tutor must use at least one of the following strategies: [Strategies]\n{solution}. Based"
                  f"on the dialogue from earlier, has the tutor employed any of these strategies to help the student overcome their cognitive error? Respond with a <<yes>>"
                  f"or <<no>> depending on whether or not the tutor has used the strategies provided, and provide a justification. Use the following format: [Format]\n <<answer>> [Justification]\n."
                  f"In the formatted response, answer is either yes or no, and Justification is a space to provide reasoning for you answer.  ")
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

def generate_transcript_practice_problems(student_id, cognitive_error, practice_problems, correct_answers, solving_strategy, policy_quality):
    deterministic_transitions = False

    examples = pickle.load(open("offline_data/offline_few_shot_exs_all.pkl", "rb"))
    all_states = pickle.load(open("offline_data/labeled_states_all.pkl", "rb"))
    all_actions = pickle.load(open("offline_data/labeled_actions_all.pkl", "rb"))
    student_example_dict = filter_examples_student(all_states, all_actions, examples)
    tutor_example_dict = filter_examples_tutor(all_states, all_actions, examples)

    sc = examples[student_id] # conversations
    tutor_identified_cognitive_error = False
    s = 'e'  # cognitive error state
    curr_conv = sc.split("\n\n")[0]  # This is the dialogue until now.
    if verbose:
        print("Cognitive Error: " + str(cognitive_error))
    if verbose: print(curr_conv)
    random_idx = np.random.choice([i for i in range(10)])
    gamma = 0.98
    for it in range(5): # TODO: change back to 10 if necessary
        if verbose:
            print("Student state: " + str(s))
        if deterministic_transitions:
            if cognitive_error is not None:
                if it >= 7:
                    s = 'ne'
            else:
                s = 'ne'
        a = adaptive_good_policy(s, curr_conv) # The policy is the same, it's just the phrasing that's different?
        if policy_quality == 'bad':
            tutor_dialogue = "Tutor: " + bad_tutor_response(a, curr_conv, tutor_example_dict)
        elif policy_quality == 'good':
            tutor_dialogue = "Tutor: " + good_tutor_response(a, curr_conv, tutor_example_dict, solving_strategy)
        else:
            print("Invalid policy quality: " + str(policy_quality))
        curr_conv += tutor_dialogue
        if verbose: print(tutor_dialogue)

        # Calculate next state depending on the transition
        if not deterministic_transitions:
            # Identify if the tutor said something that can solve the cognitive error in the current conversation
            if not tutor_identified_cognitive_error:
                tutor_identified_cognitive_error = tutor_analysis(curr_conv, cognitive_error, solving_strategy)

            # Then change the state and keep it constant
            if tutor_identified_cognitive_error:
                s = 'ne'

        if s == 'e':
            student_response = student_response_cognitive_error(a, curr_conv, student_example_dict,
                                                                cognitive_error=cognitive_error)
        elif s == 'ne':
            student_response = student_response_no_error(a, curr_conv, student_example_dict, cognitive_error)

        response = "Student: " + student_response
        if verbose: print(response)

        curr_conv += response
        gamma *= 0.98

        # Generate a response to the practice problem here.
        if deterministic_transitions:
            if it == random_idx:
                performance = ask_practice_problem(curr_conv, practice_problems, cognitive_error, s, correct_answers)
                if verbose:
                    print("\nFinishing transcript generation\n")
                return performance, s
    if not deterministic_transitions:
        performance = ask_practice_problem(curr_conv, practice_problems, cognitive_error, s, correct_answers)
        return performance*gamma, s

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
    binary_classification_format = "[Format]:  My binary prediction of whether the student has a cognitive error or not is <<number>>. In the formatted sentence, number is either 0 or 1. Make sure to place <<  and >> around the number."
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
    pattern = r'<<\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*>>'
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
    v_bad_policy = 0
    v_good_policy = 0
    practice_questions_by_student = pickle.load(open("cognitive_error_results/practice_problems_10102024.pkl", "rb"))
    n_turns = 5
    performance_by_student = {}
    gt_by_student = {}
    # correct_answers_by_student = {0: [], 1:[], 2:[60, 13], 3:[], 4:[4, 10], 5:[-9, 4], 6:[], 7:[], 8:[], 9:[], 10:[]}
    correct_answers_by_student = {0: [8.4, 0.96], 1: [8, 3], 2: [858936, 667472], 3: [9 / 4, 10 / 6], 7:[21,40], 8:[240,512], 14:[8,4], 20:[17, 1/2], 21:[34, 5], 25:[1315,4953]}
    questions_to_answer = list(correct_answers_by_student.keys())
    trajectory_returns_good = []
    trajectory_returns_bad = []
    for _, i in enumerate(tqdm.tqdm(questions_to_answer)):
        print("Student: " + str(i))
    # for _, i in enumerate(tqdm.tqdm(questions_to_answer)):
    #     cognitive_errors_by_student = pickle.load(open("offline_data/cognitive_errors_by_student_10282024.pkl", "rb"))
    #     solving_strategies_by_student = pickle.load(open("results/cognitive_error_solutions_10282024.pkl", "rb"))
        cognitive_errors_by_student = pickle.load(open("offline_data/cognitive_errors_by_student_all.pkl", "rb"))
        solving_strategies_by_student = pickle.load(open("results/cognitive_error_solutions.pkl", "rb"))
        cognitive_errors = cognitive_errors_by_student[i]
        solving_strategies = solving_strategies_by_student[i]
        practice_problems = practice_questions_by_student[i]
        performance_by_student[i] = {1:[], 2:[]}
        gt_by_student[i] = {1:[], 2:[]}
        for _, _ in enumerate(tqdm.tqdm(range(n_turns))): # Generate sufficient transcripts for training data
            correct_answers = correct_answers_by_student[i]
            # This student will always get everything right.
            # perf_none, gt_state = generate_transcript_practice_problems(i, None, practice_problems, correct_answers)
            # print("No cog error: Ground truth state: " + str(gt_state) + " Performance on problems: " + str(perf_none))
            # performance_by_student[i]["None"].append(perf_none)
            # gt_by_student[i]["None"].append(gt_state)
            for policy_quality in ['good', 'bad']: # Add good back
                perf_one, gt_state = generate_transcript_practice_problems(i, cognitive_errors[0], practice_problems, correct_answers, solving_strategy=solving_strategies[0], policy_quality=policy_quality)
                print(f"Policy quality: {policy_quality} Cog err1: Ground truth state: " + str(gt_state) + " Performance on problems: " + str(perf_one))

                perf_two, gt_state = generate_transcript_practice_problems(i, cognitive_errors[1], practice_problems, correct_answers, solving_strategy=solving_strategies[1], policy_quality=policy_quality)
                print(f"Policy quality: {policy_quality} Cog err2: Ground truth state: " + str(gt_state) + " Performance on problems: " + str(perf_two))

                if policy_quality == 'good':
                    v_good_policy += (perf_one + perf_two)
                    trajectory_returns_good.append(perf_one)
                    trajectory_returns_good.append(perf_two)
                elif policy_quality == 'bad':
                    v_bad_policy += (perf_one + perf_two)
                    trajectory_returns_bad.append(perf_one)
                    trajectory_returns_bad.append(perf_two)

                perf_both, gt_state = generate_transcript_practice_problems(i, cognitive_errors, practice_problems, correct_answers, solving_strategy=solving_strategies, policy_quality=policy_quality)
                print(f"Policy quality: {policy_quality} Cog err Both: Ground truth state: " + str(gt_state) + " Performance on problems: " + str(perf_both))

                if policy_quality == 'good':
                    v_good_policy += perf_both
                    trajectory_returns_good.append(perf_both)
                elif policy_quality == 'bad':
                    v_bad_policy += perf_both
                    trajectory_returns_bad.append(perf_both)

            pickle.dump(trajectory_returns_good, open("results/practice_problems/returns_good_11062024.pkl", "wb"))
            pickle.dump(trajectory_returns_bad, open("results/practice_problems/returns_bad_11062024.pkl", "wb"))
    v_good_policy /= (3* n_turns * len(questions_to_answer))
    v_bad_policy /= (3*n_turns * len(questions_to_answer))
    print("V(good): " + str(v_good_policy) + " V(bad): " + str(v_bad_policy))
    print("Files saved")




