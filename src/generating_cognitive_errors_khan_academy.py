import pickle
import numpy as np
from utils import api_call, CLAUDE_SONNET, transform2list
import re
import tqdm

def decode_action(action_idx):
    idx_to_text = {1: "asked you a question", 2: "gave you a hint or revealed important information",
                   3: "corrected a previous step", 4: "confirmed your prior steps", -1: "had a response", "NA": "had a response"}
    return idx_to_text[action_idx]

def decode_state(state_code):
    code_to_state = {'f': 'focused', 'd':'distracted', 'NA': 'neither focused nor distracted'}
    return code_to_state[state_code]

def adaptive_good_policy(s, sc):
    q_prompt = (f'You are an online math tutor working with a 6th grade student on a problem. In the dialogue'
                f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                f'``Student:". \n\n[Dialogue]\n{sc}. Currently, the student is in a {s} state. Based on the student\'s '
                f'state, what is the best action to take? Your options are: 1. ask a question either about the prior'
                f'steps in the problem, about the application of interest, or ask a new practice problem, 2. give the student a hint, 3. correct a '
                f'prior step of the problem derivation, and 4. encourage the student. ')

    msgs = [{"role": "user",
             "content": "If you think you should ask a question, respond <<1>>. "
                        "If you think you should give a hint, respond <<2>>. "
                        "If you think you should correct a prior step in the problem derivation, respond <<3>>. "
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

def decode_state(state_code):
    code_to_state = {'e': 'has a cognitive error', 'ne':'does not have a cognitive error'}
    return code_to_state[state_code]

def student_response(s, a, curr_conv, example_dict, p_mistake, cognitive_error):
    # s_prime_options = ['d', 'f', "NA"]
    # s_prime = np.random.choice(s_prime_options, p=P[s_prime_options.index(s)])
    state = decode_state(s)
    # s_prime = decode_state(s_prime) # Don't use the next state here.
    # student_mistakes = pickle.load(open("offline_data/mistakes_by_student.pkl", "rb"))
    examples = example_dict[a] # These are few shot examples that correspond to how other students have responded to the same tutor action
    a = decode_action(a)
    if s == 'e': # student has a cognitive error
        response = generate_student_cognitive_error_list_mistake(curr_conv, cognitive_error, examples)
    elif s == 'ne':
        response = generate_student_no_cognitive_error(curr_conv, cognitive_error, examples)
    else:
        print("Not a valid state: " + str(s))
    return response

def decode_action_tutor_response(action_idx):
    idx_to_text = {1: "ask a question either about the prior steps in the problem, about the application of interest, or provide an example problem", 2: "give the student a hint or reveal important information about the problem",
                   3: "correct a previous step of the problem derivation if necessary", 4: "confirm prior steps of the problem derivation and encourage the student in their progress", -1: "respond without a specific goal"}
    return idx_to_text[action_idx]

def tutor_response(s, a, curr_conv, examples, policy_quality):
    example_responses = examples[a]
    a = decode_action_tutor_response(a)
    if s == 'e':
        sentence_error = "Currently, the student has a cognitive error. "
    elif s == 'ne':
        sentence_error = "Currently, the student has overcome their cognitive error. "
    problem = "Sally has 8 apples and Bob has 10 apples. How many apples do they have together?"
    if policy_quality in ["good", "adaptive_good", "deterministic_bad", "bad_good_tone"]:
        # Then, write code to verify the answer.
        # If the answer is not equal to the output of the code, correct the student.

        q_prompt = (f'You are an online math tutor working with a 6th grade student on a problem. In the dialogue'
                    f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                    f'``Student:". \n\n[Dialogue]\n{curr_conv}. {sentence_error} Generate a response'
                    f'to the student\'s last utterance in the role of an online math tutor. Specifically, {a}. Here are examples'
                    f'of how an online tutor will interact with a student to {a}. \n\n[Examples]\n{example_responses}.  Keep'
                    f'your response brief and do not reveal the answer to the problem that the student is trying to work on. Additionally, try to be goal oriented in your response. For example, '
                    f'if you want to give the student a practice problem, actually provide a practice problem. '
                    f'Additionally, when verifying the math steps taken by the student, it is important to carefully verify their work and correct them if necessary. '
                    f'Verify the student\'s work by writing an equation if appropriate, and write code to solve'
                    f'that equation. For example, say we have the following addition problem. Problem: {problem}. '
                    f'The corresponding equation is 8+10=X. Once you write code, run the code in a script. Once you run the code, the output will '
                    f'say that X=18. The student is incorrect if their response does not match the output. Do not include any justification for your response or any of the code in your response'
                    f'to the student. ')
    elif policy_quality in ['bad']:
        q_prompt = (f'You are a bad online math tutor working with a 6th grade student on a problem. In the dialogue'
                    f'below, the your utterances are prefaced by ``Tutor:" and the student\'s utterances are prefaced by'
                    f'``Student:". \n\n[Dialogue]\n{curr_conv}. {sentence_error} Generate a response'
                    f'to the student\'s last utterance in the role of a poorly trained math tutor. Specifically, {a}. As a '
                    f'bad math tutor, you can easily put the student into a distracted state and get them to not '
                    f'focus on the math problem that they are trying to solve. ')
    msgs = [{"role": "user", "content": q_prompt}, {"role": "assistant", "content": "Tutor:"}]
    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    response = message.content[0].text
    return response

def stop_conversation(curr_conv):
    q_prompt =  (f"We are evaluating a dialogue between an AI tutor and a 6th grade human student. They are working to solve a math problem. "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the student’s utterances are "
        f"prefaced by ”Student:”. Here is the dialogue: {curr_conv}\n\nBased on the most recent exchange"
        f" between the student and the tutor, is the student inclined to leave the conversation? For example, they may not"
         f"want to continue the conversation if the tutor is distracting them, or if they are not learning about math. Answer <<yes>> or <<no>> "
        f"and explain your reason. "
        f"Make sure to format your answer as <<yes>> or <<no>>. If you're unsure, answer"
        f" <<unknown>>. Let's take a deep breath and think carefully. ")
    msgs = [{"role": "user", "content": q_prompt}, {"role": "assistant", "content": "Student:"}]
    message = api_call(q_prompt, msgs, CLAUDE_SONNET)
    response = message.content[0].text
    label = 'NA'
    if "<<no>>" in response:
        label = False # student wants to continue the conversation
    elif "<<yes>>" in response:
        label = True  # student wants to leave the conversation
    return label

def filter_examples_student(all_states, all_actions, conversations):
    example_dict = {1:[], 2:[], 3:[], 4:[], -1:[]}
    for it, (states, actions, sc) in enumerate(zip(all_states, all_actions, conversations)):
        conversation = sc.split("Student: ")[1:]
        for i in range(len(states)-1):
            tutor_action = actions[i]
            if tutor_action not in [1, 2, 3, 4, -1]:
                continue
            student_response = conversation[i+1].split('\n\n')[0]
            example_dict[tutor_action].append(student_response)
    return example_dict # There's no examples for -1.

def filter_examples_tutor(all_states, all_actions, conversations):
    example_dict = {1: [], 2: [], 3: [], 4: [], -1: []}
    for it, (states, actions, sc) in enumerate(zip(all_states, all_actions, conversations)):
        num_turns = len(states)
        conversation = sc.split("Student: ")[1:]
        for i in range(num_turns):
            tutor_action = actions[i]
            if tutor_action not in [1, 2, 3, 4, -1]:
                continue
            tutor_text = conversation[i].split('\n\n')[1:] # Just remove the first utterance which is the student
            tutor_sc = " ".join(tutor_text)
            example_dict[tutor_action].append(tutor_sc)
    return example_dict  # There's no examples for -1.

def generate_mistakes(verbose=False): # For right now, ask an LLM to produce possible mistakes for each of these initial conversations
    examples = pickle.load(open("offline_data/offline_few_shot_exs_10282024.pkl", "rb"))
    mistakes_by_student = {} # student_idx to list of possible mistakes
    for it, sc in enumerate(examples):
        student_goal = sc.split("\n\n")[0]
        mistake_prompt = (f"We are designing a task when an AI tutor is helping a middle school student solve math problems. "
                          f"A student began their interaction with a tutor with this request. Request: {student_goal}  "
                          f"The student may have cognitive errors that require the tutor to personalize their teaching content. "
                          f"Can you provide a list of 2 different cognitive errors that the student could make where the agent "
                          f"needs to change their teaching content for different mistakes? Reference the appropriate literature that"
                          f"cites these cognitive errors as relevant to the problem domain. Do not include information about"
                          f"how a tutor agent should respond to these mistakes. Begin your generation with `List:`.")
        msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "List:"}]
        message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
        mistakes = message.content[0].text
        if verbose:
            print("Student Goal: " + str(student_goal))
            print("Mistakes: " + str(mistakes) + "\n")
        mistake_list = transform2list(mistakes)
        mistakes_by_student[it] = mistake_list
    pickle.dump(mistakes_by_student, open("offline_data/cognitive_errors_by_student_10282024.pkl", "wb"))

def generate_student_cognitive_error_list_mistake(curr_conv, cognitive_error, examples):
    mistake_prompt = (
        f"You are a middle school student that is asking an online tutor for help with a problem. The following is an excerpt of your dialogue so far.  "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are "
        f"prefaced by ”Student:”. Here is the dialogue: {curr_conv}\n\n Since you are still learning, you have a cognitive error that causes you to make mistakes and ask questions."
        f"Currently, you have the following cognitive error: {cognitive_error}. This cognitive error affects how you respond to practice problems and your general"
        f"understanding of the problem domain. Respond to the tutor's last utterance while conditioning on this cognitive error. Do not include details about your cognitive error in the response. "
        f"Make sure to reference the tutor's prior statements and provide a productive response."
        f"Additionally, students of your age are usually abrupt when interacting with a tutor. Occasionally, your tone may be informal or unenthusiastic. Here are examples of how other"
        f"students respond to the same category of tutor response:  \n\n[Examples]\n{examples}.Preface the response with Student:. ")
    msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "Student:"}]
    message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
    mistake = message.content[0].text
    return mistake

def generate_student_no_cognitive_error(curr_conv, cognitive_error, examples):
    mistake_prompt = (
        f"You are a middle school student that is asking an online tutor for help with a problem. The following is an excerpt of your dialogue so far.  "
        f"In the dialogue, the tutor’s utterances are prefaced by ”Tutor:” and the your utterances are "
        f"prefaced by ”Student:”. Here is the dialogue: {curr_conv}\n\n Since you are still learning, you had a cognitive error at the beginning of this dialogue."
        f"This error was: {cognitive_error}. This cognitive error affected how you respond to practice problems and your general"
        f"understanding of the problem domain. You have now overcome this cognitive error, and learned from your mistakes. "
        f"Respond to the tutor's last utterance conditioning on overcoming this cognitive error. Make sure to reference the tutor's prior statements and provide a productive"
        f"response. Do not include details about your cognitive error in the response. "
        f"Additionally, students of your age are usually abrupt when interacting with a tutor. Occasionally, your tone may be informal or unenthusiastic. Here are examples of how other"
        f"students respond to the same category of tutor response:  \n\n[Examples]\n{examples}.Preface the response with Student:. ")
    msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "Student:"}]
    message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
    mistake = message.content[0].text
    return mistake

def evaluate_policy(policy_quality, student_id, verbose=False):
    examples = pickle.load(open("offline_data/offline_few_shot_exs_all.pkl", "rb"))
    all_states = pickle.load(open("offline_data/labeled_states_all.pkl", "rb")) # Number of student is the length of this vector
    all_actions = pickle.load(open("offline_data/labeled_actions_all.pkl", "rb"))
    # Filter examples for responses that student have had to tutor actions. Should produce a dictionary of tutor_action --> student response
    student_example_dict = filter_examples_student(all_states, all_actions, examples)
    tutor_example_dict = filter_examples_tutor(all_states, all_actions, examples)

    utterances_error = []
    utterances_no_error = []

    sc = examples[student_id]
    s = 'e' # cognitive error state
    curr_conv = sc.split("\n\n")[0]  # This is the dialogue until now.
    student_goal = curr_conv
    # Student cognitive error should be constant across the conversation (we'll work on fixing the cognitive error later)
    cognitive_errors_by_student = pickle.load(open("offline_data/cognitive_errors_by_student_all.pkl", "rb"))
    cognitive_error = np.random.choice(cognitive_errors_by_student[student_id])
    if verbose:
        print("Cognitive Error: " + str(cognitive_error))
        print("Student: " + str(student_id) + " Goal: " + str(curr_conv))
    value_student = 0
    if verbose: print(curr_conv)
    # Change the stopping criteria  to be 2 in a row distracted states.
    stopping_criteria = False
    stopping_count = [s]
    while not stopping_criteria:
        if policy_quality == 'adaptive_good':
            a = adaptive_good_policy(s, curr_conv)
        else:
            print("not a valid policy")
        if verbose: print("Student state: ", s)
        if verbose: print("Chosen action: ", a)
        # Generate a tutor response here
        tutor_dialogue = "Tutor: " + tutor_response(s, a, curr_conv, tutor_example_dict, policy_quality)
        curr_conv += tutor_dialogue
        if verbose: print(tutor_dialogue)

        # Generate the student response here, they should make a mistake when it is natural to make a mistake
        response = "Student: " + student_response(s, a, curr_conv, student_example_dict, p_mistake=1, cognitive_error=cognitive_error) # The student retains this cognitive error for the entire conversation?
        if s == 'e':
            utterances_error.append(response)
        elif s == 'ne':
            utterances_no_error.append(response)
        curr_conv += response
        if verbose: print(response)

        flip = np.random.uniform()
        if flip < 0.1:
            s_prime = 'ne'
        else:
            s_prime = 'e'

        s = s_prime # Reset the state to the next state
        stopping_count.append(s)
        # This is not exactly accurate, but we need to be able to identify this to generate better tutor policies.
        # if s_prime == 'ne':
        #     print("Student overcame the cognitive error. ")
        #     stopping_criteria=True
        if len(stopping_count) >= 10:
            stopping_criteria = True

    return curr_conv, utterances_error, utterances_no_error

def identify_solution(cognitive_error, student_goal):
    mistake_prompt = (
        f"We are designing a task when an AI tutor is helping a middle school student solve math problems. "
        f"A student began their interaction with a tutor with this request. Request: {student_goal}  "
        f"The student may have cognitive errors that require the tutor to personalize their teaching content. Here is one of "
        f"the cognitive errors that a student may have: \n[Cognitive Error]:\n {cognitive_error}\n. If a student had this"
        f"cognitive error, can you identify 2 ways that an online tutor could help the student overcome this cognitive"
        f"error? This can be a suggestion about a way that the online tutor can change their teaching content or "
        f"key information that should be communicated to the student. Make your suggestions very specific to "
        f"the cognitive error from earlier and the student request. Recall that the setting is online tutoring, which "
        f"means that visual cues are difficult to employ. Begin your generation with `List:`")
    msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "List:"}]
    message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
    solution = message.content[0].text
    solution_list = transform2list(solution)
    return solution_list

if __name__ == '__main__':
    n_trials = 5; verbose=False
    # generate_mistakes(verbose=verbose)
    # policy_quality = 'adaptive_good'
    # mistake_transcripts = []
    # utterances_error = []
    # utterances_no_error = []
    # for _, i in enumerate(tqdm.tqdm(range(78))): # looping over a subset of the students.
    #     transcript, utt_error, utt_no_error = evaluate_policy(policy_quality=policy_quality, verbose=verbose, student_id=i)
    #     mistake_transcripts.append(transcript)
    #     utterances_error.extend(utt_error)
    #     utterances_no_error.extend(utt_no_error)
    #
    #     pickle.dump(mistake_transcripts, open('results/cognitive_errors_transcript_full_dataset.pkl', 'wb'))
    #     pickle.dump(utterances_error, open('results/utterances_error.pkl', 'wb'))
    #     pickle.dump(utterances_no_error, open('results/utterances_no_error.pkl', 'wb')) # Use these to figure out if we can classify
    cognitive_errors_by_student = pickle.load(open("offline_data/cognitive_errors_by_student_10282024.pkl", "rb"))
    solutions_by_student = {}
    student_transcripts = pickle.load(open("offline_data/offline_few_shot_exs_10282024.pkl", "rb"))
    for _, i in enumerate(tqdm.tqdm(range(len(cognitive_errors_by_student.keys())))):
        solutions_by_student[i] = []
        sc = student_transcripts[i]
        student_goal = sc.split("\n\n")[0]
        for j in range(len(cognitive_errors_by_student[i])):
            cognitive_error = cognitive_errors_by_student[i][j]
            solution = identify_solution(cognitive_error, student_goal=student_goal)
            solutions_by_student[i].append(solution)

    pickle.dump(solutions_by_student, open('results/cognitive_error_solutions_10282024.pkl', 'wb'))

