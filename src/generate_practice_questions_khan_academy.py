'''
Generate practice questions, which we will use to determine a reward signal in the POMDP setting
'''
import tqdm
from generating_cognitive_errors_khan_academy import *

def transform_practice_problems(practice_problems): # used to transform the mistake string to a list
    llist = []
    split_problems = practice_problems.split("\n\n")
    n = 1
    for ex in split_problems:
        if len(ex) > 1:
            try:
                sample = ex.split(f"{n}. ")[1]
                llist.append(sample)
                n += 1
            except:
                print(f"{ex} not splittable")
    return llist

def generate_practice_questions(student_goal, cognitive_errors):
    mistake_prompt = (
        f"You are an online tutor helping a middle school student with a math problem. The following is the beginning of the student's "
        f"conversation with you: {student_goal}. There are 2 possible cognitive errors that this student could have. "
        f"These are: {cognitive_errors}. Can you provide a list of 2 different practice problems that a student should be "
        f"able to solve if they do not have either of the cognitive errors? Do not include the solution in these practice problems, and make sure "
        f"that the practice problems have quantitative answers with a single solution. Also avoid problems that involve graphing. The solution"
        f"should be a single number for all problems."
        f"Phrase the problems as much in mathematical notation as possible. For example instead of 3 divided by 4, write 3/4. Begin your generation with `List:`. Use"
        f"the following format: [Format] 1. [Question 1] \n\n 2. [Question 2]\n. In this format, Question 1 and Question 2 are the two practice problems that you generate. ")
    msgs = [{"role": "user", "content": mistake_prompt}, {"role": "assistant", "content": "List:"}]
    message = api_call(mistake_prompt, msgs, model=CLAUDE_SONNET)
    problems = message.content[0].text

    problem_list = transform_practice_problems(problems)
    return problem_list

if __name__ == '__main__':
    cognitive_errors_fname = ''
    fname = '' # Saved practice problems
    practice_problems_by_student = {}
    for _, i in enumerate(tqdm.tqdm(range(11))):
        cognitive_errors_by_student = pickle.load(open(cognitive_errors_fname, "rb"))
        cognitive_errors = cognitive_errors_by_student[i]
        examples = pickle.load(open("offline_data/offline_few_shot_exs_10282024.pkl", "rb"))
        sc = examples[i]
        student_goal = sc.split("\n\n")[0]
        practice_problems = generate_practice_questions(student_goal, cognitive_errors)
        print("Cognitive Errors: " + str(cognitive_errors) + "\n")
        print("Problems: " + str(practice_problems) + "\n")
        practice_problems_by_student[i] = practice_problems
    pickle.dump(practice_problems_by_student, open(fname, "wb"))

    correct_answers_by_student = {0: [3.51, 0.03], 1: [143, 3], 2: [406463, 24244591], 3: [], 4:[828, 867], 5:[], 6:[]}


