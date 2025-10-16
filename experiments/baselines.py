'''
Implement BERTScore, BLEU, Bleurt here to compare between transcripts that are from the original dataset and transcripts from the bad and good policy
'''
import evaluate
import tqdm
from run_tutortest_khan_academy import *

def generate_transcript(student_id, cognitive_error, solving_strategy, policy_quality):
    examples = pickle.load(open("offline_data/offline_few_shot_exs_all.pkl", "rb"))
    all_states = pickle.load(open("offline_data/labeled_states_all.pkl", "rb"))
    all_actions = pickle.load(open("offline_data/labeled_actions_all.pkl", "rb"))
    student_example_dict = filter_examples_student(all_states, all_actions, examples)
    tutor_example_dict = filter_examples_tutor(all_states, all_actions, examples)
    sc = examples[student_id]  # conversations
    tutor_identified_cognitive_error = False
    s = 'e'  # cognitive error state
    curr_conv = sc.split("\n\n")[0]  # This is the dialogue until now.
    for it in range(5):  # TODO: change back to 10 if necessary
        a = adaptive_good_policy(s, curr_conv)  # The policy is the same, it's just the phrasing that's different?
        if policy_quality == 'bad':
            tutor_dialogue = "Tutor: " + bad_tutor_response(a, curr_conv, tutor_example_dict)
        elif policy_quality == 'good':
            tutor_dialogue = "Tutor: " + good_tutor_response(a, curr_conv, tutor_example_dict, solving_strategy)
        else:
            print("Invalid policy quality: " + str(policy_quality))
        curr_conv += tutor_dialogue

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

        curr_conv += response

        return curr_conv

if __name__ == '__main__':
    good_policy_tutor_utterances = []
    bad_policy_tutor_utterances = []
    bleu_good = []
    bertscore_good = []
    bleu_bad = []
    bertscore_bad = []

    ka_transcripts = pickle.load(open("offline_data/offline_few_shot_exs_all.pkl", "rb"))
    cognitive_errors_by_student = pickle.load(open("offline_data/cognitive_errors_by_student_all.pkl", "rb"))
    solving_strategies_by_student = pickle.load(open("results/cognitive_error_solutions.pkl", "rb"))

    reference_transcripts = []
    correct_answers_by_student = {0: [8.4, 0.96], 1: [8, 3], 2: [858936, 667472], 3: [9 / 4, 10 / 6], 7: [21, 40],
                                  8: [240, 512], 14: [8, 4], 20: [17, 1 / 2], 21: [34, 5], 25: [1315, 4953]}
    questions_to_answer = list(correct_answers_by_student.keys())
    for _, i in enumerate(tqdm.tqdm(questions_to_answer)):
        print("Student: " + str(i))
        for j in range(5):
            khan_academy_transcript = ka_transcripts[i] # Compare scores to this.
            reference_transcripts.append(khan_academy_transcript)
            reference_transcripts.append(khan_academy_transcript)
            reference_transcripts.append(khan_academy_transcript)


            cognitive_errors = cognitive_errors_by_student[i]
            solving_strategies = solving_strategies_by_student[i]

            # Each student has one of the cognitive errors.
            conversation = generate_transcript(i, cognitive_error=cognitive_errors[0], solving_strategy=solving_strategies[0], policy_quality='good')
            good_policy_tutor_utterances.append(conversation)
            conversation = generate_transcript(i, cognitive_error=cognitive_errors[0], solving_strategy=solving_strategies[0], policy_quality='bad')
            bad_policy_tutor_utterances.append(conversation)

            # Each student has the other cognitive error
            conversation = generate_transcript(i, cognitive_error=cognitive_errors[1],
                                               solving_strategy=solving_strategies[1], policy_quality='good')
            good_policy_tutor_utterances.append(conversation)
            conversation = generate_transcript(i, cognitive_error=cognitive_errors[1],
                                               solving_strategy=solving_strategies[1], policy_quality='bad')
            bad_policy_tutor_utterances.append(conversation)

            # Each student has both of the cognitive errors
            conversation = generate_transcript(i, cognitive_error=cognitive_errors,
                                               solving_strategy=solving_strategies, policy_quality='good')
            good_policy_tutor_utterances.append(conversation)
            conversation = generate_transcript(i, cognitive_error=cognitive_errors,
                                               solving_strategy=solving_strategies, policy_quality='bad')
            bad_policy_tutor_utterances.append(conversation)


            pickle.dump(good_policy_tutor_utterances, open("results/baselines/good_transcript_11082024.pkl", "wb"))
            pickle.dump(bad_policy_tutor_utterances, open("results/baselines/bad_transcript_11082024.pkl", "wb"))
            pickle.dump(reference_transcripts, open("results/baselines/reference_11082024.pkl", "wb"))
    print("Files saved")

    good_policy_utterances = pickle.load(open("results/baselines/good_transcript_11082024.pkl", "rb"))
    bad_policy_utterances = pickle.load(open("results/baselines/bad_transcript_11082024.pkl", "rb"))
    reference_transcripts = pickle.load(open("results/baselines/reference_11082024.pkl", "rb"))

    #Calculate Bleu and BERTScores
    bertscore = evaluate.load("bertscore")
    bleu = evaluate.load("bleu")
    bleurt = evaluate.load("bleurt", module_type="metric")
    perplexity = evaluate.load("perplexity", module_type="metric")

    bleu_scores_good = []
    bleu_scores_bad = []
    for i in range(len(good_policy_utterances)):
        bleu_good = bleu.compute(predictions=[good_policy_utterances[i]], references=[reference_transcripts[i]])['bleu'] # They're both low
        bleu_bad = bleu.compute(predictions=[bad_policy_utterances[i]], references=[reference_transcripts[i]])['bleu'] # These are both low
        bleu_scores_good.append(bleu_good)
        bleu_scores_bad.append(bleu_bad)

    bert_scores_good = []
    bert_scores_bad = []
    for i in range(len(good_policy_utterances)):
        bert_good = bertscore.compute(predictions=[good_policy_utterances[i]], references=[reference_transcripts[i]], lang='en')['f1']
        bert_bad = bertscore.compute(predictions=[bad_policy_utterances[i]], references=[reference_transcripts[i]], lang='en')['f1']
        bert_scores_good.append(bert_good)
        bert_scores_bad.append(bert_bad)

    bleurt_scores_good = []
    bleurt_scores_bad = []
    for i in range(len(good_policy_utterances)):
        bleurt_good = bleurt.compute(predictions=[good_policy_utterances[i]], references=[reference_transcripts[i]])['scores'][0]
        bleurt_bad = bleurt.compute(predictions=[bad_policy_utterances[i]], references=[reference_transcripts[i]])['scores'][0]
        bleurt_scores_good.append(bleurt_good)
        bleurt_scores_bad.append(bleurt_bad)

    perplexity_good = []
    perplexity_bad = []
    for i in range(len(good_policy_utterances)):
        perp_good = perplexity.compute(predictions=[good_policy_utterances[i]], model_id='gpt2')['mean_perplexity']
        perp_bad = perplexity.compute(predictions=[bad_policy_utterances[i]], model_id='gpt2')['mean_perplexity']
        perplexity_good.append(perp_good)
        perplexity_bad.append(perp_bad)


    # Calculate baseline scores for the CIMA dataset
    ground_truth = [
        "Tree is all\'albero. Please try to fill in the blank in Italian.\n What is tree in Italian?\n Okay, I\'ll give you a hint.  tree is  all\'albero \n Oh I remember you saying that.\n Do you know how to say is in front of the?\n No can you tell me?\n all\'to the is prepended to the following word when it begins with a vowel.  This is a contraction of al ('to') and l\' ('the')\n Ok i think I got it.\n Do you know how to say tree? \n al'lalbero",
        "Tree is all\'albero. Please try to fill in the blank in Italian.\n What is tree in Italian?\n Okay, I\'ll give you a hint.  tree is  all\'albero \n Oh I remember you saying that.\n Do you know how to say is in front of the?\n No can you tell me?\n all\'to the is prepended to the following word when it begins with a vowel.  This is a contraction of al ('to') and l\' ('the')\n Ok i think I got it.\n Do you know how to say tree? \n al'lalbero",
        "Tree is all\'albero. Please try to fill in the blank in Italian.\n What is tree in Italian?\n Okay, I\'ll give you a hint.  tree is  all\'albero \n Oh I remember you saying that.\n Do you know how to say is in front of the?\n No can you tell me?\n all\'to the is prepended to the following word when it begins with a vowel.  This is a contraction of al ('to') and l\' ('the')\n Ok i think I got it.\n Do you know how to say tree? \n al'lalbero",
        "Tree is all\'albero. Please try to fill in the blank in Italian.\n What is tree in Italian?\n Okay, I\'ll give you a hint.  tree is  all\'albero \n Oh I remember you saying that.\n Do you know how to say is in front of the?\n No can you tell me?\n all\'to the is prepended to the following word when it begins with a vowel.  This is a contraction of al ('to') and l\' ('the')\n Ok i think I got it.\n Do you know how to say tree? \n al'lalbero",
        "Tree is all\'albero. Please try to fill in the blank in Italian.\n What is tree in Italian?\n Okay, I\'ll give you a hint.  tree is  all\'albero \n Oh I remember you saying that.\n Do you know how to say is in front of the?\n No can you tell me?\n all\'to the is prepended to the following word when it begins with a vowel.  This is a contraction of al ('to') and l\' ('the')\n Ok i think I got it.\n Do you know how to say tree? \n al'lalbero",
        "Tree is all\'albero. Please try to fill in the blank in Italian.\n What is tree in Italian?\n Okay, I\'ll give you a hint.  tree is  all\'albero \n Oh I remember you saying that.\n Do you know how to say is in front of the?\n No can you tell me?\n all\'to the is prepended to the following word when it begins with a vowel.  This is a contraction of al ('to') and l\' ('the')\n Ok i think I got it.\n Do you know how to say tree? \n al'lalbero",
        "Tree is all\'albero. Please try to fill in the blank in Italian.\n What is tree in Italian?\n Okay, I\'ll give you a hint.  tree is  all\'albero \n Oh I remember you saying that.\n Do you know how to say is in front of the?\n No can you tell me?\n all\'to the is prepended to the following word when it begins with a vowel.  This is a contraction of al ('to') and l\' ('the')\n Ok i think I got it.\n Do you know how to say tree? \n al'lalbero",
        "Please try to fill in the blank in Italian. \n what is in front of?\n Remember that  is in front of the is e di fronte\n what is tree?",
        "Please try to fill in the blank in Italian. \n what is in front of?\n Remember that  is in front of the is e di fronte\n what is tree?",
        "Please try to fill in the blank in Italian. \n what is in front of?\n Remember that  is in front of the is e di fronte\n what is tree?"]

    conversations_by_policy = pickle.load(open("results/practice_problems/cima_conversations_by_policy.pkl", 'rb'))
    good_policy_utterances = conversations_by_policy['good']
    bad_policy_utterances = conversations_by_policy['bad']
    bleu_scores_good = []
    bleu_scores_bad = []
    for i in range(10):
        bleu_good = bleu.compute(predictions=[good_policy_utterances[i]], references=[ground_truth[i]])['bleu'] # They're both low
        bleu_bad = bleu.compute(predictions=[bad_policy_utterances[i]], references=[ground_truth[i]])['bleu'] # These are both low
        bleu_scores_good.append(bleu_good)
        bleu_scores_bad.append(bleu_bad)

    bert_scores_good = []
    bert_scores_bad = []
    for i in range(10):
        bert_good = bertscore.compute(predictions=[good_policy_utterances[i]], references=[ground_truth[i]], lang='en')['f1']
        bert_bad = bertscore.compute(predictions=[bad_policy_utterances[i]], references=[ground_truth[i]], lang='en')['f1']
        bert_scores_good.append(bert_good)
        bert_scores_bad.append(bert_bad)

    bleurt_scores_good = []
    bleurt_scores_bad = []
    for i in range(10):
        bleurt_good = bleurt.compute(predictions=[good_policy_utterances[i]], references=[ground_truth[i]])['scores'][0]
        bleurt_bad = bleurt.compute(predictions=[bad_policy_utterances[i]], references=[ground_truth[i]])['scores'][0]
        bleurt_scores_good.append(bleurt_good)
        bleurt_scores_bad.append(bleurt_bad)

    perplexity_good = []
    perplexity_bad = []
    for i in range(10):
        perp_good = perplexity.compute(predictions=[good_policy_utterances[i]], model_id='gpt2')['mean_perplexity']
        perp_bad = perplexity.compute(predictions=[bad_policy_utterances[i]], model_id='gpt2')['mean_perplexity']
        perplexity_good.append(perp_good)
        perplexity_bad.append(perp_bad)
    import ipdb; ipdb.set_trace()


