import os.path
import numpy as np
import matplotlib.pyplot as plt
import util

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set

    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the
    smoothed estimates of q_d
    """

    spam_words = get_word_freq(file_lists_by_category[0])
    ham_words = get_word_freq(file_lists_by_category[1])

    vocab_set = set(spam_words.keys()).union(set(ham_words.keys()))
    spam_bag = sum(spam_words.values())
    ham_bag = sum(ham_words.values())

    pd_dict = {}
    qd_dict = {}

    for word in vocab_set:
        if word not in spam_words.keys():
            spam_words[word] = 0
        if word not in ham_words.keys():
            ham_words[word] = 0

    for word, freq in spam_words.items():
        p_d = (freq + 1) / (spam_bag + len(vocab_set))
        pd_dict[word] = p_d

    for word, freq in ham_words.items():
        q_d = (freq + 1) / (ham_bag + len(vocab_set))
        qd_dict[word] = q_d

    probabilities_by_category = (pd_dict, qd_dict)
    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category, hypothesis_crit = 0):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)],
    representing the log posterior probabilities
    """

    word_bag = get_word_freq([filename])
    p_d, q_d = probabilities_by_category
    prior1 = prior_by_category[0]
    prior0 = 1 - prior1

    # hypothesis_crit = 0 / np.Infinity / np.NINF
    likelihood1 = 0
    likelihood0 = 0

    for word, freq in word_bag.items():
        if word in p_d.keys():
            likelihood1 += np.log(p_d[word]) * freq
            likelihood0 += np.log(q_d[word]) * freq

    post_h1 = np.exp(likelihood1 / 10000) / (np.exp(likelihood0 / 10000) + np.exp(likelihood1 / 10000))
    post_h0 = np.exp(likelihood0 / 10000) / (np.exp(likelihood0 / 10000) + np.exp(likelihood1 / 10000))

    if likelihood1 - likelihood0 > hypothesis_crit:
        result_string = 'spam'
    else:
        result_string = 'ham'

    classify_result = (result_string, [np.log(post_h1), np.log(post_h0)])

    return classify_result

if __name__ == '__main__':

    # folder for training and testing
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))

    # Learn the distributions
    probabilities_by_category = learn_distributions(file_lists)

    # prior class distribution
    priors_by_category = [0.5, 0.5]

    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam'
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham'
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam'
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham'

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)

        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base)
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))


    error1_set = []
    error2_set = []
    hypothesis_set = [np.NINF, -140, -100, -60, -20, 20, 60, 100, 140, np.Infinity]
    for hypothesis_crit in hypothesis_set:
        # Store the classification results
        performance_measures = np.zeros([2,2])
        # explanation of performance_measures:
        # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
        # rows correspond to true label, columns correspond to guessed label
        # to be more clear, performance_measures = [[p1 p2]
        #                                           [p3 p4]]
        # p1 = Number of emails whose true label is 'spam' and classified as 'spam'
        # p2 = Number of emails whose true label is 'spam' and classified as 'ham'
        # p3 = Number of emails whose true label is 'ham' and classified as 'spam'
        # p4 = Number of emails whose true label is 'ham' and classified as 'ham'

        # Classify emails from testing set and measure the performance
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                     probabilities_by_category,
                                                     priors_by_category, hypothesis_crit)

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        # template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        # print(template % (correct[0],totals[0],correct[1],totals[1]))
        error1 = (totals[0] - correct[0]) / totals[0]
        error2 = (totals[1] - correct[1]) / totals[1]
        error1_set.append(error1)
        error2_set.append(error2)

    plt.figure()
    plt.title("Trade off between Type1 and Type 2")
    plt.scatter(error1_set, error2_set)
    plt.xlabel('Type1 Error')
    plt.ylabel('Type2 Error')
    plt.grid()
    plt.savefig("nbc.pdf")
