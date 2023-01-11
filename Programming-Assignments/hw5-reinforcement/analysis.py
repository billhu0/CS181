# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    answerDiscount = 0.9
    answerNoise = 0.01  # changed from 0.2 to 0.01. Reduced noice gives correct result.
    return answerDiscount, answerNoise


def question3a():  # Prefer +1, risking -10
    # We want to avoid the (+10), so we set a very high-cost discount,
    # letting it trying to get the closest reward and exit as soon as possible.
    answerDiscount = 0.1

    # We want to risk the cliff, so we set the noise to be zero,
    # letting it believes that there's no (-10) risk as long as you don't go downwards.
    answerNoise = 0.0
    answerLivingReward = 0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3b():  # prefer +1, avoid -10
    # We want to avoid the (+10), so we set a very high-cost discount,
    # letting it trying to get the closest reward and exit as soon as possible.
    answerDiscount = 0.1

    # We don't to risk the cliff, so we set some noise,
    # letting it detects that there's a risk if we take the shorter (red) path.
    answerNoise = 0.1
    answerLivingReward = 0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3c():  # prefer +10, risking -10
    # We want to take the +10, so the discount need to be less costly,
    # making the +10 exit more appealing than the +1 exit.
    answerDiscount = 0.9

    # We want to risk the cliff, so we set the noise to be zero,
    # letting it believes that there's no (-10) risk as long as you don't go downwards.
    answerNoise = 0.0
    answerLivingReward = 0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3d():  # prefer +10, avoiding -10
    # We want to take the +10, so the discount need to be less costly.
    answerDiscount = 0.9

    # We don't to risk the cliff, so we set some noise,
    # letting it detects that there's a risk if we take the shorter (red) path.
    answerNoise = 0.1
    answerLivingReward = 0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3e():
    answerDiscount = 0.0
    answerNoise = 0.0
    answerLivingReward = 0.5
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question8():
    return 'NOT POSSIBLE'
    # answerEpsilon = 0.99
    # answerLearningRate = 0.5
    # return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis

    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
