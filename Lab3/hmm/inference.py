import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(hidden_states)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps

    # Initialization
    backward_messages[num_time_steps - 1] = rover.Distribution()

    for z in all_possible_hidden_states:
        backward_messages[num_time_steps - 1][z] = 1


    # TODO: Compute the forward messages
    for i in range(1, num_time_steps):
        Xi = observations[i]
        forward_messages[i] = rover.Distribution()

        for zi in all_possible_hidden_states:
            if Xi is None:
                Xi_given_zi = 1

            else:
                Xi_given_zi = rover.observation_model(zi)[Xi]

            sum = 0
            for z, prev in forward_messages[i-1].items():
                factor = rover.transition_model(z)[zi]
                sum = sum + prev * factor * Xi_given_zi

            if sum == 0:
                continue

            forward_messages[i][zi] = sum

        forward_messages[i].renormalize()


    # TODO: Compute the backward messages
    for i in range(num_time_steps - 1, 0, -1):
        backward_messages[i-1] = rover.Distribution()
        Xi = observations[i]

        for zi in all_possible_hidden_states:

            sum = 0
            for z, next in backward_messages[i].items():

                if Xi is None:
                    Xi_given_zi = 1

                else:
                    Xi_given_zi = rover.observation_model(z)[Xi]

                factor = rover.transition_model(zi)[z]
                sum = sum + next * factor * Xi_given_zi

            if sum == 0:
                continue

            backward_messages[i-1][zi] = sum

        backward_messages[i-1].renormalize()

    # TODO: Compute the marginals
    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        sum = 0

        for z in all_possible_hidden_states:
            alpha_i = forward_messages[i][z]
            beta_i = backward_messages[i][z]

            if alpha_i * beta_i != 0:
                marginals[i][z] = alpha_i * beta_i
            sum = sum + alpha_i * beta_i

        if sum != 0:
            for state in marginals[i].keys():
                marginals[i][state] = marginals[i][state] / sum

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    max_prev_path = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps


    for i in range(num_time_steps):
        w[i] = rover.Distribution()
        max_prev_path[i] = rover.Distribution()

    for zi, prob in prior_distribution.items():
        X0_given_z0 = prior_distribution[zi]

        pcond_X0_z0 = observation_model(zi)[observations[0]]
        if X0_given_z0 * prob == 0:
            continue
        w[0][zi] = np.log(prob) + np.log(pcond_X0_z0)

    for i in range(1, num_time_steps):
        xi = observations[i]

        for zi in all_possible_hidden_states:
            if xi is None:
                Xi_given_zi = 1
            else:
                Xi_given_zi = observation_model(zi)[xi]

            if Xi_given_zi == 0:
                continue

            max_prev = np.NINF
            for z, prev in w[i - 1].items():

                zi_given_prev = transition_model(z)[zi]
                if zi_given_prev == 0:
                    continue

                new_prev = np.log(zi_given_prev) + prev
                if new_prev > max_prev:
                    max_prev = new_prev
                    max_prev_path[i][zi] = z
            w[i][zi] = np.log(Xi_given_zi) + max_prev


    estimate_zN = None
    for zi, max_zi in w[num_time_steps - 1].items():
        max_prob = np.NINF
        if max_zi > max_prob:
            max_prob = max_zi
            estimate_zN = zi
    estimated_hidden_states[num_time_steps - 1] = estimate_zN

    for i in range(num_time_steps-2, -1, -1):
        estimated_hidden_states[i] = max_prev_path[i + 1][estimated_hidden_states[i + 1]]

    return estimated_hidden_states



if __name__ == '__main__':

    enable_graphics = False

    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'

    # load data
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()

    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


    for timestep in range(num_time_steps):
        print("Most likely parts of marginal at time %d:" % (timestep))
        print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
        print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    print("Calculating error rate...")
    error_fb = 0
    error_vi = 0

    for i in range(num_time_steps):

        max_prob = 0
        fb_estimate = None
        for state, prob in marginals[i].items():
            if prob > max_prob:
                fb_estimate = state
                max_prob = prob

        print(i)
        print('fb:{}'.format(fb_estimate))
        print('map:{}'.format(estimated_states[i]))
        print('real:{}'.format(hidden_states[i]))

        if fb_estimate != hidden_states[i]:
            print('fb error')
            error_fb = error_fb + 1
        if estimated_states[i] != hidden_states[i]:
            print('vi error')
            error_vi = error_fb + 1

    print('error_fb: ', error_fb, '/', len(hidden_states))
    print('error_vi: ', error_vi, '/', len(hidden_states))


    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
