"""
Used for training agents for 2D scenarios
"""

if __name__ == "__main__":
    from atcenv import Environment
    from CDR_DQN import DQN
    import numpy as np
    from tensorboardX import SummaryWriter
    import torch
    import time
    from jsonargparse import ArgumentParser, ActionConfigFile
    from tqdm import tqdm

    # Hyper-parameters for DQN
    BATCH_SIZE = 1000
    LEARNING_RATE = 0.001
    EPSILON = 0.8
    GAMMA = 0.9
    TARGET_REPLACE_ITER = 500
    MEMORY_CAPACITY = 20000
    N_STATES = 36 * 3  # number of neurons in the input layer
    N_HIDDEN = 36 * 6  # number of neurons in the hidden layer
    N_ACTION = 3 * 3  # number of neurons in the output layer
    TRAINED_NN = None  # file name of the trained neural network, if provided

    # episode setting
    MAX_EPISODE = 1000000
    SHOW_ITER = 100
    SAVE_ITER = 1000

    rl = DQN(BATCH_SIZE=BATCH_SIZE,
             LEARNING_RATE=LEARNING_RATE,
             EPSILON=EPSILON,
             GAMMA=GAMMA,
             TARGET_REPLACE_ITER=TARGET_REPLACE_ITER,
             MEMORY_CAPACITY=MEMORY_CAPACITY,
             N_HIDDEN=N_HIDDEN,
             N_ACTIONS=N_ACTION,
             N_STATES=N_STATES,
             ENV_A_SHAPE=0,
             EPISODE=MAX_EPISODE,
             SHOW_ITER=SHOW_ITER,
             TRAINED_NN=TRAINED_NN
             )

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=MAX_EPISODE)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()

    # init environment
    env = Environment(**vars(args.env))

    # create file for tensorboardX
    writer = SummaryWriter(comment='EuroInno')

    # execute episode
    for episode in tqdm(range(args.episodes)):
        # initialize MDP set
        S = []  # state/ partial observation
        R = []  # reward
        A = []  # action

        # initialize the scenario and get the 1st observation
        obs = env.reset()
        S.append(obs)
        done = False

        # initialize performance metrics
        number_conflict = 0
        number_warning = 0
        eco_airspeed = 0
        eco_heading = 0
        smo_airspeed = 0
        smo_heading = 0

        # execute simulation
        while not done:
            # get actions
            action = []
            for i in range(env.num_flights):
                if i in env.done:
                    action.append(np.nan)
                else:
                    action.append(rl.choose_action(S[-1][i], episode))
            # update step
            rew, obs, done = env.step(action)
            S.append(obs)
            R.append(rew)
            A.append(action)

            # count performance metrics
            for i in range(env.num_flights):
                if i not in env.done:
                    if env.flights[i].r_conflict == -1:
                        number_conflict += 1
                    if env.flights[i].r_warning == -1:
                        number_warning += 1
                    if env.flights[i].r_eco_airspeed == -1:
                        eco_airspeed -= 1
                    if env.flights[i].r_eco_heading == -1:
                        eco_heading -= 1
                    if env.flights[i].r_smo_airspeed == -1:
                        smo_airspeed -= 1
                    if env.flights[i].r_smo_heading == -1:
                        smo_heading -= 1

            # store transition
            for i in range(env.num_flights):
                if i not in env.done:
                    rl.store_transition(S[-2][i], A[-1][i], R[-1][i], S[-1][i])
        # close environment
        env.close()

        # reward statistics
        num_rew = 0
        total_rew = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] is not np.nan:
                    num_rew += 1
                    total_rew += R[i][j]

        # flight distance statistics
        total_optimal_trajectory_length = 0
        total_trajectory_length = 0
        for i in range(env.num_flights):
            total_optimal_trajectory_length += env.flights[i].optimal_trajectory_length
            total_trajectory_length += env.flights[i].trajectory_length

        # reinforcement learning
        if rl.memory_counter > MEMORY_CAPACITY:
            rl.learn()

        # record performance metrics for the episode by tensorboardX
        # View statistical charts: open '...\atcenv-main' in terminal, then input command 'tensorboard --logdir=runs'
        if episode % SHOW_ITER == 0:
            writer.add_scalar('2D_training/1_num_conflicts', number_conflict, episode)
            writer.add_scalar('2D_training/2_num_warnings', number_warning, episode)
            writer.add_scalar('2D_training/3_total_reward', total_rew, episode)
            writer.add_scalar('2D_training/4_average_reward_per_action', total_rew / num_rew, episode)
            writer.add_scalar('2D_training/5_rate_of_total_extra_flight_distance',
                              total_trajectory_length / total_optimal_trajectory_length, episode)
            writer.add_scalar('2D_training/6_economy_of_heading', eco_heading, episode)
            writer.add_scalar('2D_training/7_economy_of_airspeed', eco_airspeed, episode)
            writer.add_scalar('2D_training/8_smoothness_of_heading', smo_heading, episode)
            writer.add_scalar('2D_training/9_smoothness_of_airspeed', smo_airspeed, episode)

        # save neural network parameters
        if episode % SAVE_ITER == 0:
            t = time.strftime('%m%d%H%M', time.localtime())
            torch.save(rl.eval_net,
                       f'./nn/input_{N_STATES}_hidden_{N_HIDDEN}_output_{N_ACTION}_time_{t}_episode_{episode}')
    writer.close()
