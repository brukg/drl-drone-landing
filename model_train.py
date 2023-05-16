#!/usr/bin/env python3
import numpy as np
import torch
import os
from sys import stdout
from time import time
import gym
import argparse
from td3 import TD3, ReplayBuffer, eval_policy

from gym_copter.rendering.threed import ThreeDLanderRenderer
from gym_copter.cmdline import make_parser_3d, parse_view_angles


def make_learn_parser():

    fmtr = argparse.ArgumentDefaultsHelpFormatter

    parser = argparse.ArgumentParser(formatter_class=fmtr)

    parser.add_argument('--env', default='gym_copter:Lander3D-v0', help='Environment id')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true',
                        help='Save at each new best')
    parser.add_argument('--nhid', default=64, type=int, help='Hidden units')
    parser.add_argument('--maxeps', default=None, type=int,
                        help='Maximum number of episodes')
    parser.add_argument('--maxtime', default=3600, type=int,
                        help='Maximum time in seconds')
    parser.add_argument('--target', type=float, default=200,
                        help='Quitting criterion for average reward')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--test-iters', default=100, type=float,
                        help='How often to test and save best')
    parser.add_argument('--eval-episodes', default=10, type=float,
                        help='How many episodes to evaluate for average')

    return parser


def _save(args, avg_reward, history, policy):

    stdout.flush()

    filename = 'td3-%s%+0010.3f' % (args.env, avg_reward)

    # Save run to CSV file
    with open('./runs/' + filename + '.csv', 'w') as csvfile:
        csvfile.write('Iter,Time,Reward\n')
        for row in history[:-1]:  # last reward is always zero
            csvfile.write('%d,%f,%f\n' % (row[0], row[1], row[2]))

    # Save network
    torch.save((policy.get(), args.env, args.nhid),
               open('./models/'+filename+'.dat', 'wb'))

def report(reward, steps, movie):

    print('Got a reward of %+0.3f in %d steps.' % (reward, steps))


def run_td3(env, parts, nhid, movie):

    policy = TD3(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            float(env.action_space.high[0]),
            nhid)

    policy.set(parts)

    report(*eval_policy(policy,
                        env,
                        render=(not movie),
                        eval_episodes=1),
           movie)

def main():

    parser = make_learn_parser()

    hlp = 'Epsiodes during which initial random policy is used'
    parser.add_argument('--start-episodes', default=125, type=int, help=hlp)
    parser.add_argument('--expl-noise', default=0.1,
                        help='Std of Gaussian exploration noise')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='Batch size for both actor and critic')
    parser.add_argument('--tau', default=0.005,
                        help='Target network update rate')
    hlp = 'Noise added to target policy during critic update'
    parser.add_argument('--policy-noise', default=0.2, help=hlp)
    parser.add_argument('--noise-clip', default=0.5,
                        help='Range to clip target policy noise')
    parser.add_argument('--policy-freq', default=2, type=int,
                        help='Frequency of delayed policy updates')

    args = parser.parse_args()

    parser1 = make_parser_3d()
    # parser1.add_argument('filename', metavar='FILENAME', help='input file')
    # args1 = parser1.parse_args()
    # viewangles = parse_view_angles(args1)

    os.makedirs('./runs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    env = gym.make(args.env)

    max_action = float(env.action_space.high[0])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = TD3(
            state_dim,
            action_dim,
            max_action,
            args.nhid,
            discount=args.gamma,
            tau=args.tau,
            policy_noise=args.policy_noise * max_action,
            noise_clip=args.noise_clip * max_action,
            policy_freq=args.policy_freq)

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    env = gym.make(args.env)

    history = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_evaluations = 0
    episode_index = 0
    best_reward = None
    report_index = 0
    total_evaluations = 0
    test_iters = args.test_iters
    first = True

    print('Running %d episodes with random action ...' % args.start_episodes)

    start = time()

    while report_index < np.inf if args.maxeps is None else args.maxeps:

        episode_evaluations += 1

        elapsed = time() - start

        # Select action randomly or according to policy
        if episode_index < args.start_episodes:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise,
                                       size=action_dim)
                    ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = (float(done)
                     if episode_evaluations < env._max_episode_steps
                     else 0)

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if episode_index >= args.start_episodes:
            policy.train(replay_buffer, args.batch_size)

        if done:

            if (episode_evaluations > 1 and
                    episode_index >= args.start_episodes):
                if first:
                    print('Starting training ...')
                    first = False
                print(('Episode %07d:\treward = %+.3f,' +
                       '\tevaluations this epsode= %d' +
                       '\ttotal evaluations = %d') %
                      (report_index+1,
                       episode_reward,
                       episode_evaluations,
                       total_evaluations))
                report_index += 1
                total_evaluations += episode_evaluations

            # Reset everything
            state, done = env.reset(), False
            episode_reward = 0
            episode_evaluations = 0
            episode_index += 1

        history.append((episode_index+1, elapsed, episode_reward))

        # Evaluate episode
        test_index = episode_index - args.start_episodes
        if test_index > 0 and test_index % test_iters == 0:

            testing = test_index != args.test_iters+1

            if testing:
                print('Testing ... ', end='')

            test_iters = args.test_iters+1

            avg_reward, _ = eval_policy(policy, env, args.eval_episodes)

            if testing:
                print('reward = %+.3f' % avg_reward)

            if (args.checkpoint and (best_reward is None or
                                     best_reward < avg_reward)):
                if best_reward is not None:
                    print('\n* Best reward updated: %+.3f -> %+.3f *\n' %
                          (best_reward, avg_reward))
                    _save(args, avg_reward, history, policy)
                best_reward = avg_reward

            if avg_reward >= args.target:
                print('Target average reward %f achieved' % args.target)
                break

        if args.maxtime is not None and elapsed >= args.maxtime:
            print('Timed out at %d seconds' % args.maxtime)
            break

        # parts, env_name, nhid = action, args.env, 100
        # fun = run_td3
        # movie_name = None
    
        # # Begin 3D rendering on main thread
        # renderer = ThreeDLanderRenderer(env,
        #                                 fun,
        #                                 (parts, nhid, movie_name),
        #                                 viewangles=None,
        #                                 outfile=movie_name)
        # renderer.start()
    # Save final net
    avg_reward, _ = eval_policy(policy, env, args.eval_episodes)
    _save(args, avg_reward, history, policy)


main()
