#!/usr/bin/env python3
import numpy as np
import torch
import gym

from gym_copter.cmdline import make_parser_3d, parse_view_angles


from td3 import TD3, eval_policy

from gym_copter.rendering.threed import ThreeDLanderRenderer


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

    # Make a command-line parser
    parser = make_parser_3d()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    args = parser.parse_args()
    viewangles = parse_view_angles(args)

    # Load network, environment name, and number of hidden units from pickled
    # file
    parts, env_name, nhid = torch.load(open(args.filename, 'rb'))
    
    # Make a gym environment from the name
    env = gym.make(env_name)

    # Set random seed if indicated
    if args.seed is not None:
        env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    fun = run_td3

    movie_name = None

    if args.movie:
        print('Running episode ...')
        movie_name = 'movie.mp4'

    # Begin 3D rendering on main thread
    renderer = ThreeDLanderRenderer(env,
                                    fun,
                                    (parts, nhid, movie_name),
                                    viewangles=viewangles,
                                    outfile=movie_name)
    renderer.start()


if __name__ == '__main__':
    main()
