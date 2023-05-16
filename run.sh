#!/bin/bash
# if flag i is set proceed to clone git repo and install it
if [ "$1" = "-i" ]; then
    git clone https://github.com/simondlevy/gym-copter.git
    pip install -e gym-copter/
    rm -rf gym-copter/
fi

# if flag r is set proceed to run the program with default model
if [ "$1" = "-r" ]; then
    python3 model_test.py models/td3-gym_copter:Lander3D-v0+00253.123_best.dat
fi
# if r flag is present with a model name, run the program with that model
if [ "$1" = "-r" ] && [ "$2" != "" ]; then
    python3 model_test.py $2
fi
# if flag t is set proceed to train the model
if [ "$1" = "-t" ]; then
    python3 model_train.py 
fi


# for plotting the results -p flag is used
if [ "$1" = "-p" ]; then
    python3 plot.py runs/td3-gym_copter:Lander3D-v0+00253.123_best.csv
fi
# p flag with a model name, plot the results of that model
if [ "$1" = "-p" ] && [ "$2" != "" ]; then
    python3 plot.py $2
fi
# if flag h is set proceed to print the help
if [ "$1" = "-h" ]; then
    echo "Usage: ./run.sh [OPTION] [MODEL]"
    echo "Run the program with the specified option"
    echo " "
    echo "Options:"
    echo "-i    Install the required dependencies"
    echo "-r    Run the program with the default model"
    echo "-r [MODEL]    Run the program with the specified model"
    echo "-t    Train the model"
    echo "-p    Plot the results"
    echo "-h    Print this help"
fi