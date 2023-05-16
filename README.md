# Usage

first install gym comptor simulator by running
assuming othr dependencies are installed eg. gym, matplotlib, pytorch ...

```bash
chmod u+x run.sh
./run.sh -i
```

to test trained model default best model

```bash
./run.sh -r #tests best model
./run.sh -r models/<model_name> #for other models that didn't perform well

```

for ploting resut

```bash
./run.sh -p #plots best model
./run.sh -p runs/<model_name> #for other models that didn't perform well

```

for training with default params

```bash
./run.sh -t
```