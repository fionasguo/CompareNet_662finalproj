# CompareNet experiment log

## Config on discovery cluster:

salloc --partition=gpu --gres=gpu:v100:1 --nodes=1 --cpus-per-task=16 --mem-per-cpu=6GB --time=2:00:00

or use following .sh script template:

  #!/bin/bash
  #SBATCH --partition=gpu
  #SBATCH --gres=gpu:v100:1
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=16
  #SBATCH --mem-per-cpu=10GB
  #SBATCH --time=48:00:00
  #SBATCH --account=lerman_316

  module purge
  module load gcc/8.3.0
  module load cuda/10.1.243
  module load cudnn/8.0.2-10.1
  source comparenet_env/bin/activate
  module load python/3.7.6

  python main.py --mode 0 -s 5 --bert_encoder 1


labels in LUN:
1 == Satire
2 == hoax
3 == propaganda
4 == trusted

## Experimental runs

### basic experiments

run_1:
test run
seed=5
default setting: python main.py --mode 0

run_2:
in domain 4-way classification task
repeat 5 times; seeds: [42,91,30,72,5]
train = fulltrain, val = balancedtest, test = test.xlsx (default)

run_3: ABORT
out of domain 2-way classification task
only take trusted and satire in the LUN data for training, still test on entire SLN
repeat 5 times; seeds: [42,91,30,72,5]
2-way classification setting: train = fulltrain_2classes, val = balancedtest_2classes, test = test.xlsx

run_4: ABORT
in domain 4-way classification task
repeat 5 times; seeds: [42,91,30,72,5]
train = 80% LUN-train, val = 20% LUN-train, test = LUN-test

  -> error:
  Traceback (most recent call last):
    File "main.py", line 142, in <module>
      main(params)
    File "main.py", line 104, in main
      dl = DataLoader(params)
    File "/home1/siyiguo/662final/CompareNet/data_loader.py", line 84, in __init__
      dataset_train = DataSet(self.train, self.adj_train, self.fea_train, self.params, self.entity_description)
    File "/home1/siyiguo/662final/CompareNet/data_loader.py", line 264, in __init__
      "dim of adj does not match the num of sent, where the idx is {}".format(i)
  AssertionError: dim of adj does not match the num of sent, where the idx is 0

run_5:
out of domain 2-way classification task - only take trusted and satire in the LUN data for training, still test on entire SLN
repeat 5 times; seeds: [42,91,30,72,5]
train = fulltrain, val = balancedtest, test = test.xlsx, ntag=2

### ablation study

  parser.add_argument("--node_type", type=int, default=3,
                        help='3 represents three types: Document&Entity&Topic; \n'
                             '2 represents two types: Document&Entiy; \n'
                             '1 represents two types: Document&Topic; \n'
                             '0 represents only one type: Document. ')

run_6:
ablation - w/o entity comp
repeat 5 times; seeds: [42,91,30,72,5]
train = fulltrain, val = balancedtest, test = test.xlsx, ntag=4, —node_type=1

run_7:
ablation - w/o topics
repeat 5 times; seeds: [42,91,30,72,5]
train = fulltrain, val = balancedtest, test = test.xlsx, ntag=4, —node_type=2

run_8:
ablation - w/o topics and entity comp
repeat 5 times; seeds: [42,91,30,72,5]
train = fulltrain, val = balancedtest, test = test.xlsx, ntag=4,—node_type=0

run_9:
ablation - w/o structured triplets in entity representation (3.3.1 in paper)
repeat 5 times; seeds: [42,91,30,72,5]
change the gating mechanism in the code:
  in models/model.py, on line 201
    from torch.mul(gate, X) + torch.mul(-gate + 1, Y) to Y
    train = fulltrain, val = balancedtest, test = test.xlsx, ntag=4,—node_type=3 (default setting)

run_10:
training size smaller -dev set size = 0.6, train = 0.4
repeat 5 times; seeds: [42,91,30,72,5]

run_11:
training size smaller - dev set size = 0.8, train = 0.2
repeat 5 times; seeds: [42,91,30,72,5]

run_12: ABORT not successful
use bert encoder for input documents and entity desccription - use branch bert
repeat 5 times; seeds: [42,91,30,72,5]

run_13:
use bert encoder for input documents and entity desccription - use branch bert
**no grad update for bert_encoder sentence embedding layer**
repeat 1 time; seeds: 91

run_14:
use bert encoder for input documents and entity desccription - use branch bert
**no grad update for bert_encoder sentence embedding layer**
repeat 5 times; seeds: [42,91,30,72,5]
