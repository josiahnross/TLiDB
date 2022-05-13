import random
import os
seed = 31415
random.seed(seed)
path = "../TLiDB/data/TLiDB_Friends/"
seedPath = path + f"seed.{seed}/"
if not os.path.exists(seedPath):
    os.makedirs(seedPath)
percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

with open(path + "TTiDB_dev_ids.txt") as file:
    devLines = [line.rstrip() for line in file]

with open(path + "TTiDB_train_ids.txt") as file:
    trainLines = [line.rstrip() for line in file]

devSize = len(devLines)
trainSize = len(trainLines)

lastDevLines = []
lastTrainLines = []
def SaveLines(path, lines):
    with open(path, 'w') as file:
        for l in lines:
            file.write(l)
            file.write("\n")

for p in percents:
    numDev = int(round(devSize * p))
    numTrain = int(round(trainSize * p))
    while len(lastDevLines) < numDev:
        rnd = random.randint(0, len(devLines)-1)
        lastDevLines.append(devLines.pop(rnd))
    SaveLines(seedPath + f"TTiDB_percent_{p}_dev_ids.txt", lastDevLines)
    while len(lastTrainLines) < numTrain:
        rnd = random.randint(0, len(trainLines)-1)
        lastTrainLines.append(trainLines.pop(rnd))
    SaveLines(seedPath + f"TTiDB_percent_{p}_train_ids.txt", lastTrainLines)