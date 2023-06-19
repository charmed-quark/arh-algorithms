import math, random

datasets = { "chess":0.9, "mushrooms":0.4, "retail":0.001, "bms1":0.001, "syn1K":0.01, "syn10K":0.01, "syn100K":0.01 }

for dataset, minsup in datasets.items():

    print("Working on " + dataset + "...")

    dirpath = "../" + dataset + "/rules/"
    rulefile = dataset + "_arm.txt"
    num = 0
    lines = None
    dataset_size = 0

    with open("../" + dataset + "/" + dataset + ".txt") as db:
        dataset_size = len(db.readlines())

    minsup = math.ceil(0.9 * dataset_size)
    support_upper_bound = math.ceil(0.1 * (dataset_size - minsup)) + minsup
    support_avg_upper_bound = math.ceil(0.25 * (dataset_size - minsup)) + minsup

    seeds = [202306, 1111, 37868, 27, 4]

    with open(dirpath + rulefile) as file:
        lines = file.readlines()

    # extract 5% of rules
    sar_count = math.ceil(0.05 * len(lines))

    print("Extracting " + str(sar_count) + " sensitive rules from " + str(len(lines)) + " total...")

    for seed in seeds:
        print("Generating file using seed " + str(seed) + "...")
        random.seed(seed)  
        
        filename = dataset + "_sarweak_0" + str(num) + ".txt"
        line_nums = list(range(len(lines)))
        random.shuffle(line_nums) # randomize the order in which we access the lines

        with open(dirpath + filename, "w") as sar:
        
            removed = []

            for i in line_nums:
                support = lines[i].split('#')[1]     # get the string in format "#SUP: [Integer]"
                support = int(support.split()[1])   # extract the integer
                if (support <= support_upper_bound):
                    removed.append(i)
                    sar.write(lines[i])
            
                if (len(removed) == sar_count):
                    break
        
            # naive way of doing it but continue adding more rules.
            while (len(removed) < sar_count):
                for i in line_nums:
                    if i in removed:
                        continue
                support = lines[i].split('#')[1]     # get the string in format "#SUP: [Integer]"
                support = int(support.split()[1])   # extract the integer
                if (support <= support_avg_upper_bound):
                    removed.append(i)
                    sar.write(lines[i])
        
        num += 1

print("Done.")
print("#########################")