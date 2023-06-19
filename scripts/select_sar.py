import math, random

datasets = ["chess", "mushrooms", "retail", "bms1", "syn1K", "syn10K", "syn100K"]

for dataset in datasets:

    print("Working on " + dataset + "...")

    dirpath = "../" + dataset + "/rules/"
    rulefile = dataset + "_arm.txt"
    num = 0
    lines = None

    seeds = [202306, 1111, 37868, 27, 4]

    with open(dirpath + rulefile) as file:
        lines = file.readlines()

    # extract 5% of rules
    count = math.ceil(0.05 * len(lines))

    print("Extracting " + str(count) + " sensitive rules from " + str(len(lines)) + " total...")

    for seed in seeds:
        print("Generating file using seed " + str(seed) + "...")
        random.seed(seed)  
        filename = dataset + "_sar_0" + str(num) + ".txt"
        line_nums = random.sample(range(len(lines)), count) # get count unique indices
        with open(dirpath + filename, "w") as sar:
            for i in line_nums:
                line = lines[i]
                sar.write(line)
        
        num += 1

print("Done.")
print("#########################")