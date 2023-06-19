from collections import Counter
import sys

print(">>> Running script to compute ARH side effects...")

# SETUP

directory = "/home/charlie/thesis"
dataset = sys.argv[1]
algorithm = sys.argv[2]

inputs = {"random00":"sar_00", "random01":"sar_01", "weakfreq00":"sarweak_00", "weakfreq01":"sarweak_01"}

for data, sarfile in inputs.items():

    for num in range(1):
        
        original_database_filepath = directory + "/" + dataset + "/" + dataset + ".txt" # e.g. chess/chess.txt
        sanitized_database_filepath = directory + "/" + dataset + "/measurements/" + algorithm + "/" + algorithm + "_" + data + "_sanitized_0" + str(num) + ".txt"
        sar_filepath = directory + "/" + dataset + "/rules/" + dataset + "_" + sarfile + ".txt"
        arm_original_filepath = directory + "/" + dataset + "/rules/" + dataset + "_arm.txt"
        arm_sanitized_filepath = directory + "/" + dataset + "/measurements/" + algorithm + "/" + algorithm + "_" + data + "_sanitized_0" + str(num) + "_mined.txt"
        arm_sanitized_filepath = directory + "/" + dataset + "/measurements/" + algorithm + "/" + algorithm + "_" + data + "_sanitized_0" + str(num) + "_minedCLEAN.txt"
        

        # VARIABLES
        size_og = 0                     # |D|
        size_san = 0                    # |D'|
        sensitive_rule_count_og = 0     # |P_s|
        sensitive_rule_count_san = 0    # |P'_s|
        nonsensitive_rule_count_og = 0  # |NOT P_s|
        nonsensitive_rule_count_san = 0 # |NOT P'_s|
        rules_in_both = 0

        # Get files...
        f_og = open(arm_original_filepath, "r") 
        f_san = open(arm_sanitized_filepath, "r") 
        f_sar = open(sar_filepath, "r") 

        arm_original = f_og.readlines()
        arm_sanitized = f_san.readlines()
        sensitive_rules = f_sar.readlines()

        sensitive_rule_count_og = len(sensitive_rules)
        nonsensitive_rule_count_og = len(arm_original) - sensitive_rule_count_og

        # reading files to get the sensitive rules minable from sanitized db
        print(">>>>> Checking for hiding failures...")
        for line1 in sensitive_rules:
            l1 = line1.split('#', 1)[0].strip()
            
            for line2 in arm_sanitized:
                l2 = line2.split('#', 1)[0].strip()
                if l1 == l2:
                    #print("Sensitive rule " + l1 + " was discoverable in the sanitized database.")
                    sensitive_rule_count_san += 1      
                    
        nonsensitive_rule_count_san = len(arm_sanitized) - sensitive_rule_count_san
        rule_count_san = len(arm_sanitized)

        # reading files to get the rules that are mined from both dbs
        print(">>>>> Finding intersection of mined rules...") 
        for line1 in arm_original:
            l1 = line1.split('#', 1)[0]
            
            for line2 in arm_sanitized:
                l2 = line2.split('#', 1)[0]
                if l1 == l2:
                    #print("Rule " + l1 + " was mined from both databases.")
                    rules_in_both += 1
        
        # closing files
        f_sar.close() 
        f_san.close() 
        f_og.close() 

        count_og = Counter()
        count_san = Counter()

        with open(original_database_filepath) as f: #the original databases dont have any empty lines
            for line in f:
                size_og += 1
                l = line.split('#', 1)[0].split()
                count_og.update(Counter(l))

        with open(sanitized_database_filepath) as f:
            for line in f:
                if line not in ['\n', '\r\n']:  # empty lines mean a transaction has been deleted
                    size_san += 1
                l = line.split('#', 1)[0].split()
                count_san.update(Counter(l))


        # COMPUTE SIDE EFFECTS
        hiding_failure = sensitive_rule_count_san / sensitive_rule_count_og
        # for misses cost, we don't want to include any potential artificial patterns. so we add the number of discovered APs, if any
        # otherwise they would inflate the nonsensitive rule count in the sanitized db. needs to be tested though. lol its failing again...
        misses_cost = ((nonsensitive_rule_count_og - nonsensitive_rule_count_san) + (rule_count_san - rules_in_both)) / nonsensitive_rule_count_og
        artificial_patterns = (rule_count_san - rules_in_both) / rule_count_san   # read size of arm_sanitized
        item_diss = sum((count_og - count_san).values()) / sum(count_og.values())
        transaction_diss = 1 - (size_san / size_og)

        # PRESENT RESULTS
        print("\n##### RESULTS #####")
        print("DATASET: " + dataset)
        print("ALGORITHM: " + algorithm)
        print("INPUT : " + data + "; RUN: " + str(num))
        print("Hiding Failure: " + str(hiding_failure))
        print("Misses Cost: " + str(misses_cost))
        print("Artificial Patterns: " + str(artificial_patterns))
        print("Item-level Database Dissimilarity: " + str(item_diss))
        print("Transaction-level Database Dissimilarity: " + str(transaction_diss))
        print("")
        print("MISC STATS:")
        print("rule count og: " + str(sensitive_rule_count_og + nonsensitive_rule_count_og))
        print("s rule count og: " + str(sensitive_rule_count_og))
        print("ns rule count og: " + str(nonsensitive_rule_count_og))
        print("rule count san: " + str(rule_count_san))
        print("s rule count san: " + str(sensitive_rule_count_san))
        print("ns rule count san: " + str(nonsensitive_rule_count_san))
        print("rules in intersection of dbs: " + str(rules_in_both))
        print("##### DONE. #####")