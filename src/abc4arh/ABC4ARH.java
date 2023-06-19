package abc4arh;

/** 
 *  ARTIFICIAL BEE COLONY ALGORITHM FOR ASSOCIATION RULE HIDING (ABC4ARH)
 *  Based on the algorithm described in the paper by Telikani et al. (2019)
 * 
 *  This source code is based on a modified version of a file from the 
 *  SPMF DATA MINING SOFTWARE (http://www.philippe-fournier-viger.com/spmf).
 *  Copyright (C) 2023 Charlie Mårtensson
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;
import java.util.stream.DoubleStream;
import java.util.stream.Collectors;

/**
 * An implementation of the ABC4ARH algorithm for hiding sensitive association
 * rules in a transaction database. 
 * @author Charlie Mårtensson
 * 
 * Some code is based on or borrowed from SPMF library by:
 * @author Philippe Fournier-Viger
 * @author Hoang Thi Dieu (2019-improvement)
 * 
 */
public class ABC4ARH {

	// variables for statistics
	int tidcount = 0; // the number of transactions in the last database read
	long startTimestamp = 0; // the start time of the last execution
	long endTimeStamp = 0; // the end time of the last execution

	final int POP_SIZE = 20;
	final int MAX_CYCLES = 15;
    final int MAX_TRIALS = 5;
    int[] trial;    // tracks the # of attempted improvements for each solution
    double[] fitness;    // tracks the latest fitness for each solution

    double minsup;
    double minconf;

    // Data structures mapping transactions to the rules they support
    Map<Transaction, Set<Rule>> tidSensitive; 
    Map<Transaction, Set<Rule>> tidNonsensitive; 
    //Map<Transaction, Set<Rule>> tidWeak; 
    //Map<Transaction, Set<Rule>> tidWeakLeftSide; 

    // Lists mapping items to rules containing them
    Map<Integer, Set<Rule>> itemIndexSensitive;
    Map<Integer, Set<Rule>> itemIndexNonsensitive;
    Map<Integer, Set<Rule>> itemIndexWeakLeftSide;

    Map<BitSet, Double> fitnessCache;

	/**
	 * Run the SA-MDP algorithm
	 * 
	 * @param input
	 *            the file path to a transaction database
	 * @param inputSensitive
	 *            the file path to a set of sensitive association rules to be
	 *            hidden
	 * @param inputNonsensitive
	 * 			  the file path to the set of nonsensitive mined rules
     * @param inputWeak
     *            the file path to a set of weak rules, meaning rules whose
     *            generating itemsets are frequent but whose confidence < minconf
	 * @param output
	 *            the output file path for writing the modified transaction
	 *            database
	 * @param minsup
	 *            the minimum support threshold
	 * @param minconf
	 *            the minimum confidence threshold
	 * @throws IOException
	 *             exception if an error while writing the file
	 */
	public void runAlgorithm(String input, String inputSensitive, String inputNonsensitive,
            String inputWeak, String output,
			double minsup, double minconf) throws IOException {
		// record the start time
		startTimestamp = System.currentTimeMillis();

        this.minsup = minsup;
        this.minconf = minconf;

        List<Rule> sensitiveRules = new ArrayList<>();
        List<Rule> nonsensitiveRules = new ArrayList<>();
        List<Rule> weakRules = new ArrayList<>();

        fitnessCache = new HashMap<>();

		/* STEP 1: PREPROCESSING */
        readRulesIntoMemory(inputSensitive, sensitiveRules);
        readRulesIntoMemory(inputNonsensitive, nonsensitiveRules);
        readRulesIntoMemory(inputWeak, weakRules);

        tidSensitive = new HashMap<>(); 
        tidNonsensitive = new HashMap<>(); 
        //tidWeak = new HashMap<>(); 
        //tidWeakLeftSide = new HashMap<>(); 

        itemIndexSensitive = new HashMap<>();
        itemIndexNonsensitive = new HashMap<>();
        itemIndexWeakLeftSide = new HashMap<>();

        List<Transaction> sensitiveTransactions = new ArrayList<>();

        String line;
		BufferedReader reader = new BufferedReader(new FileReader(input));

		// Identify the critical transactions and calculate rule support in noncritical transactions
        System.out.println("Reading database...");
		while (((line = reader.readLine()) != null)) {

            if(line.isEmpty()) continue;

			tidcount++;

            String[] transactionItems = line.split(" ");

			Set<Integer> items = new HashSet<>(transactionItems.length);

            // for each item in the current transaction
			for (int i = 0; i < transactionItems.length; i++) {
				// convert from string to int
				int item = Integer.parseInt(transactionItems[i]);
				// add it to the transaction
				items.add(item);
			}

            Transaction t = new Transaction(tidcount, items);

            boolean isSensitive = false;
            
            /** 
			 * Go through every SAR and see if this transaction supports any of them.
			 * If yes, the transaction is sensitive.
			 */
			for (Rule rule : sensitiveRules) {

				Set<Integer> matchLeft = new HashSet<>();
				Set<Integer> matchRight = new HashSet<>();

				for (Integer i : t.items) {

					// if the left side of this sensitive rule matches with this
					// transaction
					if (matchLeft.size() != rule.leftSide.size()
							&& rule.leftSide.contains(i)) {
						matchLeft.add(i);
					} // else if the item appears in the right side of this
						// transaction
						// but we have not seen all items from the right side
						// yet
					else if (matchRight.size() != rule.rightSide.size()
							&& rule.rightSide.contains(i)) {
						matchRight.add(i);
					}

					// if the rule completely matches with this transaction...
					// (both left and right sides)
					if (matchLeft.size() == rule.leftSide.size()
							&& matchRight.size() == rule.rightSide.size()) {
						// increase the support of the rule
						//rule.support++;
						if(!isSensitive) {
							isSensitive = true;
							sensitiveTransactions.add(t);
						}
						break;
					}
				}
			}

            if (isSensitive) continue;

            // If the transaction is NOT sensitive, we need to compute the
            // support it may provide for other rules.
            // These values are added to the support computed from sensitive
            // transactions.
            // Remember: a nonsensitive transaction doesn't support any
            // sensitive rules in full, but it MAY support their antecedent.
            // Furthermore, it might support nonsensitive rules and weak rules,
            // either only the antecedent or in full.

            // Compute antecedent support of sensitive rules
            for (Rule rule : sensitiveRules) {
				Set<Integer> matchLeft = new HashSet<>();
				for (Integer i : t.items) {
					// if the left side of this sensitive rule matches with this
					// transaction
					if (matchLeft.size() != rule.leftSide.size()
							&& rule.leftSide.contains(i)) {
						matchLeft.add(i);
						if (matchLeft.size() == rule.leftSide.size()) {
							rule.leftSideSupportNonSensitive++;
						}
					}
				}
			}

            // Compute support of nonsensitive rules and weak rules in nonsensitive transaction
            // By definition a sensitive rule can't be suppored by a nonsensitive transaction
            findRuleSupportInTransaction(t, nonsensitiveRules);
            findRuleSupportInTransaction(t, weakRules);

        }   // Finish reading all transactions

        System.out.println("Generating index files...");
        // Generate the index files.
        for (Transaction t : sensitiveTransactions) {

            addToMapping(t, sensitiveRules, tidSensitive);
            addToMapping(t, nonsensitiveRules, tidNonsensitive);
        }

            // Special case for weak rules
            /*for (Rule r : weakRules) {
                // check if this transaction supports the antecedent of the rule
                if (t.items.containsAll(r.leftSide)) {

                    // if the transaction also supports the consequent, add to weak rule mapping
                    if(t.items.containsAll(r.rightSide)) {
                        if (tidWeak.containsKey(t)) {
                            // add this rule to the mapping
                            tidWeak.get(t).add(r);
                        } else {
                            // initialize the mapping with the transaction as key
                            Set<Rule> ruleSet = new HashSet<>();
                            ruleSet.add(r);
                            tidWeak.put(t, ruleSet);
                        }
                    } 
                    
                    // add to the antecedent mapping
                    if (tidWeakLeftSide.containsKey(t)) {
                        // add this rule to the mapping
                        tidWeakLeftSide.get(t).add(r);
                    } else {
                        // initialize the mapping with the transaction as key
                        Set<Rule> ruleSet = new HashSet<>();
                        ruleSet.add(r);
                        tidWeakLeftSide.put(t, ruleSet);
                    }
                }
            }
        }*/

        System.out.println("Populating item indices...");
        // Populate the item indices, tracking all rules containing items
        // that appear on the right side of a sensitive rule.
        // First initialize the maps with all the relevant keys
        for (Rule r : sensitiveRules) {
            for (Integer i : r.rightSide) {
                itemIndexSensitive.putIfAbsent(i, new HashSet<Rule>());
                itemIndexNonsensitive.putIfAbsent(i, new HashSet<Rule>());
                itemIndexWeakLeftSide.putIfAbsent(i, new HashSet<Rule>());
            }
        }

        // Because they all have the same keys, iterate over the keyset of one map
        // and go through each rule list, adding the rules that are relevant.
        for (Integer i : itemIndexSensitive.keySet()) {
            for (Rule r : sensitiveRules) {
                if (r.rightSide.contains(i) || r.leftSide.contains(i))
                    itemIndexSensitive.get(i).add(r);
            }
            for (Rule r : nonsensitiveRules) {
                if (r.rightSide.contains(i) || r.leftSide.contains(i))
                    itemIndexNonsensitive.get(i).add(r);
            }
            for (Rule r : weakRules) {
                if (r.leftSide.contains(i))
                    itemIndexWeakLeftSide.get(i).add(r);
            }
        }


        // Calculate N_iter_conf and N_iter_supp for each sensitive rule
        double maxSanitizationRatio = -1;

        for (Rule r : sensitiveRules) {
            double support = r.supportNonSensitive;
            double supportLHS = r.leftSideSupportNonSensitive;

            // Each transaction containing the generating itemset of the rule increases the support by 1
            for (Map.Entry<Transaction, Set<Rule>> entry : tidSensitive.entrySet()) {
                if (entry.getValue().contains(r))
                    support++;
                if (entry.getKey().items.containsAll(r.leftSide))
                    supportLHS++;
            }

            // For nIterConf we want the support as a fraction, for nIterSupp we want it as an integer count.
            double supportFrac = (double) support / tidcount;
            double supportLHSFrac = (double) supportLHS / tidcount;

            double nIterConf = Math.ceil(tidcount * ((supportFrac / minconf) - supportLHSFrac));
            double nIterSupp = (support == minsup * tidcount) ? 1 : Math.ceil(tidcount - tidcount * ((minsup * tidcount) / support));

            r.iterations = Math.min(nIterConf, nIterSupp);

            // Apparently a bug can happen where the sanitization ratio exceeds the number of sensitive transactions.
            // the maximum sanitization ratio is the largest value among the minimum number of
            // modifications required to hide each rule. obviously since we only modify sensitive
            // transactions the number cannot be larger than the number of sensitive transactions.
            if (r.iterations < sensitiveTransactions.size() && r.iterations > maxSanitizationRatio)
                maxSanitizationRatio = r.iterations;
        }


        System.out.println("Initializing population...");
        /* Initialize population */
        // Each solution is encoded as a binary vector of length sensitiveTransactions.size()
        int solutionSize = sensitiveTransactions.size();
        Random rand = new Random();
        BitSet bestSol = new BitSet();
        double bestFit = Integer.MAX_VALUE;
        List<BitSet> population = new ArrayList<>();

        trial = new int[POP_SIZE];   // keeps track of whether to abandon a food source
        Arrays.fill(trial, 0);
        fitness = new double[POP_SIZE];

        for (int i = 0; i < POP_SIZE; i++) {
            BitSet foodSource = initializeFoodSource(solutionSize, maxSanitizationRatio);
            population.add(foodSource);

            if (fitnessCache.containsKey(foodSource)) {
				fitness[i] = fitnessCache.get(foodSource);
			} else {
                fitness[i] = calculateFitness(foodSource, sensitiveRules, nonsensitiveRules, sensitiveTransactions);
				fitnessCache.put(foodSource, fitness[i]);
			}

            if (fitness[i] < bestFit) {
                bestFit = fitness[i];
                bestSol = foodSource;
            }
        }

        System.out.println("Starting main algorithm");

        /* MAIN ALGORITHM */
        for (int iter = 0; iter < MAX_CYCLES; iter++) {
            System.out.println("ITERATION: " + iter);

            // In the paper, phi is defined as phi_max - ((phi_max - phi_min)/MAX_CYCLES) * iter.
            // Most sources seem to define phi as a number ranging between 1 and -1 (hence max and min)
            double phi = (double) 2 / MAX_CYCLES;
            phi *= iter;
            phi = 1 - phi;
            //double theta = 1;
            sendEmployedBees(population, sensitiveRules, nonsensitiveRules,
                sensitiveTransactions, bestSol, phi);
            sendOnlookerBees(population, sensitiveRules, nonsensitiveRules,
                sensitiveTransactions, bestSol, phi);
            
            // Update best solution
            for (int i = 0; i < POP_SIZE; i++) {
                if (fitness[i] < bestFit) {
                    bestFit = fitness[i];
                    bestSol = population.get(i);
                }
            }

            sendScoutBees(population, solutionSize, maxSanitizationRatio);
        }

        // save the end time. don't bother with time required to write to file.
		endTimeStamp = System.currentTimeMillis();

        // Sanitize the database.
        System.out.println("Sanitizing DB...");

        List<Integer> indices = getIndicesOfSetBits(bestSol, sensitiveTransactions.size());

        for (Rule sr : sensitiveRules) {
            // this rule cannot be hidden if there aren't enough transactions selected for sanitization to hide it.
            if (sr.iterations > indices.size()) 
                continue;

            Collections.shuffle(indices);
            int idx = 0;
            int victim = selectVictimItem(sr);
            
            for (int i = 0; i < Math.min(sr.iterations, indices.size()); i++) {
                boolean isModified = false; // look through transactions until a modification can be made
                while (!isModified && idx < indices.size()) {
                    Transaction t = sensitiveTransactions.get(indices.get(idx));
                    if (tidSensitive.get(t).contains(sr)) {
                        t.items.remove(victim);
                        isModified = true;
                    }
                    idx++;
                }
            }
        }
        
		writeResultToFile(sensitiveTransactions, input, output);

		System.out.println("Best fitness: " + bestFit);
		System.out.println("Time: " + (endTimeStamp - startTimestamp));
	}


    // Helper to compute the support of rules in nonsensitive transactions
    private void findRuleSupportInTransaction(Transaction t, List<Rule> ruleList) {
        for (Rule rule : ruleList) {
			Set<Integer> matchLeft = new HashSet<>();
			Set<Integer> matchRight = new HashSet<>();
			for (Integer i : t.items) 
				// if the left side of this nonsensitive rule matches with this
				// transaction
				if (matchLeft.size() != rule.leftSide.size()
						&& rule.leftSide.contains(i)) {
					matchLeft.add(i);
					if (matchLeft.size() == rule.leftSide.size()) {
						rule.leftSideSupportNonSensitive++;
					}
				}   // else if the item appears in the right side of this
					// transaction
					// but we have not seen all items from the right side
					// yet
				else if (matchRight.size() != rule.rightSide.size()
						&& rule.rightSide.contains(i)) {
					matchRight.add(i);
				
				// if the rule completely matches with this transaction...
				// (both left and right sides)
				if (matchLeft.size() == rule.leftSide.size()
						&& matchRight.size() == rule.rightSide.size()) {
					// increase the support of the rule
					rule.supportNonSensitive++;
					break;
				}
			}
		}
    }

    private BitSet initializeFoodSource(int solutionSize, double maxSanitizationRatio) {
        BitSet foodSource = new BitSet(solutionSize);
        
        Random rand = new Random();

        // we want maxSanitizationRatio total modifications to be made
        int count = 0;
        for (int i = 0; i < maxSanitizationRatio; i++) {
            // Select a random dimension
            int dim = rand.nextInt(solutionSize);
            
            // Make sure to find a dimension that is 0
            while (foodSource.get(dim)) {
                dim = rand.nextInt(solutionSize);  
                count++;
            }
            //System.out.print(" " + i);
            foodSource.set(dim);

            // The modified bernoulli process described in the paper doesnt seem to actually be used

            // If the corresponding bit is 0, invert it.
            //if (!foodSource.get(dim)) foodSource.flip(dim);

            // If the random value between 0 and 1 is 1, set the bit; otherwise clear it.
            /*if (rand.nextDouble() > 0.5) {
                foodSource.set(dim);
            } else {
                foodSource.clear(dim);
            }*/
        }
        return foodSource;        
    }

	private double calculateFitness(BitSet solution, List<Rule> sensitiveRules, List<Rule> nonsensitiveRules,
                                    List<Transaction> sensitiveTransactions) {

		double w1 = 0.9;
		double w2 = 0.2;
		double w3 = 0.1;

        double hf = 0.0;
        double mc = 0.0;
        double ap = 0.0;
        double rhd = 0.0;
        double rld = 0.0;

        // First, generate the candidate solution.
        List<Transaction> sanitizedTransactions = new ArrayList<>();
        for (Transaction t : sensitiveTransactions) {
            Transaction tNew = new Transaction(t.id, t.items);  // MAKE SURE the original set is not affected!!
            sanitizedTransactions.add(tNew);
        }

        // For each rule, we need to make a certain number of modifications to hide it, calculated as N_iter.
        // Only look at those indices that are applicable in the solution.
        List<Integer> indices = getIndicesOfSetBits(solution, sensitiveTransactions.size());

        for (Rule sr : sensitiveRules) {
            Collections.shuffle(indices);
            int idx = 0;
            int victim = selectVictimItem(sr);
            for (int i = 0; i < Math.min(sr.iterations, indices.size()); i++) {
                boolean isModified = false;
                while (!isModified && idx < indices.size()) {
                    int targetTransaction = indices.get(idx);
                    // this bug can occur. not sure why
                    if (targetTransaction > sensitiveTransactions.size()) {
                        idx++;
                        continue;
                    }
                    Transaction tOriginal = sensitiveTransactions.get(targetTransaction);
                    Transaction tSanitized = sanitizedTransactions.get(targetTransaction);
                    if (tidSensitive.get(tOriginal).contains(sr)) { //index list is consulted to see if rule is included in transaction
                        tSanitized.items.remove(victim);
                        isModified = true;
                    }
                    idx++;
                }
            }
        }

        // Determine the support and confidence of sensitive and nonsensitive rules in the new db
        updateSupport(sensitiveRules, sanitizedTransactions);
        updateSupport(nonsensitiveRules, sanitizedTransactions);
        //updateSupport(weakRules, sanitizedTransactions);

        // Determine HF, MC, and AP as described by the fitness function in the paper
        for (Rule r : sensitiveRules) {
			double[] supConf = getSupAndConf(r);
			if (supConf[0] >= minsup && supConf[1] >= minconf) { 
                hf++;
                rhd += supConf[1] - minconf + 1;
            }
        }
        for (Rule r : nonsensitiveRules) {
            double[] supConf = getSupAndConf(r);
			if (supConf[0] < minsup || supConf[1] < minconf) {
                mc++;
            }

            if (supConf[0] >= minsup || supConf[1] >= minconf) {
                rld++;
            } else if (supConf[1] < minconf) {
                rld += (minconf - supConf[1]);
            } else {
                rld += (minsup - supConf[0]);
            }
        }

        // normalize
        hf /= sensitiveRules.size();
        mc /= nonsensitiveRules.size();
        ap = (rhd / sensitiveRules.size()) + (rld / nonsensitiveRules.size());
        ap /= 2;

        if (hf == 1.0) hf += 100;
        if (mc == 1.0) mc += 100;
        
		return w1 * hf + w2 * mc + w3 * ap;
	}

    // This function computes the support of rules in the sanitized transactions to
    // determine the fitness of the sanitized database.
    // The support in nonsensitive (and therefore definitely unaltered) transactions
    // has already been computed for each rule in the preprocessing stage.
    private void updateSupport(List<Rule> rules, List<Transaction> sanitizedTransactions) {

        // Iterate through the sanitized transactions and
        // see if they support this rule
        for (Rule r : rules) {
            r.leftSideSupportSensitive = 0;
            r.supportSensitive = 0; // reset the value
            for (Transaction t : sanitizedTransactions) {
                if (t.items.containsAll(r.leftSide)) {
                    r.leftSideSupportSensitive++;
                    if (t.items.containsAll(r.rightSide))
                        r.supportSensitive++;
                }
            }
        }
    }

    private double[] getSupAndConf(Rule r) {
        int support = r.supportSensitive + r.supportNonSensitive;
        int lhsCount = r.leftSideSupportSensitive + r.leftSideSupportNonSensitive;
        double sup = (double) support / tidcount;
		double conf = lhsCount == 0 ? 0 : (double) support / lhsCount;

        return new double[]{sup, conf};
    }

    private void addToMapping(Transaction t, List<Rule> ruleList, Map<Transaction, Set<Rule>> map) {

        for (Rule r : ruleList) {
            // check if this transaction supports a rule
            if (t.items.containsAll(r.leftSide)
                && t.items.containsAll(r.rightSide)) 
            {
                if (map.containsKey(t)) {
                    // add this rule to the mapping
                    map.get(t).add(r);
                } else {
                    // initialize the mapping with the transaction as key
                    Set<Rule> ruleSet = new HashSet<>();
                    ruleSet.add(r);
                    map.put(t, ruleSet);
                }
            }
        }

    }

    private void sendEmployedBees(List<BitSet> population, List<Rule> sensitiveRules, List<Rule> nonsensitiveRules,
        List<Transaction> sensitiveTransactions, BitSet x_best, double theta) {

        System.out.println("Sending employed bees...");

        int M11 = 0;
        int M01 = 0;
        int M10 = 0;

        BitSet x_r1;
        BitSet x_r2;
        BitSet x_r;
        BitSet x_i;
        BitSet v;
        int r1;
        int r2;
        int numZeroBits;
        int numOneBits;
        
        // List of all food sources so that two random solutions can be chosen
        List<Integer> indices = IntStream.range(0, population.size())
									.boxed()
									.collect(Collectors.toList());

        for (int i = 0; i < POP_SIZE; i++) {
            System.out.print(i + " ");
            Collections.shuffle(indices);

            // population must be of size at least 3
            // it's not possible for both indices[0] and indices[1] to be i,
            // so only one of r1 and r2 will be assigned indices[2]
            r1 = (indices.get(0) == i) ? indices.get(2) : indices.get(0);
            r2 = (indices.get(1) == i) ? indices.get(2) : indices.get(1);

            x_r = new BitSet();
            x_r.or(population.get(r1));

            x_r2 = new BitSet();
            x_r2.or(population.get(r2));

            x_r.and(x_r2);

            x_i = population.get(i);

            v = new BitSet(sensitiveTransactions.size());   // initialize the new solution as zero vector

            numOneBits = x_i.cardinality();
            numZeroBits = sensitiveTransactions.size() - numOneBits;

            double dissimilarity = theta * dissimilarity(x_i, x_r, sensitiveTransactions.size());
            double bestCombination = Integer.MAX_VALUE;
            double d;

            // Find the optimal values of M11, M10, M01
            // Try all possible positive integer combinations given constraints
            // m11 + m01 = numOneBits and m10 <= numZeroBits
            for (int m11 = 0; m11 <= numOneBits; m11++) {
                int m01 = numOneBits - m11;
                for (int m10 = 0; m10 <= numZeroBits; m10++) {
                    d = (m11 == 0) ? 1 : 1 - (m11 / (m11 + m01 + m10));
                    if ((d - dissimilarity) < bestCombination) {
                        bestCombination = d - dissimilarity;
                        M11 = m11;
                        M01 = m01;
                        M10 = m10;
                    }
                }
            }

            // Generate the new solution v

            // Set M11 bits in v to 1 if they are 1 in both x_i and x_r

            // Get all the bits that are 1 in x_i and put their indices in a random order.
            //System.out.println("Set M11 ("+M11+") bits in v to 1 if they are 1 in both x_i and x_r");
            List<Integer> setBits = getIndicesOfSetBits(x_i, sensitiveTransactions.size());
            Collections.shuffle(setBits);
            int modifications = 0;
            for (int j = 0; j < Math.min(M11, setBits.size()); j++) {
                int idx = setBits.get(j);
                if (x_i.get(idx) && x_r.get(idx)) { 
                    v.set(idx);
                    modifications++;
                }
            }

            // If we were not able to invert M11 bits, set the remaining bits to 1 if they are equal to 1 in both x_i and x_best
            if (modifications < M11) {
                for (int j = modifications; j < Math.min((M11 - modifications), setBits.size()); j++) {
                    int idx = setBits.get(j);
                    if (x_i.get(idx) && x_best.get(idx) && !v.get(idx)) { 
                        //System.out.println("bit " + idx + " was 1 in both x_i and x_best");
                        v.set(idx);
                        modifications++;
                    }
                }
            }

            // If there are still remaining modifications, carry them out with randomly chosen bits
            // that are set to 1 in X_i.
            if (modifications < M11) {

                // Check all the bits in v that are still empty and see if the corresponding one is 1 in x_i
                List<Integer> clearBitsInNewSol = getIndicesOfClearBits(v, sensitiveTransactions.size());

                Collections.shuffle(clearBitsInNewSol);

                for (Integer idx : clearBitsInNewSol) {
                    if (x_i.get(idx)) {
                        v.set(idx);
                        modifications++;
                    }
                    if (modifications >= M11) break;
                }

            }

            // Choose M10 bits that are 0 in both x_i and v_i to set to 1 in v_i
            // First, index all the bits that are 0 in x_i
            List<Integer> clearBits = getIndicesOfClearBits(x_i, sensitiveTransactions.size());

            Collections.shuffle(clearBits);

            modifications = 0;
            int idx;
            for (int j = 0; j < clearBits.size(); j++) {
                idx = clearBits.get(j);
                if (!v.get(idx)) { 
                    v.set(idx);
                    modifications++;
                }
                if (modifications >= M10) break;
            }

            double f;

            if (fitnessCache.containsKey(v)) {
				f = fitnessCache.get(v);
			} else {
                f = calculateFitness(v, sensitiveRules, nonsensitiveRules, sensitiveTransactions);
				fitnessCache.put(v, f);
			}

            if (f < fitness[i]) {
                population.set(i, v);
                trial[i] = 0;
                fitness[i] = f;
            } else {
                trial[i]++;
            }

        }

        System.out.print("\n");
    }

    private void sendOnlookerBees(List<BitSet> population, List<Rule> sensitiveRules, List<Rule> nonsensitiveRules,
        List<Transaction> sensitiveTransactions, BitSet x_best, double theta) {

        System.out.println("Sending onlooker bees...");

        Random rand = new Random();
        BitSet foodSource;
        // Calculate probability intervals for each solution
        double[] probability = calculateProbabilities(rand);

        int M11 = 0;
        int M10 = 0;
        int M01 = 0;

        int numOneBits;
        int numZeroBits;

        for (int t = 0; t < POP_SIZE; t++) {
            System.out.print(t + " ");

            // select a food source based on its relative fitness
            double r = rand.nextDouble();
            int i = rand.nextInt(POP_SIZE);
            for (int j = 0; j < probability.length; j++) {
                if (r < probability[j]) { 
                    i = j;
                    break;
                }
            }

            foodSource = population.get(i);

            // Calculate mean dissimilarity
            // Essentially they give two different options.
            // If we do this we also need to find the diss closest to meanDiss.
            double meanDiss = 0;
            double[] diss = new double[POP_SIZE];
            int neighborIndex = -1;
            
            for (int j = 0; j < POP_SIZE; j++) {
                if (j == i) continue;
                diss[j] = dissimilarity(foodSource, population.get(j), sensitiveTransactions.size());
                meanDiss += diss[j];
            }
            meanDiss /= (POP_SIZE - 1);
            
            double minDiff = Integer.MAX_VALUE;
            
            for (int j = 0; j < POP_SIZE; j++) {
                if (j == i) continue;
                for (double d : diss) {
                    if (Math.abs(meanDiss - d) < minDiff) {
                        minDiff = Math.abs(meanDiss - d);
                        neighborIndex = j;
                    }
                }
            }

            double dissimilarity = theta * dissimilarity(foodSource, population.get(neighborIndex), sensitiveTransactions.size());
            double bestCombination = Integer.MAX_VALUE;
            double d;
            
            numOneBits = foodSource.cardinality();
            numZeroBits = sensitiveTransactions.size() - numOneBits;

            // Find the optimal values of M11, M10, M01
            // Try all possible positive integer combinations given constraints
            // m11 + m01 = numOneBits and m10 <= numZeroBits
            for (int m11 = 0; m11 <= numOneBits; m11++) {
                int m01 = numOneBits - m11;
                for (int m10 = 0; m10 <= numZeroBits; m10++) {
                    d = (m11 == 0) ? 1 : 1 - (m11 / (m11 + m01 + m10));
                    if ((d - dissimilarity) < bestCombination) {
                        bestCombination = d - dissimilarity;
                        M11 = m11;
                        M01 = m01;
                        M10 = m10;
                    }
                }
            }

            BitSet newSolution = new BitSet(sensitiveTransactions.size());
            
            // Change up to M11 dimensions to 1 using random selection.
            // Find the indices of bits that are set to 1 in the current source,
            // shuffle them and select the M11 first ones.
            List<Integer> setBits = getIndicesOfSetBits(foodSource, sensitiveTransactions.size());
            Collections.shuffle(setBits);
            for (int j = 0; j < Math.min(M11, setBits.size()); j++)
                newSolution.set(setBits.get(j));
                        
            // Change M10 dimensions to 1 in the new solution, using either random selection from the neighbor
            // or greedy selection from the current solution
            List<Integer> clearBits;
            int modifications = 0;

            if (rand.nextDouble() >= 0.5) {
                // OPTION 1: Random selection from the neighbor
                // Find all the bits that are 0 in the neighbor and set the corresponding ones in the new solution

                clearBits = getIndicesOfClearBits(population.get(neighborIndex), sensitiveTransactions.size());
                Collections.shuffle(clearBits);
                
                for (int j = 0; j < clearBits.size(); j++) {
                    if (!newSolution.get(clearBits.get(j))) {
                        newSolution.set(clearBits.get(j));
                        modifications++;
                    }
                    if (modifications >= M10) break;
                }
            } else {
                // OPTION 2: Greedy selection from current sol
                // find the bits that are 0 in the currently selected food source and 1 in x_best
                List<Integer> indices = new ArrayList<>();
                
                for (int j = 0; j < sensitiveTransactions.size(); j++) {
                    if (x_best.get(j) && !foodSource.get(j))
                        indices.add(j);
                }
                Collections.shuffle(indices);

                for (int j = 0; j < Math.min(M10, indices.size()); j++) {
                    newSolution.set(indices.get(j));
                    modifications++;
                    if (modifications >= M10) break;
                }
                
                clearBits = indices;    // for calculation below
            }

            int target = M11 + M10;

            // Get the fitness of the new solution
            double f; 
            
            if (fitnessCache.containsKey(newSolution)) {
			    f = fitnessCache.get(newSolution);
			} else {
                f = calculateFitness(newSolution, sensitiveRules, nonsensitiveRules, sensitiveTransactions);
			    fitnessCache.put(newSolution, f);
			}
            
            if (f < fitness[i]) {
                population.set(i, newSolution);
                trial[i] = 0;
                fitness[i] = f;
            } else {
                trial[i]++;
            }

        }
        System.out.print("\n");
    }

    private void sendScoutBees(List<BitSet> population, int solutionSize, double maxSanitizationRatio) {

        System.out.println("Sending scout bees...");
        // Any solutions that couldn't be improved after a certain number of trials are abandoned
        // New solutions are generated to replace them
        for (int i = 0; i < POP_SIZE; i++) {
            if (trial[i] > MAX_TRIALS) {
                System.out.println("\tAbandoning food source " + i + "...");
                population.set(i, initializeFoodSource(solutionSize, maxSanitizationRatio));
                trial[i] = 0;
            }
        }
    }

    // Returns the index of the probabilistically selected food source.
    // Probability of a food source being selected depends on its relative fitness
    private double[] calculateProbabilities(Random rand) {
        double fitnessSum = DoubleStream.of(fitness).sum();
        double cumulativeFitness = 0;
        double[] probabilityInterval = new double[fitness.length];

        for (int i = 0; i < fitness.length; i++) {
            probabilityInterval[i] = cumulativeFitness + (fitness[i] / fitnessSum);
            cumulativeFitness += (fitness[i] / fitnessSum);
        }

        return probabilityInterval;

    }

    private double dissimilarity(BitSet solA, BitSet solB, int solutionSize) {
        int M11 = 0;
        int M10 = 0;
        int M01 = 0;

        //int size = solA.length();   // error if they're not the same size
        for (int i = 0; i < solutionSize; i++) {
            if (solA.get(i) && solB.get(i)) {
                M11++;
            } else if (solA.get(i) && !solB.get(i)) {
                M10++;
            } else if (!solA.get(i) && solB.get(i)) {
                M01++;
            }
        }

        // Handles the case where both solutions are all 0 and the
        // sum used for division is 0. In this case, evidently,
        // the solutions are the same
        if (M11 == 0) return 1;

        return 1 - (M11 / (M11 + M10 + M01));
    }

    private int selectVictimItem(Rule targetRule) {
        // the conflict degree of an item is the number of rules affected by removing it
        double maxConflictDegree = -1;
        int victim = 0;

        for (Integer i : targetRule.rightSide) {
            double cdS = itemIndexSensitive.get(i).size();
            double cdN = itemIndexNonsensitive.get(i).size();
            double cdW = itemIndexWeakLeftSide.get(i).size(); 
            // affects sensitive rules only which is good, so provide a high multiplier
            if (cdN + cdW == 0) cdN = 0.001;
            double cd = cdS / (cdN + cdW);
            if (cd > maxConflictDegree) {
                maxConflictDegree = cd;
                victim = i;
            }
        }

        return victim;
    }

    private List<Integer> getIndicesOfClearBits(BitSet sol, int size) {
        List<Integer> clearBits = new ArrayList<>();
        int idx = 0;
        while (idx < size) {  // nextClearBit returns -1 when no such bit exists, aka we reach the end of the set
            idx = sol.nextClearBit(idx);
            if (idx < 0 || idx >= size) break;
            clearBits.add(idx);
            idx++;
        }
        return clearBits;
    }

    private List<Integer> getIndicesOfSetBits(BitSet sol, int size) {
        List<Integer> setBits = new ArrayList<>();
        int idx = 0;
        while (idx < size) {  // nextSetBit returns -1 when no such bit exists, aka we reach the end of the set
            idx = sol.nextSetBit(idx);
            if (idx < 0 || idx >= size) break;
            setBits.add(idx);
            idx++;
        }
        return setBits;
    }

	/**
	 * This method reads the sensitive rules into memory
	 * 
	 * @param inputSAR
	 *            the file path to a set of sensitive association rules
	 * @param rules
	 *            a structure for storing the sensitive association rules
	 * @throws IOException
	 *             if error reading the file
	 */
	private void readRulesIntoMemory(String inputSAR, List<Rule> rules)
			throws IOException {
		// open the input file
		BufferedReader reader = new BufferedReader(new FileReader(inputSAR));
		String line;
		// for each line (rule) until the end of the file
		while (((line = reader.readLine()) != null)) {
			// Each rule should have the format "4 ==> 5" in the file
			// So we split the line according to the arrow:

			String[] lineSplited = line.split("==> ");
			// left side
			String[] leftStrings = lineSplited[0].split(" ");
			// right side
			String[] rightStrings = lineSplited[1].split(" ");
			Rule rule = new Rule(); // create the rule
			// add each item from the left side after converting from string to
			// int
			for (String string : leftStrings) {
				rule.leftSide.add(Integer.parseInt(string));
			}
			// add each item from the right side after converting from string to
			// int
			for (String string : rightStrings) {
				// if the string starts with #, we stop reading the line because
				// what is after is not part of the rule
				if (string.length() > 0 && string.charAt(0) == '#') {
					break;
				}
				// Otherwise, convert the string to int and add it to the right
				// side of the rule
				rule.rightSide.add(Integer.parseInt(string));
			}
			// add the rule to the set of rules
			rules.add(rule);
		}
		// close the input file
		reader.close();
	}

	private void writeResultToFile(List<Transaction> sensitiveTransactions, String input, String output)
								   throws IOException {
		// Now, write the transformed transaction database to disk!
		BufferedReader reader = new BufferedReader(new FileReader(input));
		BufferedWriter writer = new BufferedWriter(new FileWriter(output));
		int count = 0;
        int sensitiveIndex = 0;
        int nextSensitiveId = sensitiveTransactions.get(sensitiveIndex).id;

		String line;
		
		while (((line = reader.readLine()) != null)) {

			if(line.isEmpty()) continue;

			count++;

            if (count < nextSensitiveId) {
                writer.write(line);
                writer.newLine();
                continue;
            }

            Transaction t = sensitiveTransactions.get(sensitiveIndex);

            for (Integer item : t.items)
                writer.write(item + " ");

			writer.newLine();
            sensitiveIndex++;
            if (sensitiveIndex < sensitiveTransactions.size()) {
                nextSensitiveId = sensitiveTransactions.get(sensitiveIndex).id;
            } else {
                nextSensitiveId = tidcount + 1;
            }
		}

		writer.close();
	}

}