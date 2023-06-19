package vidpso;

/** 
 *  VICTIM ITEM DELETION-BASED PARTICLE SWARM OPTIMIZATION (VIDPSO)
 *  Based on the algorithm described in the paper by Jangra & Toshniwal (2020)
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
import java.util.stream.Collectors;

/**
 * An implementation of the VIDPSO algorithm for hiding sensitive association
 * rules (or more specifically frequent itemsets) in a transaction database. 
 * 
 * Some code is based on or borrowed from SPMF library by:
 *
 * @author Philippe Fournier-Viger
 * @author Hoang Thi Dieu (2019-improvement)
 * 
 * @author Charlie Mårtensson (modifications)
 */
public class VIDPSO {

	private class Particle {
		List<Set<Transaction>> velocity;
		List<Set<Transaction>> position;

		Particle() {
			velocity = new ArrayList<>();
			position = new ArrayList<>();
		}  

        public void print() {
            for (Set<Transaction> subparticle : position) {
                System.out.print("{");
                for (Transaction t : subparticle) System.out.print(t.tid + "; ");
                System.out.print("}, ");
            }
            System.out.print("\n");
        }
	}

    private class Itemset {
        int antecedentSuppNonsensitive;
        int supportSensitive;
        int supportNonsensitive; // will be 0 for sensitive itemsets
        int numRules;
        Set<Integer> items;

        Itemset() {
            this.antecedentSuppNonsensitive = 0;
            this.supportSensitive = 0;
            this.supportNonsensitive = 0;
            this.items = new HashSet<>();
            this.numRules = 0;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Itemset)) {
                return false;
            }
            Itemset i = (Itemset) o ;
            return this.items.equals(i.items);
        }

        @Override
        public int hashCode() {
            return this.items.hashCode();
        }
    }

	private class Transaction {
		int tid;	// the row number in the transaction database
		Set<Integer> items;

		Transaction(int tid, Set<Integer> items) {
			this.tid = tid;
			this.items = items;
		}

        Transaction(Transaction t) {
            this.tid = t.tid;
            this.items = new HashSet<Integer>(t.items);
        }

        public void print() { 
            System.out.print("#" + tid + ": ");
            for (Integer item : items)
                System.out.print(item + " ");
            System.out.println();
        }
	}

	// variables for statistics
	int tidcount = 0; // the number of transactions in the last database read
	long startTimestamp = 0; // the start time of the last execution
	long endTimeStamp = 0; // the end time of the last execution

	final int POP_SIZE = 20;
	final int MAX_GENS = 15;    //CHANGE
	final int ATTEMPT_LIMIT = 7;
    final int N = 1;

    int numSensitiveRules;
    int numNonsensitiveRules;

    double minsup;

    Map<List<Set<Transaction>>, Double> fitnessCache;
    Map<Itemset, Integer> sensitiveRuleCountsPerItemset;
    Map<Itemset, Integer> nonsensitiveRuleCountsPerItemset;

	/**
	 * Run the VIDPSO algorithm
	 * 
	 * @param input
	 *            the file path to a transaction database
	 * @param inputSAR
	 *            the file path to a set of sensitive association rules to be
	 *            hidden
	 * @param inputRules
	 * 			  the file path to the set of all mined association rules
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
	public void runAlgorithm(String input, String inputSAR, String inputRules, String output,
			double minsup, double minconf) throws IOException {
		// record the start time
		startTimestamp = System.currentTimeMillis();

		/* STEP 1: PREPROCESSING */
		List<Rule> sensitiveRules = new ArrayList<>();
		List<Rule> allRules = new ArrayList<>();

        Set<Itemset> sensitiveItemsets = new HashSet<>();
        Set<Itemset> nonsensitiveItemsets = new HashSet<>();

        sensitiveRuleCountsPerItemset = new HashMap<>();
        nonsensitiveRuleCountsPerItemset = new HashMap<>();

        List<Transaction> sensitiveTransactions = new ArrayList<>();

        fitnessCache = new HashMap<>();

        this.minsup = minsup;

        System.out.println("Reading rules into memory...");
		readRulesIntoMemory(inputSAR, sensitiveRules);
		readRulesIntoMemory(inputRules, allRules);

        numSensitiveRules = sensitiveRules.size();
        numNonsensitiveRules = allRules.size() - numSensitiveRules;

        // Add any itemsets that generate a sensitive rule to the set of sensitive frequent itemsets.
        // Add the remaining itemsets to the set of nonsensitive frequent itemsets.
        System.out.println("Generating itemsets...");
        Set<Itemset> allGenerating = new HashSet<>();
        for (Rule r : allRules) {
            Itemset itemset = new Itemset();
            itemset.items.addAll(r.leftSide);
            itemset.items.addAll(r.rightSide);

            //allGenerating.add(itemset);

            if (!allGenerating.contains(itemset)) {
                allGenerating.add(itemset);
                nonsensitiveRuleCountsPerItemset.put(itemset, 1);
            } else {
                int val = nonsensitiveRuleCountsPerItemset.get(itemset);
                nonsensitiveRuleCountsPerItemset.put(itemset, val + 1);
            }
        }

        for (Itemset itemset : allGenerating) {
            Set<Integer> r = itemset.items;
            for (Rule s : sensitiveRules) {
                if (r.containsAll(s.leftSide) && r.containsAll(s.rightSide)
                    && r.size() == (s.leftSide.size() + s.rightSide.size())) {
                    
                    if (!sensitiveItemsets.contains(itemset)) {
                        sensitiveItemsets.add(itemset);
                        sensitiveRuleCountsPerItemset.put(itemset, 1);
                    } else {
                        int val = sensitiveRuleCountsPerItemset.get(itemset);
                        sensitiveRuleCountsPerItemset.put(itemset, val + 1);
                    }
                    
                } 
            }
            
            if (!sensitiveItemsets.contains(itemset)) {
                nonsensitiveItemsets.add(itemset);
            } else {
                int val = nonsensitiveRuleCountsPerItemset.get(itemset);
                int diff = sensitiveRuleCountsPerItemset.get(itemset);
                nonsensitiveRuleCountsPerItemset.put(itemset, val - diff);
            }
        }

        // Identify the sensitive transactions
        System.out.println("Identifying sensitive transactions...");
        String line;
		BufferedReader reader = new BufferedReader(new FileReader(input));

		while (((line = reader.readLine()) != null)) {

			if(line.isEmpty()) continue;

			tidcount++;

			String[] transactionItems = line.split(" ");

			Set<Integer> transaction = new HashSet<>(transactionItems.length);

            // for each item in the current transaction
			for (int i = 0; i < transactionItems.length; i++) {
				// convert from string to int
				int item = Integer.parseInt(transactionItems[i]);
				// add it to the transaction
				transaction.add(item);
			}

            // a transaction is critical if it contains any sensitive itemset
            boolean added = false; 
            // We need to go through all the rules to obtain support counts
            // but we only want to add each rule once.
            for (Itemset s : sensitiveItemsets) {
                if (transaction.containsAll(s.items)) {
                    s.supportSensitive++;    // we will reset this value each time we calculate fitness, but we need it for subparticle size
                    if (!added) {
                        sensitiveTransactions.add(new Transaction(tidcount, transaction));
                        added = true;
                    }
                }
            }

            for (Itemset ns : nonsensitiveItemsets) {
                // For nonsensitive items, we want a support count from the nonsensitive transactions.
                // Their support in sensitive ones will be recalculated in the fitness function later,
                // but we need to add this value to get the total support.
                if (transaction.containsAll(ns.items) && !added)   
                    ns.supportNonsensitive++;
            }
        } // finish reading transactions
        System.out.println("Sensitive transactions: " + sensitiveTransactions.size());
        double percentage = (double)sensitiveTransactions.size() / tidcount;
        System.out.println("Percentage of transactions that are sensitive: " + (percentage * 100) + "%");

        System.out.println("Selecting victim items...");
        List<Integer> victimItems = selectVictimItems(sensitiveItemsets, nonsensitiveItemsets);
        Map<Integer, List<Transaction>> victimItemTransactions = new HashMap<>();

        for (Integer v : victimItems) {
            victimItemTransactions.put(v, new ArrayList<>());
            for (Transaction t : sensitiveTransactions) {
                if (t.items.contains(v)) 
                    victimItemTransactions.get(v).add(t);
            }
        }

        // Initialize population
        System.out.println("Initializing population...");
        List<Particle> population = new ArrayList<>();

        //Get subparticle sizes
        int[] sizes = new int[victimItems.size()];
        for (int i = 0; i < victimItems.size(); i++) {
            int maxSupport = -1;
            for (Itemset s : sensitiveItemsets) {
                if (s.items.contains(victimItems.get(i)) 
                        && s.supportSensitive > maxSupport)
                    maxSupport = s.supportSensitive;
            }
            sizes[i] = N * (int) (Math.floor(maxSupport - minsup * tidcount) + 1);
        }

        for (int i = 0; i < POP_SIZE; i++) {
            Particle p = new Particle();

            for (int j = 0; j < victimItems.size(); j++) {
                Set<Transaction> subparticle = new HashSet<>();

                Collections.shuffle(victimItemTransactions.get(victimItems.get(j)));
                for (int k = 0; k < sizes[j]; k++)
                    subparticle.add(victimItemTransactions.get(victimItems.get(j)).get(k));
                p.position.add(subparticle);
                p.velocity.add(new HashSet<Transaction>()); // just to initialize this list position, so it can be overwritten later.
            }
            population.add(p);

        }

        Particle gbest = population.get(0); // set to the first particle to start, just to ensure initialization.
        Particle[] pbest = new Particle[POP_SIZE];
        double fitness;
        double gbestFit = 999;
        double[] pbestFit = new double[POP_SIZE];
        Arrays.fill(pbestFit, 999);
        
        int generations = 0;
        Random rand = new Random();

        for (int i = 0; i < POP_SIZE; i++) {
            Particle p = population.get(i);

            if (fitnessCache.containsKey(p.position)) {
				fitness = fitnessCache.get(p.position);
			} else {
                fitness = calculateFitness(p, sensitiveTransactions, victimItems, sensitiveItemsets, nonsensitiveItemsets, minsup, minconf);
				fitnessCache.put(p.position, fitness);
			}

            pbest[i] = p;
            pbestFit[i] = fitness;
            if (pbestFit[i] < gbestFit) {
                gbest = pbest[i];
                gbestFit = pbestFit[i];
            }
        }

        int attempts = 0;
        /* STEP 2: MAIN ALGORITHM */
        while (generations < MAX_GENS && attempts < ATTEMPT_LIMIT) {

            System.out.println("Generation: " + generations);
            boolean globalImproved = false;

            // Create the exploring-transactions set for each victim item
            List<List<Transaction>> exploringTransactions = new ArrayList<>();

            for (int j = 0; j < victimItems.size(); j++) {

                List<Transaction> et = new ArrayList<>();

                for(int i = 0; i < POP_SIZE; i++) {
                    //et.addAll(population.get(i).position.get(j));   // p_ij
                    for (Transaction t : population.get(i).position.get(j))
                        if (!et.contains(t)) et.add(t);
                }

                Integer victim = victimItems.get(j);
                List<Transaction> temp = new ArrayList<>(victimItemTransactions.get(victim));

                temp.removeAll(et);
                exploringTransactions.add(temp);
            }

            // Update position and velocity of subparticle p_ij
            for (int i = 0; i < POP_SIZE; i++) {
                Particle p = population.get(i);

                for (int j = 0; j < p.position.size(); j++) {
                    int m = p.position.get(j).size();   // size of the subparticle p_ij

                    List<Transaction> diff1 = new ArrayList<>(pbest[i].position.get(j));
                    
                    diff1.removeAll(p.position.get(j)); // set pbest_ij - p_ij
                    
                    List<Transaction> diff2 = new ArrayList<>(gbest.position.get(j));
                    diff2.removeAll(p.position.get(j)); // set gbest_j - p_ij

                    List<Transaction> et = exploringTransactions.get(j);
                    List<Transaction> prev = new ArrayList<>(p.position.get(j));

                    int n1 = (int) Math.min(Math.floor(0.3 * m), diff1.size());    // local search
                    int n2 = (int) Math.min(Math.floor(0.3 * m), diff2.size());    // global search
                    int n3 = (int) Math.min(Math.floor(0.3 * m), et.size());       // exploration capability
                    // testing out something.
                    // if all values are zero we get no randomness at all
                    // may have to introduce something in the case where the subparticles are Small.

                    int n4 = m - n1 - n2 - n3;                                     // inertia

                    Collections.shuffle(diff1);
                    Collections.shuffle(diff2);
                    Collections.shuffle(et);
                    Collections.shuffle(prev);

                    // select n1 random items from diff1 and n2 random items from diff2 and n3 random items from ET
                    // v_ij(t + 1) = n_1 * rand(pbest_ij - p_ij) U n_2 * rand(gbest_j - p_ij) U n_3 * rand(ET)
                    Set<Transaction> v_ij = new HashSet<>();    // might not be necessary to save the velocity but let's do it for now.
                    v_ij.addAll(diff1.subList(0, n1));
                    v_ij.addAll(diff2.subList(0, n2));
                    v_ij.addAll(et.subList(0, n3));

                    // p_ij(t + 1) = v_ij(t + 1) U n_4 * rand(p_ij(t))
                    Set<Transaction> p_ij = new HashSet<>(v_ij);
                    p_ij.addAll(prev.subList(0, n4));

                    if (v_ij.isEmpty())
                        System.out.println("\tvelocity of (" + i + "," +j+") is empty, position won't change.");

                    p.position.set(j, p_ij);
                    p.velocity.set(j, v_ij);

                }

                if (fitnessCache.containsKey(p.position)) {
				    fitness = fitnessCache.get(p.position);
			    } else {
                    fitness = calculateFitness(p, sensitiveTransactions, victimItems, sensitiveItemsets, nonsensitiveItemsets, minsup, minconf);
				    fitnessCache.put(p.position, fitness);
			    }

                if (fitness < pbestFit[i]) {
                    pbest[i] = p;
                    pbestFit[i] = fitness;

                    if(pbestFit[i] < gbestFit) {
                        gbest = p;
                        gbestFit = fitness;
                        globalImproved = true;
                    }
                }
            }

            if (!globalImproved) { 
                attempts++;
                if (attempts >= ATTEMPT_LIMIT) System.out.println("Attempt limit reached.");
            }

            generations++;
        }

		// save the end time. don't bother with time required to write to file.
		endTimeStamp = System.currentTimeMillis();

        // Sanitize the database
		writeResultToFile(gbest, victimItems, input, output);

		System.out.println("Best fitness: " + gbestFit);
		System.out.print("\n");
		System.out.println("Time: " + (endTimeStamp - startTimestamp));
	}

    private List<Integer> selectVictimItems(Set<Itemset> sensitive,
                            Set<Itemset> nonsensitiveItemsets) {
        Set<Itemset> sensitiveItemsets = new HashSet<>(sensitive);
        List<Integer> victimItems = new ArrayList<>();   // we use a list because the victim items must be ordered

        while (!sensitiveItemsets.isEmpty()) {
            Map<Integer, Integer> itemOccurrences = new HashMap<>();
            List<Integer> candidateVictims = new ArrayList<>();
            for (Itemset s : sensitiveItemsets) {
                for (Integer i : s.items) {
                    // add the item to the map if not yet present; otherwise increment the count
                    itemOccurrences.merge(i, 1, Integer::sum);
                }
            }

            // find the item with the highest occurrence count
            int max = Collections.max(itemOccurrences.values());
            // make a list containing the items with that occurrence count
            candidateVictims = itemOccurrences.entrySet().stream()
                .filter(entry -> entry.getValue() == max)
                .map(entry -> entry.getKey())
                .collect(Collectors.toList());


            // if there are several items with the same number of occurrences
            // in sensitive itemsets, break the tie by finding the one that
            // occurs the least in nonsensitive itemsets.
            Integer vMin = 99999;  // this will be replaced because there is guaranteed to be an item with a smaller support than maxint
            if (candidateVictims.size() > 1) {
                int minCount = Integer.MAX_VALUE;
                for (Integer i : candidateVictims) {
                    int count = 0;
                    for (Itemset ns : nonsensitiveItemsets) {
                        if (ns.items.contains(i)) count++;
                    }
                    if (count < minCount) {
                        vMin = i;
                        minCount = count;
                    }
                }
            } else {
                // single candidate victim item
                vMin = candidateVictims.get(0);
            }

            if (!victimItems.contains(vMin)) victimItems.add(vMin);

            Iterator<Itemset> itr = sensitiveItemsets.iterator();
            while (itr.hasNext()) {
                Itemset s = itr.next();
                if (s.items.contains(vMin)) {
                    itr.remove();
                }
            }
        }

        return victimItems;
    }

	// we want this function to be able to access the matrix of critical transactions and the sensitive items list
	private double calculateFitness(Particle p, List<Transaction> sensitiveTransactions, 
                                    List<Integer> victimItems, Set<Itemset> sensitiveItemsets,
                                    Set<Itemset> nonsensitiveItemsets, double minsup, double minconf) {

		double w1 = 0.7;
		double w2 = 0.3;

        double hf = 0.0;
        double mc = 0.0;

        // Create the candidate solution
        List<Transaction> candidateSolution = new ArrayList<>();
        for (Transaction t : sensitiveTransactions)
            candidateSolution.add(new Transaction(t));

        // Sanitize the candidate solution
        for (int i = 0; i < victimItems.size(); i++) {
            Integer victim = victimItems.get(i);
            // which transactions does the particle tell us to remove the current victim from?
            // this is specified in the position's i:th subparticle.
            for (Transaction t1 : p.position.get(i)) {
                // get the corresponding id. this seems super inefficient ... no direct lookup.
                for (Transaction t2 : candidateSolution) {
                    if (t1.tid == t2.tid) {
                        t2.items.remove(victim);
                    }
                }
            }
        }

        System.out.println("candidate solution is: ");
        for (Transaction t : candidateSolution)
            System.out.println(t.tid + ": " + t.items);

        for (Itemset s : sensitiveItemsets) {
            s.supportSensitive = 0;
            for (Transaction t : candidateSolution) {
                if (t.items.containsAll(s.items))
                    s.supportSensitive++;
            }
            if (s.supportSensitive >= minsup * tidcount) {
                hf += sensitiveRuleCountsPerItemset.get(s);  // all rules generated by this itemset are failed to be hidden
            }
        }

        for (Itemset ns : nonsensitiveItemsets) {
            ns.supportSensitive = 0;
            for (Transaction t : candidateSolution) {
                if (t.items.containsAll(ns.items))
                    ns.supportSensitive++;
            }
            if ((ns.supportSensitive + ns.supportNonsensitive) < (minsup * tidcount)) {
                System.out.println("itemset " + ns.items + " supports " + nonsensitiveRuleCountsPerItemset.get(ns) + " ns rules and was lost with support " + (ns.supportSensitive + ns.supportNonsensitive));
                mc += nonsensitiveRuleCountsPerItemset.get(ns);
            }
        }

        hf /= numSensitiveRules;
        mc /= numNonsensitiveRules;

		return w1 * hf + w2 * mc;
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

	private void writeResultToFile(Particle solution, List<Integer> victimItems, String input, String output)
								   throws IOException {
		// Now, write the transformed transaction database to disk!
		BufferedReader reader = new BufferedReader(new FileReader(input));
		BufferedWriter writer = new BufferedWriter(new FileWriter(output));
		int count = 0;

		String line;
		
		while (((line = reader.readLine()) != null)) {

			if(line.isEmpty()) continue;

			count++;

            String[] transactionItems = line.split(" ");

			Set<Integer> transaction = new HashSet<>(transactionItems.length);

            // for each item in the current transaction
			for (int i = 0; i < transactionItems.length; i++) {
				// convert from string to int
				int item = Integer.parseInt(transactionItems[i]);
				// add it to the transaction
				transaction.add(item);
			}

            // Sanitize transactions that are present in the solution
            for (int i = 0; i < victimItems.size(); i++) {
                Integer victim = victimItems.get(i);
                // which transactions does the particle tell us to remove the current victim from?
                for (Transaction t : solution.position.get(i)) {
                    // get the corresponding id. this seems super inefficient ... no direct lookup.
                    if (t.tid == count) {
                        //System.out.println("Removing " + victim + " from transaction " + count + ". ");
                        transaction.remove(victim);
                    }
                }
            }

            for (Integer item : transaction)
                writer.write(item + " ");

			writer.newLine();
		}

		writer.close();
	}

}