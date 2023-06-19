package samdp;

/** 
 *  OPTIMIZED SANITIZATION APPROACH FOR MINABLE DATA PUBLICATION (SA-MDP)
 *  Based on the algorithm described in the paper by Yang & Liao (2022)
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
import java.util.stream.Collectors;

/**
 * An implementation of the SA-MDP algorithm for hiding sensitive association
 * rules in a transaction database. Some code is borrowed from the SPMF library
 * by: 
 * 
 * @author Philippe Fournier-Viger
 * @author Hoang Thi Dieu (2019-improvement)
 * 
 * @author Charlie Mårtensson
 */
public class AlgoSAMDP {

	private class Particle {
		int numChildren;
		Set<Integer> velocity;
		Set<Integer> position;

		Particle() {
			numChildren = 1;
			velocity = new HashSet<Integer>();
			position = new HashSet<Integer>();
		}  
	}

	private class Transaction {
		int tid;	// the row number in the transaction database
		int criticalIndex;	// the index number in the list of critical transactions
		Set<Integer> items;

		Transaction(int tid, int criticalIndex, Set<Integer> items) {
			this.tid = tid;
			this.criticalIndex = criticalIndex;
			this.items = items;
		}
	}

	// variables for statistics
	int tidcount = 0; // the number of transactions in the last database read
	long startTimestamp = 0; // the start time of the last execution
	long endTimeStamp = 0; // the end time of the last execution

	final int POP_SIZE = 20;
	final int MAX_GENS = 15;
	final int ATTEMPT_LIMIT = 7;

	Map<Set<Integer>, Double> fitnessCache;

	double minsup;
	double minconf;

	/**
	 * Run the SA-MDP algorithm
	 * 
	 * @param input
	 *            the file path to a transaction database
	 * @param inputSAR
	 *            the file path to a set of sensitive association rules to be
	 *            hidden
	 * @param inputNAR
	 * 			  the file path to the set of nonsensitive association rules
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
	public void runAlgorithm(String input, String inputSAR, String inputNAR, String output,
			double minsup, double minconf, int sMax, int sMin) throws IOException {
		// record the start time
		startTimestamp = System.currentTimeMillis();

		/* STEP 1: Preprocessing */
		this.minsup = minsup;
		this.minconf = minconf;

		List<Rule> sensitiveRules = new ArrayList<>();
		List<Rule> nonSensitiveRules = new ArrayList<>();
		List<Rule> criticalRules = new ArrayList<>();

		fitnessCache = new HashMap<>();

		System.out.println("Reading rules into memory...");

		readRulesIntoMemory(inputSAR, sensitiveRules);
		readRulesIntoMemory(inputNAR, nonSensitiveRules);

		// the transactions from the database
		List<Transaction> criticalTransactions = new ArrayList<>();

		// map of each distinct item in the db to a sequential index
		Map<Integer, Integer> distinctItems = new HashMap<>();
		int distinctItemCtr = 0;
		int criticalCount = 0;

		System.out.println("Generating sensitive items...");
		// the set S_I considered worthy of protection
		Set<Integer> sensitiveItems = generateSensitiveItems(sensitiveRules, nonSensitiveRules);
		System.out.println(sensitiveItems);

		System.out.println("Generating critical rules...");
		// Generate the critical rules (nars containing sensitive items)
		for (Rule r : nonSensitiveRules) {
			for (Integer s : sensitiveItems) {
				if (r.leftSide.contains(s) || r.rightSide.contains(s)) {
					criticalRules.add(r);
					break;	// only add it once
				}
			}
		}

		// Delete the critical rules from the list of regular NARs so they can be treated separately
		for (Rule r : criticalRules)
			nonSensitiveRules.remove(r);

		String line;
		BufferedReader reader = new BufferedReader(new FileReader(input));

		System.out.println("Identifying critical transactions...");
		// Identify the critical transactions and calculate rule support in noncritical transactions
		while (((line = reader.readLine()) != null)) {

			if(line.isEmpty()) continue;

			tidcount++;

			String[] transactionItems = line.split(" ");

			Set<Integer> transaction = new HashSet<>(transactionItems.length);

			boolean isCritical = false;

			// for each item in the current transaction
			for (int i = 0; i < transactionItems.length; i++) {
				// convert from string to int
				int item = Integer.parseInt(transactionItems[i]);
				// add it to the transaction
				transaction.add(item);
				// check if this item has been seen previously in the db; if not, map it to a sequential index.
				if(!distinctItems.containsKey(item)) {
					distinctItems.put(item, distinctItemCtr);
					distinctItemCtr++;
				}
			}

			/* 
			   Go through every SAR and see if this transaction supports any of them.
			   If yes, the transaction is critical.
			*/
			for (Rule rule : sensitiveRules) {

				Set<Integer> matchLeft = new HashSet<>();
				Set<Integer> matchRight = new HashSet<>();

				for (Integer i : transaction) {

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
						rule.supportCritical++;
						if(!isCritical) {
							isCritical = true;
							Transaction t = new Transaction(tidcount, criticalCount, transaction);
							criticalTransactions.add(t);
							criticalCount++;
							// save the index for writing back to file.
						}
						break;
					}
				}
			}

			if (!isCritical) {
				// If not critical, we still need to compute the antecedent support for SARs
				for (Rule rule : sensitiveRules) {

					Set<Integer> matchLeft = new HashSet<>();
					for (Integer i : transaction) {
						// if the left side of this sensitive rule matches with this
						// transaction
						if (matchLeft.size() != rule.leftSide.size()
								&& rule.leftSide.contains(i)) {
							matchLeft.add(i);
							if (matchLeft.size() == rule.leftSide.size()) {
								rule.leftSideCountNonCritical++;
							}
						}
					}
				}

				// If not critical, we also need to determine support for critical NARs.
				// If the transaction IS critical, the support of critical NARs is computed in the
				// fitness function.
				// They contain sensitive items and may therefore be affected, so it can't be
				// precomputed.
				for (Rule rule : criticalRules) {
					Set<Integer> matchLeft = new HashSet<>();
					Set<Integer> matchRight = new HashSet<>();

					for (Integer i : transaction) {

						// if the left side of this nonsensitive rule matches with this
						// transaction
						if (matchLeft.size() != rule.leftSide.size()
								&& rule.leftSide.contains(i)) {
							matchLeft.add(i);
							if (matchLeft.size() == rule.leftSide.size()) {
								rule.leftSideCountNonCritical++;
							}
						}   // else if the item appears in the right side of this
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
							rule.supportNonCritical++;
							break;
						}
					}
				}
			}

			/** 
			 * Go through every noncritical NAR and see if this transaction supports any of them.
			 */
			for (Rule rule : nonSensitiveRules) {

				Set<Integer> matchLeft = new HashSet<>();
				Set<Integer> matchRight = new HashSet<>();

				for (Integer i : transaction) {

					// if the left side of this nonsensitive rule matches with this
					// transaction
					if (matchLeft.size() != rule.leftSide.size()
							&& rule.leftSide.contains(i)) {
						matchLeft.add(i);
						if (matchLeft.size() == rule.leftSide.size()) {
							if (isCritical) {
								rule.leftSideCountCritical++;
							} else {
								rule.leftSideCountNonCritical++;
							}
						}
					}   // else if the item appears in the right side of this
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
						if (isCritical) {
							rule.supportCritical++;
						} else {
							rule.supportNonCritical++;
						}
						break;
					}
				}
			}


		}	// Finish reading all transactions.
		double percentage = (double) criticalTransactions.size() / tidcount;
		System.out.println("Percentage of transactions that are critical: " + (percentage*100) + "%");

		// Generate the matrix of bit vectors corresponding to critical transactions
		BitSet[] critical = new BitSet[criticalTransactions.size()];

		// Convert each critical transaction into a bit vector. 
		// Bits corresponding to items present in the transaction are set to 1
		int n = distinctItems.size();

		for (int i = 0; i < criticalTransactions.size(); i++) {
			// the length of each bit vector corresponds to the number of distinct items.
			critical[i] = new BitSet(n);
			for (Integer item : distinctItems.keySet()) {
				if (criticalTransactions.get(i).items.contains(item)) {
					int index = distinctItems.get(item);
					critical[i].set(index);
				}
			}
		}

		// Calculate upper and lower bounds for modifiable transactions in child particles
		int maxSup = -1;
		double maxConf = -1;
		int auxSup = 0;
		for (Rule r : sensitiveRules) {
			if (r.supportCritical > maxSup)
				maxSup = r.supportCritical;
			
			double conf = (double) r.supportCritical / (r.supportCritical + r.leftSideCountNonCritical);
			if (conf > maxConf) {
				maxConf = conf;
				auxSup = r.supportCritical;
			}
		}
		
		int d1 = (int) Math.ceil(maxSup - minsup * tidcount);
		int d2 = (int) Math.ceil((maxConf - minconf) * auxSup * tidcount);

		int mMax = criticalTransactions.size();
		int mMin = Math.min(d1, d2);


		/* STEP 2: Initialize the population */
		System.out.println("Initializing population...");
		List<Particle> population = new ArrayList<Particle>();

		Set<Integer> gbest;
		List<Set<Integer>> pbest = new ArrayList<>(POP_SIZE);
		double gbestFit = 999;
		double[] pbestFit = new double[POP_SIZE];
		double fitness;
		
		Random rand = new Random();
		
		// Sequential list of integers in the range [0, bound)
		// that can be shuffled to retrieve the first k items.
		// This lets us pick k critical transactions.
		List<Integer> dimensions = IntStream.range(0, criticalTransactions.size())
									.boxed()
									.collect(Collectors.toList());
		
		System.out.println("Calculating initial fitness...");
		for (int i = 0; i < POP_SIZE; i++) {

			System.out.print(i + " ");

			Set<Integer> p = new HashSet<Integer>();
			
			// how many dimensions (= victim transactions) to add to this particle
			int numDims = rand.nextInt(dimensions.size());

			// Randomly select which transactions to include by choosing the first numDims
			// items from the shuffled list of indices.
			Collections.shuffle(dimensions);
			for (int j = 0; j < numDims; j++)
				p.add(dimensions.get(j));
			
			Particle particle = new Particle();
			particle.position = p;
			population.add(particle);
			
			// Find the fittest particle in the initial population
			if (fitnessCache.containsKey(p)) {
				fitness = fitnessCache.get(p);
			} else {
				fitness = calculateFitness(p, critical, sensitiveItems, distinctItems, sensitiveRules, nonSensitiveRules, criticalRules);
				fitnessCache.put(p, fitness);
			}

			pbest.add(i, p);
			pbestFit[i] = fitness;
		}
		System.out.print("\n");

		gbest = pbest.get(0);
		gbestFit = pbestFit[0];
		for (int i = 0; i < POP_SIZE; i++) {
			if (pbestFit[i] < gbestFit) {
				gbest = pbest.get(i);
				gbestFit = pbestFit[i];
			}
		}


		/* STEP 3: MAIN ALGORITHM */
		System.out.println("Starting main SA-MDP algorithm...");
		int generations = 0;
		int improvementAttempts = 0;

		while(generations < MAX_GENS && improvementAttempts < ATTEMPT_LIMIT) {

			System.out.println("GENERATION: " + generations);
			
			int totalNumChildren = 0;	// for the computation of m, which is a bit unclear.
						
			// Compute number of child particles for each particle in the population
			for (Particle particle : population) {
				int s = (int) Math.floor((sMax - sMin) * rand.nextFloat()) + sMin;
				particle.numChildren = s;
				totalNumChildren += s;
			}

			// Update velocity and position based on gbest
			for (int p = 0; p < POP_SIZE; p++) {
				System.out.print(p + " ");
				Particle particle = population.get(p);

				double r = rand.nextDouble();
				int dimsToChange = r < 0.5 ? (int) (Math.floor(r) * particle.position.size()) 
										   : (int) (Math.ceil(r) * particle.position.size());

				/* 
				   Update the velocity and position by below equations
        		   v^{n+1, i} <-- [p_best - p^n_i] U [g_best - p^n_i]
                   p^{n+1, i} <-- {p^n_i, null} * random number U v^{n+1, i}
				*/

				// velocity
				HashSet<Integer> v = new HashSet<>(pbest.get(p));
				HashSet<Integer> temp = new HashSet<>(gbest);
				v.removeAll(particle.position);
				temp.removeAll(particle.position);
				v.addAll(temp);

				// position
				HashSet<Integer> pos = new HashSet<>(v);
				ArrayList<Integer> l = new ArrayList<>(particle.position);
				Collections.shuffle(l);
				for (int i = 0; i < dimsToChange; i++)
					pos.add(l.get(i));
				
				particle.velocity = v;
				particle.position = pos;

				// Find distance of child particle
				int m = (int) Math.floor((mMax - mMin) * rand.nextDouble());

				// Generate s - 1 child particles. The first particle is the one we just updated.
				double parentFitness;
				if (fitnessCache.containsKey(particle.position)) {
					parentFitness = fitnessCache.get(particle.position);
				} else {
					parentFitness = calculateFitness(particle.position, critical, sensitiveItems, distinctItems, sensitiveRules, nonSensitiveRules, criticalRules);
					fitnessCache.put(particle.position, parentFitness);
				}

				Set<Integer> fittestChild = particle.position;
				for (int i = 1; i < particle.numChildren; i++) {
					Set<Integer> child = new HashSet<>(particle.position);

					// Randomly mutate the child
					Collections.shuffle(dimensions);
					for (int j = 0; j < m; j++) {
						int dim = dimensions.get(j);
						// mMin denotes the minimum number of transactions that can be removed,
						// so we add a check to ensure we do not remove too many dimensions 
						// during the mutation process.
						if (child.contains(dim) && (child.size() - 1) >= mMin) {
							child.remove(dim);
						} else {
							child.add(dim);
						}
					}

					// If any of the children are fitter than the parent, replace the parent with the child.
					double childFitness;
					if (fitnessCache.containsKey(child)) {
						childFitness = fitnessCache.get(child);
					} else {
						childFitness = calculateFitness(child, critical, sensitiveItems, distinctItems, sensitiveRules, nonSensitiveRules, criticalRules);
						fitnessCache.put(child, childFitness);
					}
					
					if (childFitness < parentFitness) {
						parentFitness = childFitness;
						fittestChild = child;
					}
				}

				particle.position = fittestChild;

				if (fitnessCache.containsKey(particle.position)) {
					fitness = fitnessCache.get(particle.position);
				} else {
					fitness = calculateFitness(particle.position, critical, sensitiveItems, distinctItems, sensitiveRules, nonSensitiveRules, criticalRules);
					fitnessCache.put(particle.position, fitness);
				}

				if (fitness < pbestFit[p]) {
					pbest.set(p, particle.position);
					pbestFit[p] = fitness;
				}
			}
			System.out.print("\n");

			// Determine the best fitness throughout the population
			double bestFitness = pbestFit[0];
			Set<Integer> bestSol = pbest.get(0);
			for (int i = 1; i < POP_SIZE; i++) {
				if (pbestFit[i] < bestFitness) {
					bestFitness = pbestFit[i];
					bestSol = pbest.get(i);
				}
			}

			if (bestFitness < gbestFit) {
				gbest = bestSol;
				gbestFit = bestFitness;
			} else {
				improvementAttempts++;
				if (improvementAttempts >= ATTEMPT_LIMIT) System.out.println("Attempt limit reached.");
			}

			generations++;
		}

		// save the end time. don't bother with time required to write to file.
		endTimeStamp = System.currentTimeMillis();

		writeResultToFile(gbest, criticalTransactions, sensitiveItems, input, output);

		System.out.println("M_max: " + mMax + ", M_min: "+mMin);
		System.out.println("Best fitness: " + gbestFit);
		System.out.println("Size of solution: " + gbest.size());
		System.out.println("Time: " + (endTimeStamp - startTimestamp));
	}

	// we want this function to be able to access the matrix of critical transactions and the sensitive items list
	private double calculateFitness(Set<Integer> p, BitSet[] original, Set<Integer> sensitiveItems, Map<Integer, Integer> distinctItems,
									List<Rule> sars, List<Rule> nars, List<Rule> critNars) {

		double w1 = 0.5;
		double w2 = 0.45;
		double w3 = 0.05;

		// Create the candidate solution by copying the original
		BitSet[] candidateSolution = new BitSet[original.length];
		for (int i = 0; i < original.length; i++) {
			candidateSolution[i] = new BitSet(distinctItems.size());
			candidateSolution[i].or(original[i]);
		}

		// Delete sensitive items from the specified rows in the candidate solution
		for (Integer tid : p) {
			for(Integer item : sensitiveItems) {
				int index = distinctItems.get(item);	// access the index mapping for this item
				if(candidateSolution[tid].get(index)) {
					candidateSolution[tid].clear(index);
				}
			}
		}

		double hfr = 0;
		double lrr = 0;

		// Go through all sensitive rules and determine their support and confidence in the sanitized solution.
		// If sup or conf exceeds the minimum threshold the rule has failed to be hidden.
		for (Rule r : sars) {
			int support = r.supportNonCritical;	// will be 0 for sars
			int lhsCount = r.leftSideCountNonCritical;
			int index;

			for (BitSet b : candidateSolution) {
				boolean containsLeft = true;
				boolean containsRight = true;

				for (Integer lItem : r.leftSide) {
					index = distinctItems.get(lItem);

					if (!b.get(index)) {
						containsLeft = false;
						break;
					}
				}

				if (containsLeft) {
					lhsCount++;
					for (Integer rItem : r.rightSide) {
						index = distinctItems.get(rItem);

						if(!b.get(index)) {
							containsRight = false;
							break;
						}
					}
					if (containsRight) support++;
				}
			}

			r.sup = (double) support / tidcount;
			r.conf = lhsCount == 0 ? 0 : (double) support / lhsCount;
			
			if (r.sup >= minsup && r.conf >= minconf) hfr++;

		}

		// Go through all nonsensitive critical rules (those containing sensitive items)
		// and determine new support and confidence.
		// Since only sensitive items are removed, a NAR not containing any (noncritical rule)
		// shouldn't be affected.
		// If the support or confidence are below the minimum threshold, the rule has been lost.
		for (Rule r : critNars) {
			
			int support = r.supportNonCritical;
			int lhsCount = r.leftSideCountNonCritical;
			int index;

			for (BitSet b : candidateSolution) {
				
				boolean containsLeft = true;
				boolean containsRight = true;

				for (Integer litem : r.leftSide) {
					index = distinctItems.get(litem);
					
					if (!b.get(index)) {
						containsLeft = false;
						break;
					}
				}
				if (containsLeft) {
					lhsCount++;
					for (Integer ritem : r.rightSide) {
						index = distinctItems.get(ritem);

						if(!b.get(index)) {
							containsRight = false;
							break;
						}
					}
					if (containsRight) support++;
				}
			}

			r.sup = (double) support / tidcount;
			r.conf = lhsCount == 0 ? 0 : (double) support / lhsCount;

			if (r.sup < minsup || r.conf < minconf) lrr++;
		}

		hfr /= sars.size();
		lrr /= (nars.size() + critNars.size());
		
		double osd;
		double a = 0;
		for (Rule l : sars) {
			a += Math.max(Math.min(l.conf - minconf, l.sup - minsup), 0);
		}

		double b = 0;
		for (Rule u : critNars) {
			b += Math.max(Math.max(minconf - u.conf, minsup - u.sup), 0);
		}
		for (Rule u : nars) {
			double supTotal = u.supportCritical + u.supportNonCritical;
			double lhsCountTotal = u.leftSideCountCritical + u.leftSideCountNonCritical;
			double sup = (double) supTotal / tidcount;
			double conf = (double) supTotal / lhsCountTotal; // risk of zero div? no right?
			b += Math.max(Math.max(minconf - conf, minsup - sup), 0);
		}

		osd = a / sars.size() + b / (nars.size() + critNars.size());

		// severely penalize complete failure
		if (hfr == 1) hfr += 100;
		if (lrr == 1) lrr += 100;

		double fitness = w1 * hfr + w2 * lrr + w3 * osd;

		return fitness;
	}

	private Set<Integer> generateSensitiveItems(List<Rule> sars, List<Rule> nars) {
		// each item is mapped to two counts.
		// the count at index 0 of the list tracks the occurrences of the item in the rhs of a SAR
		// the count at index 1 of the list tracks the occurrences of the item in a NAR (either side)
		HashMap<Integer, Integer> occurrenceCountsSAR = new HashMap<>();
		HashMap<Integer, Integer> occurrenceCountsNAR = new HashMap<>();
		Set<Integer> sensitiveItems = new HashSet<>();

		// Count the frequency of each item in the RHS of SARs
		for (Rule r : sars) {
			
			for (Integer i : r.rightSide) {
				// if the item has not yet been seen, add it
				// to the map with a count of 1; otherwise,
				// add 1 to the existing count.
				
				if(occurrenceCountsSAR.containsKey(i)) {
					int val = occurrenceCountsSAR.get(i) + 1;
					occurrenceCountsSAR.put(i, val);
				} else {
					occurrenceCountsSAR.put(i, 1);
				}
				
			}
		}

		// Count the frequency of each item in NARs
		for (Rule r : nars) {
			Set<Integer> items = new HashSet<>(r.leftSide);
			items.addAll(r.rightSide);
			for (Integer i : items) {	// items is just the set of every item in the NAR
				if (occurrenceCountsNAR.containsKey(i)) {
					int val = occurrenceCountsNAR.get(i) + 1;
					occurrenceCountsNAR.put(i, val);
				} else {
					occurrenceCountsNAR.put(i, 1);
				}
			}
		}

		// Make another pass through the sensitive rules. 
		// If the rule has only 1 item in RHS, designate it sensitive.
		// Otherwise, choose the item in RHS that has the highest frequency in
		// RHS of SARs. If there is a tie, choose the item that has the lowest
		// frequency in NAR.
		for (Rule r : sars) {
			if (r.rightSide.size() == 1) {
				sensitiveItems.add(r.rightSide.iterator().next());
			} else {
				// find the item in rhs that has the highest frequency in RHS of SAR.
				int maxCountSAR = -1;
				Set<Integer> candidateItems = new HashSet<>();
				for (Integer i : r.rightSide) {
					if (occurrenceCountsSAR.get(i) > maxCountSAR)
						maxCountSAR = occurrenceCountsSAR.get(i);
				}
				// find all items that occur with that frequency
				for (Integer i : r.rightSide) {
					if (occurrenceCountsSAR.get(i) == maxCountSAR)
						candidateItems.add(i);
				}

				// if there is only one, add that item
				if (candidateItems.size() == 1) {
					sensitiveItems.add(candidateItems.iterator().next());
				} else {
				// otherwise, find the item with lowest occurrence in NAR
				// in case of a tie, just pick one.
					int minCountNAR = Integer.MAX_VALUE;
					Integer sensitiveItem = candidateItems.iterator().next();
					for (Integer i : r.rightSide) {
						if (occurrenceCountsNAR.get(i) < minCountNAR) {
							minCountNAR = occurrenceCountsNAR.get(i);
							sensitiveItem = i;
						}
					}
					sensitiveItems.add(sensitiveItem);

				}

			}
		}

		return sensitiveItems;
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

	private void writeResultToFile(Set<Integer> solution, List<Transaction> critical,
								   Set<Integer> sensitiveItems, String input, String output)
								   throws IOException {
		// Now, write the transformed transaction database to disk!
		BufferedReader reader = new BufferedReader(new FileReader(input));
		BufferedWriter writer = new BufferedWriter(new FileWriter(output));
		int count = 0;
		int idx = 0;

		String line;
		
		while (((line = reader.readLine()) != null)) {

			if(line.isEmpty()) continue;

			count++;

			// Is this transaction one of the critical ones? (idx lets us go through critical from the top)
			if (idx < critical.size() && critical.get(idx).tid == count) {
				// Is this critical transaction designated to be changed by the solution?
				// If so, remove sensitive items. Otherwise write it as is.
				if (solution.contains(critical.get(idx).criticalIndex)) {
					// write only nonsensitive items from this transaction.
					for(Integer item : critical.get(idx).items) {
						if(!sensitiveItems.contains(item))
							writer.write(item + " ");
					}
				} else {
					writer.write(line);
				}
				idx++;
			} else {
				writer.write(line);
			}

			writer.newLine();
		}

		writer.close();
	}

}