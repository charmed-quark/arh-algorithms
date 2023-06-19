package samdp;

/** 
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

import java.util.HashSet;
import java.util.Set;

/**
 * This class is an association rule implementation.
 * A rule stores its support as well as the support of its 
 * antecedent, split up between critical and noncritical transactions.
 * The antecedent and consequent are stored as
 * as Set of Integers for fast inclusion checking.
 * 
 * @author Philippe Fournier-Viger
 * 
 * @author Charlie Mårtensson (modifications)
 */

class Rule {
	// the support of the rule in critical transactions
	int supportCritical;
	// the support of the antecedent (left side of the rule) in critical transactions
	int leftSideCountCritical;
	// the support of the antecedent in the noncritical transactions
	int leftSideCountNonCritical;
	// the support of the full generating itemset in the noncritical transactions
	int supportNonCritical;

	// the tidset of the antecedent (IDs of transaction containing the antecedent)
	Set<Integer> leftSide = new HashSet<Integer>();
	// the tidset of the consequent (IDs of transaction containing the consequent)
	Set<Integer> rightSide = new HashSet<Integer>();

	// values used during fitness calculation. will be overwritten.
	double sup;
	double conf;
}
