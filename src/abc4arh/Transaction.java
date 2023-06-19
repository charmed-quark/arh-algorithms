package abc4arh;

/** 
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

public class Transaction {
    // In the case of multiple transactions containing
    // the same set of items, we store all their IDs
    // in this set. The size of the set can then be
    // used to adjust support counts, i.e. if the
    // transaction is chosen for deletion and has 2
    // duplicates, the support should decrease by 3.
    //Set<Integer> ids;

    // the id of this transaction. the duplicate thing isn't working out.
    int id;

    // The items in this transaction.
    Set<Integer> items;

    Transaction(int id, Set<Integer> items) {
        //ids = new HashSet<>();
        //ids.add(id);
        this.id = id;
        this.items = new HashSet<>(items);
    }

}
