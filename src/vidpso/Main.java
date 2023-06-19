package vidpso;

/** 
 *  Runner code for VICTIM ITEM DELETION-BASED PARTICLE SWARM OPTIMIZATION
 *  (VIDPSO)
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

import java.io.IOException;

class Main {
    public static void main(String[] args) throws IOException {
        System.out.println("Running VIDPSO algorithm.");
        /* 
         * Parameters:
         * input file path
         * SAR file path
         * NAR file path
         * output file path
         * minsup
         * minconf
         */

        String[] inputNamesSAR = {"sar_00", "sar_01", "sarweak_00", "sarweak_01"};
        String[] outputNames = {"random00", "random01", "weakfreq00", "weakfreq01"};
        String directory = "/home/charlie/thesis/";

        final int NUM_RUNS = 5;

        // Settings CHESS
        String dataset = "chess";
        double minsup = 0.9;
        double minconf = 0.9;

        // Settings MUSHROOM
        /*String dataset = "mushrooms";
        double minsup = 0.4;
        double minconf = 0.7;*/

        // Settings BMS1
        /*String dataset = "bms1";
        double minsup = 0.001;
        double minconf = 0.05;*/

        // Settings RETAIL
        /*String dataset = "retail";
        double minsup = 0.001;
        double minconf = 0.05;*/

        // Settings SYN1K
        /*String dataset = "syn1K";
        double minsup = 0.01;
        double minconf = 0.05;*/

        // Settings SYN10K
        /*String dataset = "syn10K";
        double minsup = 0.01;
        double minconf = 0.05;*/

        // Settings SYN100K
        /*String dataset = "syn100K";
        double minsup = 0.01;
        double minconf = 0.05;*/
        
        String input = directory + dataset + "/" + dataset + ".txt";  //e.g. "/home/charlie/thesis/chess/chess.txt"

        for (int inputRun = 0; inputRun < inputNamesSAR.length; inputRun++) {
            String inputSAR = directory + dataset + "/rules/" + dataset + "_" + inputNamesSAR[inputRun] + ".txt";
            String inputAllRules = directory + dataset + "/rules/" + dataset + "_arm.txt";

            for (int i = 0; i < NUM_RUNS; i++) {
                System.out.println("############ VIDPSO RUN no." + i + " ############");
                System.out.println("DATASET: " + dataset + "; INPUT TYPE: " + outputNames[inputRun]);
                String output = directory + dataset + "/measurements/vidpso/vidpso_" + outputNames[inputRun] + "_sanitized_0" + i + ".txt";
                new VIDPSO().runAlgorithm(input, inputSAR, inputAllRules, output, minsup, minconf);
            }
        }

    }
}
