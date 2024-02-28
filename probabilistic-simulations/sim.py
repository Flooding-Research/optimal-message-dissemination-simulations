import math, random, csv, sys
import numpy as np
import tqdm

from enum import Enum
from functools import partial
from multiprocessing import Pool 

# --------------------------------------------------
# Global variables for the simulations
# --------------------------------------------------

# IMPORTANT: The script requires the folder [result_path] to exist or an error
# will be produced.
RESULT_PATH = './results/'  # <-- Ensure that this exists!

# Number of cores to be used when executing the simulations
NUMBER_OF_CORES = 1  # <-- Remember to set this!

# Variables for communication estimation.
HASH_SIZE = 32  # In bytes. Only used to estimate the size of the Merkle tree.
BLOCK_SIZE = 10 ** 6 # In bytes. Only used when estimating actual pr. party
                     # communication.

# --------------------------------------------------

def main():
    """Run simulations and export results"""

    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python sim.py <number_of_repetetions> <figure_number>")
        sys.exit(1)
    
    # Get the command-line arguments
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    
    # Convert arguments to integers
    try:
        number_of_repetitions = int(arg1)
        figure_number = int(arg2)
    except ValueError:
        print("Error: Both arguments must be integers")
        sys.exit(1)
    
    # Print the converted integers
    print("Argument 1:", arg1)
    print("Argument 2:", arg2)

    # Produce the data for the relevant figure
    if figure_number in [1,10]:
        produce_data_for_fig_1_10(number_of_repetitions)
    elif figure_number in [2,3,4,5]:
        produce_data_for_fig_2_3_4_5(number_of_repetitions)
    elif figure_number in [6,7]:
        produce_data_for_fig_6_7(number_of_repetitions)
    elif figure_number in [8]:
        produce_data_for_fig_8(number_of_repetitions)
    elif figure_number in [9]:
        produce_data_for_fig_9(number_of_repetitions)
    else:
        print("Error: <number_of_repetitions> must be an integer in {1,...,10}")
        sys.exit(1)
        
def produce_data_for_fig_1_10(iterations):
    """Produce data for fig 1 and 10"""

    # Define the config for FFlood for the figure
    config = Config()
    config.number_runs = iterations
    config.number_of_parties = [4096, 8192, 16384]
    config.expected_degrees = [x* 3 for x in range(1,21)]
    config.number_shares_list = [1]

    # Simulate FFlood for the figures
    results = config.run()

    # Export the results
    export_weak_flood_latency_per_n(results, config)    
    
    # Define the config for ECFlood(8) for the figures
    config = Config()
    config.number_runs = iterations
    Config.number_of_parties = [4096, 8192, 16384]
    config.expected_degrees = [8]
    config.number_shares_list = [25]

    # Simulate ECFlood(8) for the figures
    results = config.run()

    # Export the results
    export_flood_amplifier_per_n_d_and_mu(results, config)

    # Define the config for ECFlood(20) for the figures
    config = Config()
    config.number_runs = iterations
    config.number_of_parties = [4096, 8192, 16384]
    config.expected_degrees = [20]
    config.number_shares_list = [10]

    # Simulate ECFlood(20) for the figures
    results = config.run()

    # Export the results
    export_flood_amplifier_per_n_d_and_mu(results, config)
    
def produce_data_for_fig_2_3_4_5(iterations):
    """Produce data for figure 2, 3, 4, and 5"""

    # Define the config for FFlood
    config = Config()
    config.number_runs = iterations
    config.number_of_parties = [2 ** x for x in range (9,15)]
    config.expected_degrees = [x for x in range(1,46)]
    config.number_shares_list = [1]

    # Simulate FFlood
    results = config.run()

    # Export the results
    export_weak_flood_per_n(results, config) # Fig 2, 4, 5
    export_weak_flood_per_n_and_d(results, config) # Fig 3

def produce_data_for_fig_6_7(iterations):
    """Produce data for figure 6 and 7"""

    # Define the config for EClood
    config = Config()
    config.number_runs = iterations
    config.number_of_parties = [8192]
    config.expected_degrees = [x*2+1 for x in range(2,8)] + [20]
    config.number_shares_list = [x*3 for x in range(1,50)] # <-- If too slow
                                                           # then granualarity
                                                           # of this can be
                                                           # changed.
    config.max_neighbors = 545

    # Simulate ECFlood
    results = config.run()
    
    # Export the results
    export_flood_amplifier_per_n_and_d(results, config)

def produce_data_for_fig_8(iterations):
    """Produce data for figure 8"""
    
    # Define the config for EClood
    config = Config()
    config.number_runs = iterations
    config.number_of_parties = [8192]
    config.expected_degrees = [x for x in range(3,11)]
    config.number_shares_list = [20]
    
    # Simulate ECFlood
    results = config.run()

    # Export the results
    export_flood_amplifier_per_n_d_and_mu(results, config)

def produce_data_for_fig_9(iterations):
    """Produce data for figure 9"""

    # Define the config for EClood
    config = Config()
    config.number_runs = iterations
    config.number_of_parties = [2 ** x for x in range (10,15)] 
    config.expected_degrees = [x for x in range(5,21)]
    config.number_shares_list = [30]

    # Simulate ECFlood
    results = config.run()

    # Export the results
    export_flood_amplifier_latency_per_n_and_mu(results, config)

class WeakFloodingProtocol(Enum):
    """Enum for defining which weak flooding protocol to simulate"""

    ERFLOOD = 1 # Erdös-Renyi Flood. Each party sends the message to all other
                # parties with probability expected degree/number of parties
    FFLOOD = 2 # Fanout-Flood. Each party simply chooses expected degree number
               # of neighbors at random.

    def __str__(self):
        if self.name == 'ERFLOOD':
            return 'ER'
        if self.name == 'FFLOOD':
            return 'FF'

class Config:
    """Class to encapsulate a configuration for ECFlood"""

    def __init__(self):
        """Constructor for a config. Variables should be explicitly assigned after
        construction"""

        # Initialize variables to default

        # The number of simulations that should be performed for each set of
        # parameters.
        self.number_runs = 10 ** 2 
        # A list of number of parties' that are to be simulated.
        self.number_of_parties = []
        # The fraction of parties that should be corrupted.
        self.corruption_frac = 0.5 
        # A list of expected degrees. 
        self.expected_degrees = [] 
        # A list of number of shares a message should be shared into.
        self.number_shares_list = [] #
        # Set the protocol to be FFLOOD by default. This if should be changed
        # if Erdös-Renyi simulations are needed instead instead
        self.protocol = WeakFloodingProtocol.FFLOOD 
        # Set the maximum number of neighbors of a config. The default is that
        # there are no max on the number of neighbors.
        self.max_neighbors = sys.maxsize # 

    def print(self):
        """Print the content of a config"""
        print("-------------------- Configuration --------------------")
        print("Number of simulations pr. configuration:", self.number_runs)
        print("Number of parties:", self.number_of_parties)
        print("Fraction of corrupted parties:", self.corruption_frac)
        print("Number of share list:", self.number_shares_list)
        print("Expected degrees:", self.expected_degrees)
        print("Mode:", self.protocol)
        print("Maximum Neighbors:", self.max_neighbors)
        print("-------------------------------------------------------")

    def run(self):
        """Run the simulations defined by the config"""

        # Print config
        self.print()

        # Simulate  this config
        results = simulate_config_par(self)

        print('Simulations done.')
        print('Exporting the results...')

        return results

class SimulationResult:
    """A class that represent a result of a simulation"""

    def __init__(self,
                 id,
                 n,
                 d,
                 mu,
                 party_one_received,
                 min_shares_received,
                 frac_parties_received,
                 max_dist,
                 max_dist_to_receive_shares):
        # Initialize variables
        self.number_of_parties = n
        # Id of the simulation (not used)
        self.sim_id = id
        # The degree used for the simulation
        self.degree = d
        # Number of shares used in the simulation
        self.number_of_shares = mu
        # A bool that describes if the first party has received the message
        self.party_one_received = party_one_received 
        # The minimum number of shares received 
        self.min_shares_received = min_shares_received 
        # The maximum distance of any party to receive a share
        self.max_dist = max_dist
        # The fraction of parties having received any share
        self.frac_parties_received = frac_parties_received
        # An arary that contain that maximum distance for any party to receive
        # share i at index i-1
        self.max_dist_to_receive_shares = max_dist_to_receive_shares

class Party:
    """ A class to represent a party"""

    def __init__(self, id):
        """Constructor for a party. The corruption status should be explicitly assigned
        after construction

        """

        self.id = id            # The identity of the party.
        self.corrupted = False  # Whether or not this party is corrupted.
        self.received_msgs = {} # A dictionary containing the messages this
                                # party received and the distance they received
                                # each message at.


    def shares_within_dist(self, dist):
        """Returns the number of shares received within distance dist"""

        return len([d for b, d in self.received_msgs.values() if d <= dist])
    
    def max_dist_to_receive_shares(self):
        """Returns the a list that contains the minimum distance for this party to
        receive i shares at index i-1."""

        # Get the distances for all received shares
        dists = [d for _, d in self.received_msgs.values()]
        
        # Sort the distances
        dists.sort()
        
        # Return the value
        return dists

    def max_dist(self):
        """Calculate maximum distance of any received message. Messages that has not
        been received are considered to have distance 0"""

        if self.received_msgs:
            return max([d for b,d in self.received_msgs.values()])
        return 0
    

def create_parties(number_of_parties):
    """Method to initialize a list of part"""

    # Create a list of parties with ids 0,1,...,number_of_parties-1
    return [Party(id) for id in range(number_of_parties)]

def corrupt_parties(parties, corruption_frac):
    """ Method to corrupt a certain number of parties"""

    # We corrupt by letting every party above a certain id be corrupted. 
    corruption_above_id = len(parties) - math.floor(len(parties) * corruption_frac)
    
    # It simply corrupts the last parties list of parties
    for p in parties:
        p.corrupted = p.id >= corruption_above_id

def setup_parties(number_of_parties, corruption_frac):
    """Setup number_of_parties and corrupt the appropiate number"""

    # Initialize a list of parties
    parties = create_parties(number_of_parties)    
    
    # Corrupt parties
    corrupt_parties(parties, corruption_frac)

    return parties
    
def choose_neighborhood(parties, expected_degree, protocol):
    """Draw neighborhood in an efficient ER manner, such that each party is
    included in the neighborhood with prqobability expected_degree/number of
    parties"""

    # Instantiate the variable before if-clauses which sets it.
    number_of_neighbors = 0

    if protocol == WeakFloodingProtocol.ERFLOOD:
        # Calculate probability to pick each party
        p = expected_degree/len(parties)
    
        # Draw number of neighbors to be drawn
        number_of_neighbors = np.random.binomial(len(parties), p)
    
    if protocol == WeakFloodingProtocol.FFLOOD:

        # Set number of neighbors directly to expected degree
        number_of_neighbors = expected_degree

    # Draw neighbors from list of parties
    neighbors = random.sample(parties,number_of_neighbors)

    return neighbors

def simulate(parties, expected_degree, number_of_shares, protocol): 
    """ Simulate an execution among parties where a sender sends
    number_of_shares shares out."""

    # Create a queue of tuples. The tuples consist of parties that are
    # to forward a message, a message and the distance they got the
    # message at. The queue will only contain honest parties and each
    # honest party will at most be added once.
    to_forward = []

    for s in range(number_of_shares): 
        # Initially only the sender is in this queue (with index 0 in
        # the list of parties)
        to_forward.append((parties[0], s, 0))

        # Let the inital sender receive the message
        parties[0].received_msgs[s] = (True, 0)

    # As long as their are yet parties to forward the message the
    # simulation continues
    while len(to_forward) > 0:
        
        # pop the next party to send, the message to send, and the
        # distance they got it delivered at.
        sender, msg, dist = to_forward.pop(0)
                
        # Draw the neighbors
        neighbors = choose_neighborhood(parties, expected_degree, protocol)
        
        # Add these to the queue of parties waiting to receive and
        # update that s has now forwarded the message
        for p in neighbors: 
            # Check if party received the message. Note that in Python
            # any non-empty tuple is considered True.
            if not p.received_msgs.get(msg):
                p.received_msgs[msg] = (True, dist + 1)

                # If p is honest add p to to_forward.
                if not p.corrupted:
                    to_forward.append((p, msg, dist + 1))

def run_simulation(corruption_frac,
                   protocol,
                   n,
                   d,
                   number_of_shares,
                   simulation_number):
    """Create and run a particular simulation"""

    # Instantiate parties
    parties = setup_parties(n, corruption_frac)

    # Do the actual simulation
    simulate(parties, d, number_of_shares, protocol)
                
    # Collect statistics
    
    # Minimum number of received shares for any party
    min_shares_received = min([len(p.received_msgs) for p in parties])

    # Get the lists of sorted lists of distances to receive a number of shares
    # of index + 1
    sorted_dists_lists = [p.max_dist_to_receive_shares() for p in parties]

    # Contains maximum distance for any party to receive i shares at index i-1
    max_dist_to_receive_shares = [max([r[mu - 1] for r in sorted_dists_lists])
                                  for mu in range(1, min_shares_received + 1)]
        
    # Check if party one received message one
    party_one_received_msg_one = bool(parties[1].received_msgs.get(0))

    # Calculate the fraction of reached parties
    fraction_of_parties_that_received = len([p for p in parties if
                                             len(p.received_msgs)]) / n
    
    # Calculate maximum distance of any received share
    max_dist = max([p.max_dist() for p in parties])

    # Create and return the simulation result
    return SimulationResult(simulation_number,
                            n,
                            d,
                            number_of_shares,
                            party_one_received_msg_one,
                            min_shares_received,
                            fraction_of_parties_that_received,
                            max_dist,
                            max_dist_to_receive_shares)

def simulate_config_par(config): 
    """Run all simulations of a configuration in parrallel"""

    # Instantiate list of results
    results = []

    # Create the function that should be mapped
    f = partial(run_simulation, config.corruption_frac, config.protocol)

    with Pool(NUMBER_OF_CORES) as p: 
        
        # Create list of all configurations that should be simulated
        configs_to_run = [(n, d, mu, r)
                          for n in config.number_of_parties
                          for d in config.expected_degrees
                          for mu in config.number_shares_list
                          for r in range(1, config.number_runs +1)
                          if mu * d <= config.max_neighbors]

        # Obtain results with a progress bar
        results = list(p.starmap(f, tqdm.tqdm(configs_to_run,total = len(configs_to_run))
                                 , chunksize = 1))

    return results

# --------------------------------------------------
# Utility functions for calculating communication complexity of ECFlood
# --------------------------------------------------

def per_party_communication(reconstruction_frac, number_of_shares, expected_degree):
    """ The per party communication in mega bytes"""

    # We follow the upper bound on pr. party communication in the paper. Note
    # that \rho * n in the paper is equal to [expected_degree].
    communication = number_of_shares * expected_degree * (share_size(number_of_shares, reconstruction_frac)
                                                          + share_number_size(number_of_shares)
                                                          + acc_proof_size(number_of_shares)
                                                          + acc_size())
    # Convert this to MB before returning
    return communication / (10 ** 6)

def share_size(number_of_shares, reconstruction_frac):
    """Calculate the share_size"""
    
    # The share size is length of the message divided by the number of shares
    # times the fraction of shares need for reconstruction.
    bits_pr_share = math.ceil(safe_div(BLOCK_SIZE * 8, number_of_shares *
                                       reconstruction_frac))

    # We calculate this in bytes
    return bits_pr_share/8 

def share_number_size(number_of_shares):
    """Calculate the size of the number used to identify a share."""
    return math.ceil(math.log2(number_of_shares)) / 8

def acc_size():
    """The size of an accumulator in bytes when implemented as a Merkle-tree."""
    return HASH_SIZE            

def acc_proof_size(number_of_elements):
    """The size of a proof for an accumulator with [number_of_elements] elements
    when implemented as a Merkle-tree"""

    # This is simply the depth of the merkle tree multiplied with the size the
    # hash.
    return HASH_SIZE * math.ceil(math.log2(number_of_elements))

# --------------------------------------------------
# Utility functions for exporting data from the simulations to csv
# --------------------------------------------------

def export_weak_flood_per_n(results, config):
    """Export weak flooding results one file for each different number of
    parties"""
    
    # Iterate through the different number of parties used. 
    for real_n in config.number_of_parties:

        # Initialize list of data points
        data = []
        
        # Add one data point for each different degree
        for real_d in config.expected_degrees:
            
            # Filter out irrelevant results and obtain results for party one
            # and maximum distance for any party to receive the message
            relevant_results = [(r.party_one_received, r.max_dist) for r in
                                results if r.number_of_parties == real_n and
                                r.degree == real_d and r.number_of_shares == 1]

            
            # Unzip the results
            succeses, max_dists = list(zip(*relevant_results))
            
            # Calculate delivery estimate as the fraction of successes
            estimate = sum(succeses) / len(succeses)

            # Calculate the redundancy estimate as the degree divided by the
            # delivery estimate
            redundancy_estimate = safe_div(real_d, estimate)
            
            # Calculate the maximum distance for any received message among all results
            max_max_dist = max(max_dists)

            # Calculate the mean maximum distance among the simulations
            mean_max_dist = sum(max_dists) / len(max_dists)
            
            # Append the data to the data points that should be exported
            data.append((real_d, estimate, redundancy_estimate, max_max_dist,
                         mean_max_dist))


        # Create file name from the parameters
        filename = RESULT_PATH + str(config.protocol) + 'WeakFlood-delivery-estimate-n-' + str(real_n) + '-r-'+str(config.number_runs) + '.csv'

        # Create header for csv file
        header = ['degree','delivery estimate', 'redundancy estimate' ,
                  'maximum max dist', 'mean max dist']

        # Export data to csv          
        write_to_csv(filename, header, data)        

def export_weak_flood_per_n_and_d(results, config):
    """Export weak flooding results one file for each combination of different
    number of parties and degree"""

    # Iterate through number of parties
    for real_n in config.number_of_parties:
        
        # Iterate through different degrees
        for real_d in config.expected_degrees:
            
            # Obtain the fraction of reached parties for each relevant simulation 
            frac_reached_parties = [r.frac_parties_received for r in results
                                    if r.number_of_parties == real_n
                                    and r.degree == real_d
                                    and r.number_of_shares == 1]

            # Sort the results
            frac_reached_parties.sort()

            # The data for the y-axis is simply the fraction of data points higher than or equal to this index
            frac_simulations = [(len(frac_reached_parties) - i) /
                                len(frac_reached_parties) for i in
                                range(0,len(frac_reached_parties))]
            

            # Create file name
            filename = RESULT_PATH + str(config.protocol) + 'WeakFlood-n-' + str(real_n) + '-d-' + str(real_d) +'-r-' + str(config.number_runs) +'.csv'

            # Create header for the csv file
            header = ['fraction of reached parties', 'fraction of simulations']

            # Zip the lists of data points
            data = list(zip(frac_reached_parties, frac_simulations))

            # Export data to csv
            write_to_csv(filename, header, data)

def export_flood_amplifier_per_n_and_d(results, config):
    """Export flooding amplifications results, one file for each combination of different
    number of parties and degree"""

    # Iterate through the number of parties
    for real_n in config.number_of_parties:
        
        # Iterate through the different degrees
        for real_d in config.expected_degrees:

            # Initialize list of data points for specific n and d 
            data = []

            # Collect results for different number of shares (mu)
            for real_mu in config.number_shares_list: 

                # Obtain the minimum number of shares received by any party and
                # the maximum distance to receive a share for the relevant
                # results
                relevant_results = [(r.min_shares_received, r.max_dist_to_receive_shares) for r in results
                                     if r.number_of_parties == real_n
                                    and r.degree == real_d
                                    and r.number_of_shares == real_mu]

                # In case there are some relevant results
                if len(relevant_results) > 0:

                    # Unzip results
                    min_min_shares, max_dists_to_receive_shares = map(list,(zip(*relevant_results)))

                    # Get minimum number of shares received across all
                    # simulations
                    min_received_shares = min(min_min_shares)
                    # Calculate this as a fraction
                    min_received_shares_frac = min_received_shares / real_mu
                    # Calculate redundancy as degree over the fraction of
                    # received shares
                    redundancy_estimate = safe_div(real_d,
                                                   min_received_shares_frac)
                    # Calculate the maximum number of neighbors of any honest
                    # party
                    expected_neighbors = real_mu * real_d
                    # Calculate the maximum distance across all simulations to
                    # receive the minimum number of shares received in any simulation
                    max_max_dist_to_min_share = 0
                    if min_received_shares > 0: 
                        max_max_dist_to_min_share = max([r[min_received_shares-1] for r in
                             max_dists_to_receive_shares])


                    # Append data to axis
                    data.append((real_mu,
                                 min_received_shares_frac,
                                 redundancy_estimate,
                                 max_max_dist_to_min_share,
                                 expected_neighbors))
                
            # Construct the file name for the appropriate parameters
            filename = RESULT_PATH + str(config.protocol) + 'FloodAmplifier-n-' + str(real_n) +'-d-' + str(real_d) + '-r-' + str(config.number_runs)+ '.csv'
              
            # Construct the header for the csv file
            header = ['number of shares',
                      'minimum fraction of received shares for all parties in any simulation',
                      'redundancy estimates',
                      'maximum max dist',
                      'expected number of neighbors']

            # Export data to csv
            write_to_csv(filename, header, data)

def export_flood_amplifier_per_n_d_and_mu(results, config):
    """Export flooding amplifications results, one file for each combination of different
    number of parties, degrees and number of shares"""

    # Iterate through all combinations
    for real_n in config.number_of_parties:
        for real_d in config.expected_degrees:
            for real_mu in config.number_shares_list: 

                # Export results for a share number that is a multilple of 5
                if real_mu % 5 == 0: 

                    # Obtain the minimum fraction of shares and maximum
                    # distance to receive shares for relevant results
                    relevant_results = [(r.min_shares_received / real_mu,
                                         r.max_dist_to_receive_shares) for r in
                                        results if r.number_of_parties ==
                                        real_n and r.degree == real_d and
                                        r.number_of_shares == real_mu]
                    
                    # Unzip results
                    min_frac_shares, max_dists_to_receive_shares = map(list,(zip(*relevant_results)))

                    # Sort the results
                    min_frac_shares.sort()

                    # Calculate the fraction of simulations with minimum
                    # fraction of received shares higher than or equal to this
                    # index
                    frac_simulations = [(len(min_frac_shares) - i) / len(min_frac_shares) for i in range(0,len(min_frac_shares))]

                    # Initialize list of data points
                    data_points = []

                    # Filter out shares where percentages of received shares does not change.
                    old_share_frac = -1
                    for frac_shares, frac_sims in zip(min_frac_shares,frac_simulations):
                        if frac_shares > old_share_frac:
                            data_points.append((frac_shares, frac_sims))
                            old_share_frac = frac_shares
                    
                    # Add extra data point for the smallest fraction of shares that no runs satisfy
                    last_frac_where_some_simulations_received, _ = data_points[-1]
                    if last_frac_where_some_simulations_received < 1: 
                        data_points.append((last_frac_where_some_simulations_received + (1 / real_mu ), 0))

                    
                    # Update all data points with per part communication estimates and latency.
                    data_points = [(min_shares,
                                    sims,
                                    per_party_communication(min_shares,
                                                            real_mu, real_d),
                                    get_max_dist_to_share_if_any(max_dists_to_receive_shares,
                                                                 math.ceil(min_shares * real_mu)),
                                    safe_div(real_d, min_shares) ) for
                                   (min_shares, sims) in data_points] 
                        

                    # Create file name
                    filename = RESULT_PATH + str(config.protocol) + 'FloodAmplifier-n-' + str(real_n) +'-d-' + str(real_d) + '-mu-'+ str(real_mu) +'-r-' + str(config.number_runs) +'.csv'

                    # Create header for csv file
                    header = ['Shares received by all parties',
                              'Simulations',
                              'Pr. party communication in MB',
                              'Latency',
                              'Naive redundancy estimate']

                    # Export data to csv
                    write_to_csv(filename, header, data_points)

def get_max_dist_to_share_if_any(max_dists_to_receive_shares, number_of_shares):
    """Get maxímum distance to receive a certain number of shares if such maximum
    distance exists"""

    # Only consider maximum distances which has more received shares than the threshold
    filtered_max_dists = [max_dist_to_share for max_dist_to_share in max_dists_to_receive_shares if len(max_dist_to_share) >= number_of_shares]

    if len(filtered_max_dists) > 0 and number_of_shares > 0:
        return max([r[number_of_shares - 1] for r in filtered_max_dists])
    else:
        return 0
    
def export_flood_amplifier_latency_per_n_and_mu(results, config):
    """Export flooding amplifications results about latency, one file for each combination of different
    number of parties and number of shares"""

    # Iterate through relevant parameters
    for real_n in config.number_of_parties:
         for real_mu in config.number_shares_list: 
             # Initialize list of data points
             data = []

             # Iterate through each degree
             for real_d in config.expected_degrees:

                 # Obtain the minimum received shares and maximum
                 # distances for relevant results
                 relevant_results = [(r.min_shares_received,
                                      r.max_dist_to_receive_shares) for r in
                                     results if r.number_of_parties == real_n
                                     and r.degree == real_d and
                                     r.number_of_shares == real_mu]

                 # If there are relevant results
                 if len(relevant_results) > 0:

                    # Unzip lists
                    min_shares, max_dists_to_receive_shares = map(list,(zip(*relevant_results)))

                    # Obtain the minimum received shares across all simulations
                    min_min_received_shares = min(min_shares)
                    # Obtain the minimum fraction of shares across all simulations
                    min_received_shares_frac = min_min_received_shares / real_mu
                    # Calculate the redundancy estimate
                    redundancy_estimate = safe_div(real_d, min_received_shares_frac)
                    # Calculate the maximum number of neighbors for any honest party
                    expected_neighbors = real_mu * real_d

                    # Calculate the maximum distance to receive the minimum
                    # number of shares across all simulations
                    max_max_dist_to_min_share = 0
                    if min_min_received_shares > 0: 
                       max_max_dist_to_min_share = max([r[min_min_received_shares-1] for r in
                            max_dists_to_receive_shares])

                    # Append data to axis
                    data.append((real_d,
                                 min_received_shares_frac,
                                 redundancy_estimate,
                                 max_max_dist_to_min_share,
                                 expected_neighbors))

             # Create file name 
             filename = RESULT_PATH + str(config.protocol) + 'FloodAmplifier-latency-n-' + str(real_n) + '-mu-' + str(real_mu) +'-r-' + str(config.number_runs)+ '.csv'
             # Create header for csv file
             header = ['degree',
                       'minimum fraction of received shares for all parties in any simulation',
                       'redundancy estimates',
                       'maximum max dist',
                       'expected number of neighbors']
             # Export data to csv
             write_to_csv(filename, header, data)

def export_weak_flood_latency_per_n(results, config):
    """Export weak flooding results about latency, one file for each combination of different
    number of parties."""


    # Iterate through the number of parties
    for real_n in config.number_of_parties:
        
        # Instantiate variable to save relevant data
        data_points = []
        for real_d in config.expected_degrees:
        
            # Obtains distances for a share number of 1 where all parties
            # received at least one share
            relevant_results = [r.max_dist_to_receive_shares for r in results
                                if r.number_of_parties == real_n
                                and r.degree == real_d
                                and r.number_of_shares == 1
                                and r.min_shares_received == 1]

            # Calculate the fraction of simulations where all parties received
            success_rate = len(relevant_results) / config.number_runs

            # Calculate the maximum distance to receive the share across all simulations
            max_max_dist_to_min_share = 0
            if success_rate > 0: 
                max_max_dist_to_min_share = max([r[0] for r in relevant_results])
            
            # Append data to list of data points
            data_points.append((success_rate, real_d, max_max_dist_to_min_share))

        # Create file name
        filename = RESULT_PATH + str(config.protocol) + '-n-' + str(real_n) + '-r-' + str(config.number_runs) + '.csv'
        # Create header for csv file
        header = ['Success rate','degree', 'latency']
        # Export data to csv
        write_to_csv(filename, header, data_points)

def write_to_csv(filename, header, data):
    """Utility function that writes to csv"""

    # Open file and create if it does not exist
    file = open(filename,'w+')
    
    # Create writer
    writer = csv.writer(file)

    # Write a header
    writer.writerow(header)
                
    # Write the data
    writer.writerows(data)

    # Close the file
    file.close()
    
    print('Wrote data to ' + filename + '.')

def safe_div(x,y):
    """Division that returns 0 if one divides by zero. This will only be
    used to calculate estimated redundancies. In the plots redundancies
    of 0 should be ignored."""
    if y == 0:
        return 0
    return x / y

if __name__  ==  '__main__':
    main()
