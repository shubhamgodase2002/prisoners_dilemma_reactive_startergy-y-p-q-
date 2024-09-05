#Import necessary libraries
import random
import matplotlib.pyplot as plt
import numpy as np

#Define the payoff matrix
def payoff_matrix():
    return {
        ('C', 'C'): (10, 10),
        ('C', 'D'): (0, 30),
        ('D', 'C'): (30, 0),
        ('D', 'D'): (20, 20)
    }

matrix = payoff_matrix()

# Define the prisoner's dilemma function
def prisoner_dilemma(player1, player2, matrix):
    return matrix[(player1, player2)]

#Define the reactive strategy for Player 1

def reactive_strategy_player1(y, p, q, opponent_action, matrix):
    if opponent_action == 'C':
        player1_action = 'C' if random.random() <= p else 'D'
    else:
        player1_action = 'C' if random.random() <= q else 'D'
    return player1_action

# Define the strategies for Player 2
def grim_trigger_strategy_player2(history, matrix):
    if 'D' in history:
        return 'D'
    else:
        return 'C'

def pavlov_strategy_player2(history, matrix):
    if len(history) == 0 or history[-1] == ('C', 'C'):
        return 'C'
    else:
        return 'D'

def discriminating_altruist_strategy_player2(history, matrix):
    return 'C'

def zd_strategy_player2(history, matrix):
    return random.choice(['C', 'D'])

def all_c_strategy_player2(history, matrix):
    return 'C'

def all_d_strategy_player2(history, matrix):
    return 'D'

def random_strategy_player2(history, matrix):
    return random.choice(['C', 'D'])

def tit_for_tat_strategy_player2(history, matrix):
    if len(history) == 0:
        return 'C'
    else:
        return history[-1][0]

def generous_tit_for_tat_strategy_player2(history, matrix):
    if len(history) == 0:
        return 'C'
    else:
        if history[-1][0] == 'D' and random.random() < 0.3:  # 30% chance to forgive
            return 'C'
        return history[-1][0]


    # comapre the results player1  and player2 
    def play_round_reactive_vs_other(strategy_func_2, y, p, q, rounds, matrix):
        history = []
        payoffs_p1 = []
        payoffs_p2 = []

        player2_action = random.choice(['C', 'D'])  # Player 2's first action is random
        player1_action = 'C' if random.random() <= y else 'D'  # Player 1's first action based on Reactive strategy

        for i in range(rounds):
            if i > 0:
                player2_action = strategy_func_2(history, matrix)

            player1_action = reactive_strategy_player1(y, p, q, player2_action, matrix)
            p1_payoff, p2_payoff = prisoner_dilemma(player1_action, player2_action, matrix)
            history.append((player1_action, player2_action))
            payoffs_p1.append(p1_payoff)
            payoffs_p2.append(p2_payoff)

        return sum(payoffs_p1), sum(payoffs_p2)

# inputs for reactive stratergy
y = 0.5
p = 0.7
q = 0.3
rounds = 100  # iteration for player 1 and player 2 
num_replications = 1000  #  replicate the result over 1000 simulation

grim_trigger_payoffs = np.zeros((num_replications, 2))
pavlov_payoffs = np.zeros((num_replications, 2))
discriminating_altruist_payoffs = np.zeros((num_replications, 2))
zd_payoffs = np.zeros((num_replications, 2))
all_c_payoffs = np.zeros((num_replications, 2))
all_d_payoffs = np.zeros((num_replications, 2))
random_payoffs = np.zeros((num_replications, 2))
tit_for_tat_payoffs = np.zeros((num_replications, 2))
generous_tit_for_tat_payoffs = np.zeros((num_replications, 2))

# run the and simulate  the results over 1000 iteration over all stratergy 
for i in range(num_replications):
    grim_trigger_payoffs[i] = play_round_reactive_vs_other(grim_trigger_strategy_player2, y, p, q, rounds, matrix)
    pavlov_payoffs[i] = play_round_reactive_vs_other(pavlov_strategy_player2, y, p, q, rounds, matrix)
    discriminating_altruist_payoffs[i] = play_round_reactive_vs_other(discriminating_altruist_strategy_player2, y, p, q, rounds, matrix)
    zd_payoffs[i] = play_round_reactive_vs_other(zd_strategy_player2, y, p, q, rounds, matrix)
    all_c_payoffs[i] = play_round_reactive_vs_other(all_c_strategy_player2, y, p, q, rounds, matrix)
    all_d_payoffs[i] = play_round_reactive_vs_other(all_d_strategy_player2, y, p, q, rounds, matrix)
    random_payoffs[i] = play_round_reactive_vs_other(random_strategy_player2, y, p, q, rounds, matrix)
    tit_for_tat_payoffs[i] = play_round_reactive_vs_other(tit_for_tat_strategy_player2, y, p, q, rounds, matrix)
    generous_tit_for_tat_payoffs[i] = play_round_reactive_vs_other(generous_tit_for_tat_strategy_player2, y, p, q, rounds, matrix)
   
# crete the function for plotting 
def plot_head_to_head_comparison(player1_payoffs, other_payoffs, strategy_name):
    strategies = ['Reactive (Player 1)', strategy_name + ' (Player 2)']
    player1_avg_payoff = np.mean(player1_payoffs[:, 0])
    player2_avg_payoff = np.mean(other_payoffs[:, 1])

    fig, ax = plt.subplots()
    bar1 = ax.bar(strategies[0], player1_avg_payoff, color='blue', label='Player 1 Payoff')
    bar2 = ax.bar(strategies[1], player2_avg_payoff, color='orange', label='Player 2 Payoff')

    ax.set_xlabel('Strategy')
    ax.set_ylabel('Total Payoff')
    ax.set_title(f'Reactive Strategy (Player 1) vs. {strategy_name} (Player 2)')
    
    ax.legend()  # Add the legend for the players
    plt.savefig(f'Reactive_vs_{strategy_name}.png')  # Save the figure with title name
    plt.show()

# call the function  Plot head-to-head comparison for each strategy bar plot 
plot_head_to_head_comparison(grim_trigger_payoffs, grim_trigger_payoffs, "Grim Trigger")
plot_head_to_head_comparison(pavlov_payoffs, pavlov_payoffs, "Pavlov")
plot_head_to_head_comparison(discriminating_altruist_payoffs, discriminating_altruist_payoffs, "Discriminating Altruist")
plot_head_to_head_comparison(zd_payoffs, zd_payoffs, "ZD Strategy")
plot_head_to_head_comparison(all_c_payoffs, all_c_payoffs, "All-C")
plot_head_to_head_comparison(all_d_payoffs, all_d_payoffs, "All-D")
plot_head_to_head_comparison(random_payoffs, random_payoffs, "Random")
plot_head_to_head_comparison(tit_for_tat_payoffs, tit_for_tat_payoffs, "Tit-for-Tat")
plot_head_to_head_comparison(generous_tit_for_tat_payoffs, generous_tit_for_tat_payoffs, "Generous Tit-for-Tat")

#   def function for print statistical results
def print_statistical_results(reactive, other, strategy_name):
    print(f"\nReactive Strategy (Player 1) vs. {strategy_name} (Player 2):")
    print(f"Player 1 wins: {np.sum(reactive[:, 0] > other[:, 1])} times")
    print(f"Player 2 wins: {np.sum(reactive[:, 0] < other[:, 1])} times")
    print(f"Ties: {np.sum(reactive[:, 0] == other[:, 1])} times")

# print statistcal results for each stratergy
print_statistical_results(grim_trigger_payoffs, grim_trigger_payoffs, "Grim Trigger")
print_statistical_results(pavlov_payoffs, pavlov_payoffs, "Pavlov")
print_statistical_results(discriminating_altruist_payoffs, discriminating_altruist_payoffs, "Discriminating Altruist")
print_statistical_results(zd_payoffs, zd_payoffs, "ZD Strategy")
print_statistical_results(all_c_payoffs, all_c_payoffs, "All-C")
print_statistical_results(all_d_payoffs, all_d_payoffs, "All-D")
print_statistical_results(random_payoffs, random_payoffs, "Random")
print_statistical_results(tit_for_tat_payoffs, tit_for_tat_payoffs, "Tit-for-Tat")
print_statistical_results(generous_tit_for_tat_payoffs, generous_tit_for_tat_payoffs, "Generous Tit-for-Tat")