import math
import random
import matplotlib.pyplot as plt
from util import City, read_cities, write_cities_and_return_them, generate_cities, visualize_tsp, path_cost


class SimAnneal(object):
    def __init__(self, cities, temperature=-1, alpha=-1, stopping_temperature=-1, stopping_iter=-1):
        self.cities = cities
        self.num_cities = len(cities)
        self.temperature = math.sqrt(self.num_cities) if temperature == -1 else temperature
        self.T_save = self.temperature
        self.alpha = 0.999 if alpha == -1 else alpha
        # self.stopping_temperature = 1e-8 if stopping_temperature == -1 else stopping_temperature
        self.stopping_temperature = 1e-8 if stopping_temperature == -1 else stopping_temperature
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.route = None
        self.best_fitness = float("Inf")
        self.progress = []
        self.cur_cost = None

    def greedy_solution(self):
        start_node = random.randint(0, self.num_cities)  # start from a random node
        unvisited = self.cities[:]
        del unvisited[start_node]
        route = [cities[start_node]]
        while len(unvisited):
            index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
            route.append(nearest_city)
            del unvisited[index]
        current_cost = path_cost(route)
        self.progress.append(current_cost)
        return route, current_cost

    def accept_probability(self, candidate_fitness):
        return math.exp(-abs(candidate_fitness - self.cur_cost) / self.temperature)

    def accept(self, guess):
        guess_cost = path_cost(guess)
        if guess_cost < self.cur_cost:
            self.cur_cost, self.route = guess_cost, guess
            if guess_cost < self.best_fitness:
                self.best_fitness, self.route = guess_cost, guess
        else:
            if random.random() < self.accept_probability(guess_cost):
                self.cur_cost, self.route = guess_cost, guess

    def run(self):
        self.route, self.cur_cost = self.greedy_solution()
        print(f"\n{'=' * 80}")
        print(f"STARTING SIMULATED ANNEALING")
        print(f"{'=' * 80}")
        print(f"Number of cities: {self.num_cities}")
        print(f"Starting distance: {self.cur_cost:.2f} km")
        print(f"Starting temperature: {self.temperature:.4f}")
        print(f"Cooling rate (alpha): {self.alpha}")
        print(f"Stopping temperature: {self.stopping_temperature}")
        print(f"{'=' * 80}\n")

        # Initialize live plot
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        while self.temperature >= self.stopping_temperature and self.iteration < self.stopping_iter:
            guess = list(self.route)
            left_index = random.randint(2, self.num_cities - 1)
            right_index = random.randint(0, self.num_cities - left_index)
            guess[right_index: (right_index + left_index)] = reversed(guess[right_index: (right_index + left_index)])
            self.accept(guess)
            self.temperature *= self.alpha
            self.iteration += 1
            self.progress.append(self.cur_cost)

            # Update display every 100 iterations
            if self.iteration % 100 == 0:
                # LEFT PLOT: Distance progress
                ax1.clear()
                ax1.plot(self.progress, 'b-', linewidth=2, label='Distance')
                ax1.axhline(y=self.best_fitness, color='r', linestyle='--',
                            linewidth=2, label=f'Best: {self.best_fitness:.2f}')
                ax1.set_xlabel('Iterations', fontsize=12)
                ax1.set_ylabel('Distance (km)', fontsize=12)
                ax1.set_title('Distance Over Time', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # RIGHT PLOT: Temperature progress
                temps = []
                temp = self.temperature
                for i in range(len(self.progress)):
                    temps.append(temp)
                    temp /= self.alpha

                ax2.clear()
                ax2.plot(temps, 'r-', linewidth=2)
                ax2.set_xlabel('Iterations', fontsize=12)
                ax2.set_ylabel('Temperature', fontsize=12)
                ax2.set_title('Temperature Cooling', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')  # Log scale to see cooling better

                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)

                # Console output
                improvement = ((self.cur_cost - self.best_fitness) / self.best_fitness) * 100
                print(f"Iteration {self.iteration:7d} | "
                      f"Temp: {self.temperature:10.6f} | "
                      f"Current: {self.cur_cost:10.2f} km | "
                      f"Best: {self.best_fitness:10.2f} km | "
                      f"Gap: {improvement:6.2f}%")

        plt.ioff()
        print(f"\n{'=' * 80}")
        print(f"✅ OPTIMIZATION COMPLETE!")
        print(f"{'=' * 80}")
        print(f"Total iterations: {self.iteration}")
        print(f"Final temperature: {self.temperature:.8f}")
        print(f"Best fitness obtained: {self.best_fitness:.2f} km")
        print(f"{'=' * 80}\n")

    def visualize_routes(self):
        visualize_tsp('simulated annealing TSP', self.route)

    def plot_learning(self):
        fig = plt.figure(1)
        plt.plot([i for i in range(len(self.progress))], self.progress)
        plt.ylabel("Distance")
        plt.xlabel("Iterations")
        plt.show(block=False)


if __name__ == "__main__":
    # cities = write_cities_and_return_them(500)
    cities = read_cities(64)
    sa = SimAnneal(cities,
                   temperature=100,
                   alpha=0.9999,
                   stopping_temperature=0.001,
                   stopping_iter=10000)
    sa.run()
    sa.plot_learning()
    sa.visualize_routes()

