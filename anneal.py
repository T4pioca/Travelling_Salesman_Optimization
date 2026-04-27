import math
import random
import time
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
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Simulated Annealing - Live Optimization', fontsize=14, fontweight='bold')

        start_time = time.time()

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
                elapsed = time.time() - start_time
                time_per_iter = elapsed / self.iteration if self.iteration > 0 else 0
                improvement = ((self.cur_cost - self.best_fitness) / self.best_fitness) * 100

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

                # MIDDLE PLOT: Temperature progress
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

                # RIGHT PLOT: Live animated path
                ax3.clear()
                route_x = [city.x for city in self.route] + [self.route[0].x]
                route_y = [city.y for city in self.route] + [self.route[0].y]
                ax3.plot(route_x, route_y, 'g-', linewidth=1.5, alpha=0.8)
                ax3.plot([city.x for city in self.route],
                         [city.y for city in self.route],
                         'ro', markersize=5, zorder=3)
                # Mark starting city with a blue square
                ax3.plot(self.route[0].x, self.route[0].y,
                         'bs', markersize=10, zorder=4, label='Start')
                ax3.set_title('Live Route', fontsize=14, fontweight='bold')
                ax3.set_xlabel('X', fontsize=12)
                ax3.set_ylabel('Y', fontsize=12)
                ax3.legend(fontsize=9)
                ax3.text(0.02, 0.98,
                         f"Iter: {self.iteration}\nBest: {self.best_fitness:.1f} km\n"
                         f"Gap: {improvement:.2f}%\nElapsed: {elapsed:.1f}s",
                         transform=ax3.transAxes, fontsize=8,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)

                # Console output
                print(f"Iteration {self.iteration:7d} | "
                      f"Temp: {self.temperature:10.6f} | "
                      f"Current: {self.cur_cost:10.2f} km | "
                      f"Best: {self.best_fitness:10.2f} km | "
                      f"Gap: {improvement:6.2f}% | "
                      f"Elapsed: {elapsed:.2f}s"
                      + (f" | Iter/s: {1/time_per_iter:.1f}" if time_per_iter > 0 else ""))

        total_time = time.time() - start_time
        plt.ioff()
        print(f"\n{'=' * 80}")
        print(f"✅ OPTIMIZATION COMPLETE!")
        print(f"{'=' * 80}")
        print(f"Total iterations: {self.iteration}")
        print(f"Final temperature: {self.temperature:.8f}")
        print(f"Best fitness obtained: {self.best_fitness:.2f} km")
        print(f"Total time: {total_time:.2f}s")
        print(f"Time per iteration: {total_time / self.iteration * 1000:.4f} ms" if self.iteration > 0 else "Time per iteration: N/A")
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

