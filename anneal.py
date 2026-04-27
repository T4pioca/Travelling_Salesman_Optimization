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
        self.stopping_temperature = 1e-8 if stopping_temperature == -1 else stopping_temperature
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.route = None
        self.best_route = None
        self.best_fitness = float("Inf")
        self.progress = []
        self.cur_cost = None
        self.temperature_history = []

    def greedy_solution(self):
        start_node = random.randint(0, self.num_cities)
        unvisited = self.cities[:]
        del unvisited[start_node]
        route = [self.cities[start_node]]
        while len(unvisited):
            index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
            route.append(nearest_city)
            del unvisited[index]
        current_cost = path_cost(route)
        self.progress.append(current_cost)
        self.best_route = list(route)
        return route, current_cost

    def accept_probability(self, candidate_fitness):
        return math.exp(-abs(candidate_fitness - self.cur_cost) / self.temperature)

    def accept(self, guess):
        guess_cost = path_cost(guess)
        if guess_cost < self.cur_cost:
            self.cur_cost, self.route = guess_cost, guess
            if guess_cost < self.best_fitness:
                self.best_fitness = guess_cost
                self.best_route = list(guess)
        else:
            if random.random() < self.accept_probability(guess_cost):
                self.cur_cost, self.route = guess_cost, guess

    def run(self):
        self.route, self.cur_cost = self.greedy_solution()
        print(f"\n{'=' * 120}")
        print(f"STARTING SIMULATED ANNEALING - TRAVELLING SALESMAN PROBLEM")
        print(f"{'=' * 120}")
        print(f"Number of cities: {self.num_cities}")
        print(f"Starting distance: {self.cur_cost:.2f} km")
        print(f"Starting temperature: {self.temperature:.4f}")
        print(f"Cooling rate (alpha): {self.alpha}")
        print(f"Stopping temperature: {self.stopping_temperature}")
        print(f"Max iterations: {self.stopping_iter}")
        print(f"{'=' * 120}\n")

        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Simulated Annealing TSP - Live Optimization', fontsize=14, fontweight='bold')

        start_time = time.time()

        while self.temperature >= self.stopping_temperature and self.iteration < self.stopping_iter:
            guess = list(self.route)
            left_index = random.randint(2, self.num_cities - 1)
            right_index = random.randint(0, self.num_cities - left_index)
            guess[right_index: (right_index + left_index)] = reversed(guess[right_index: (right_index + left_index)])
            self.accept(guess)
            self.temperature *= self.alpha
            self.temperature_history.append(self.temperature)
            self.iteration += 1
            self.progress.append(self.cur_cost)

            if self.iteration % 50 == 0:
                elapsed = time.time() - start_time
                improvement = ((self.cur_cost - self.best_fitness) / self.best_fitness) * 100

                ax1.clear()
                ax1.plot(self.progress, 'b-', linewidth=2, label='Distance')
                ax1.axhline(y=self.best_fitness, color='r', linestyle='--',
                            linewidth=2, label=f'Best: {self.best_fitness:.2f}')
                ax1.set_xlabel('Iterations', fontsize=11, fontweight='bold')
                ax1.set_ylabel('Distance (km)', fontsize=11, fontweight='bold')
                ax1.set_title('Distance Progress', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend(fontsize=10)

                ax2.clear()
                ax2.plot(range(len(self.temperature_history)), self.temperature_history, 'r-', linewidth=2)
                ax2.set_xlabel('Iterations', fontsize=11, fontweight='bold')
                ax2.set_ylabel('Temperature', fontsize=11, fontweight='bold')
                ax2.set_title('Temperature Cooling Schedule', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)

                ax3.clear()

                x_coords = [city.x for city in self.cities]
                y_coords = [city.y for city in self.cities]
                ax3.scatter(x_coords, y_coords, c='red', s=100, zorder=5, edgecolors='darkred', linewidth=1.5)

                route_x = [city.x for city in self.best_route] + [self.best_route[0].x]
                route_y = [city.y for city in self.best_route] + [self.best_route[0].y]
                ax3.plot(route_x, route_y, 'g-', linewidth=2.5, alpha=0.8, label='Best Route', zorder=3)

                ax3.plot(self.best_route[0].x, self.best_route[0].y,
                         'bs', markersize=12, zorder=10, markeredgecolor='darkblue',
                         markeredgewidth=2, label='Start')

                ax3.set_title('Best Route Found', fontsize=12, fontweight='bold')
                ax3.set_xlabel('X Coordinate', fontsize=11, fontweight='bold')
                ax3.set_ylabel('Y Coordinate', fontsize=11, fontweight='bold')
                ax3.legend(fontsize=10, loc='upper right')
                ax3.grid(True, alpha=0.2)
                ax3.set_aspect('equal', adjustable='box')

                info_text = (f"Iteration: {self.iteration}\n"
                             f"Best Distance: {self.best_fitness:.2f} km\n"
                             f"Gap: {improvement:.2f}%\n"
                             f"Time: {elapsed:.2f}s")
                ax3.text(0.02, 0.98, info_text,
                         transform=ax3.transAxes, fontsize=10, fontweight='bold',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='black', linewidth=1.5))

                fig.suptitle(f'Simulated Annealing TSP - Live Optimization | Time: {elapsed:.2f}s',
                             fontsize=13, fontweight='bold')

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.draw()
                plt.pause(0.01)

                print(f"Iteration {self.iteration:7d} | "
                      f"Temp: {self.temperature:10.6f} | "
                      f"Current: {self.cur_cost:10.2f} km | "
                      f"Best: {self.best_fitness:10.2f} km | "
                      f"Gap: {improvement:6.2f}% | "
                      f"Time: {elapsed:8.2f}s")

        total_time = time.time() - start_time
        plt.ioff()

        print(f"\n{'=' * 120}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'=' * 120}")
        print(f"Total iterations: {self.iteration}")
        print(f"Final temperature: {self.temperature:.8f}")
        print(f"Best fitness obtained: {self.best_fitness:.2f} km")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"{'=' * 120}\n")

    def visualize_routes(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        x_coords = [city.x for city in self.cities]
        y_coords = [city.y for city in self.cities]
        ax.scatter(x_coords, y_coords, c='red', s=150, zorder=5, edgecolors='darkred', linewidth=2, label='Cities')

        route_x = [city.x for city in self.best_route] + [self.best_route[0].x]
        route_y = [city.y for city in self.best_route] + [self.best_route[0].y]
        ax.plot(route_x, route_y, 'g-', linewidth=3, alpha=0.8, label='Best Route', zorder=3)
        ax.plot(self.best_route[0].x, self.best_route[0].y, 'bs', markersize=15,
                label='Start/End', zorder=10, markeredgecolor='darkblue', markeredgewidth=2)

        ax.set_xlabel('X Coordinate', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Coordinate', fontsize=12, fontweight='bold')
        ax.set_title(f'Simulated Annealing TSP - Final Solution\nBest Distance: {self.best_fitness:.2f} km',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        plt.show(block=False)

    def plot_learning(self):
        fig = plt.figure(figsize=(12, 6))
        plt.plot([i for i in range(len(self.progress))], self.progress, 'b-', linewidth=2.5, label='Distance')
        plt.axhline(y=self.best_fitness, color='r', linestyle='--', linewidth=2.5,
                    label=f'Best: {self.best_fitness:.2f} km')
        plt.ylabel("Distance (km)", fontsize=12, fontweight='bold')
        plt.xlabel("Iterations", fontsize=12, fontweight='bold')
        plt.title("Simulated Annealing - Distance Optimization Progress", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)

        plt.tight_layout()
        plt.show(block=False)


if __name__ == "__main__":
    print("\n" + "=" * 120)
    print("TRAVELLING SALESMAN OPTIMIZATION - SIMULATED ANNEALING")
    print("=" * 120 + "\n")

    cities = read_cities(20)
    sa = SimAnneal(cities,
                   temperature=150,
                   alpha=0.9995,
                   stopping_temperature=0.001,
                   stopping_iter=20000)
    sa.run()
    sa.plot_learning()
    sa.visualize_routes()

    plt.show(block=True)