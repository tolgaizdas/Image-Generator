import numpy as np

from genetic import GeneticAlgorithm
from plotter import ImagePlotter


def main():
    # Load image
    img_name = 'flower'
    with open(f'images/{img_name}.img', 'r') as f:
        image = f.readlines()
        f.close()

    # Parse image
    x, y = image.pop(0).replace('(', '').replace(')', '').replace(',', '').split()
    x, y = int(x), int(y)
    image = np.array(image, dtype=int)

    # Create genetic algorithm instance
    gen = GeneticAlgorithm(pop_size=200, max_gen=1000, mut_rate=0.01, fitness_function='euclidean_distance')
    
    # Fit genetic algorithm
    gen.fit(image, x, y)

    # Get best prediction
    latest = gen.latest_prediction
    latest_fitness = gen.latest_fitness

    best = gen.best_prediction
    best_fitness = gen.best_fitness

    # Draw results
    WIDTH, HEIGHT = 900, 300
    LINE_LENGTH, LINE_THICKNESS = 15, 2

    plt = ImagePlotter(WIDTH, HEIGHT, plot_name='target - best - latest')

    plt.draw_img(image, x, y, line_length=LINE_LENGTH, line_thickness=LINE_THICKNESS)
    plt.draw_img(best, x + WIDTH // 3, y, line_length=LINE_LENGTH, line_thickness=LINE_THICKNESS)
    plt.draw_img(latest, x +  2 * WIDTH // 3, y, line_length=LINE_LENGTH, line_thickness=LINE_THICKNESS)

    print(f'Latest: {latest}\nLatest Fitness: {latest_fitness}\n')
    print(f'Best: {best}\nBest Fitness: {best_fitness}\n')

    plt.display(keep_open=True)


if __name__ == '__main__':
    main()
