class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.header_written = False

    def log(self, generation, pop_size, max_gen, mut_rate, fitness_function, fitness_value):
        if not self.header_written:
            with open(self.log_file, 'w') as f:
                f.write('generation,pop_size,max_gen,mut_rate,fitness_function,fitness_value\n')
            self.header_written = True
            
        with open(self.log_file, 'a') as f:
            f.write(f'{generation},{pop_size},{max_gen},{mut_rate},{fitness_function},{fitness_value}\n')
        