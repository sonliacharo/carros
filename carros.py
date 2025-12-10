import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pygad
import math
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML

rc('animation', html='jshtml')

MAP_SIZE = 100
track_map = np.zeros((MAP_SIZE, MAP_SIZE))
track_map[0, :] = 1
track_map[-1, :] = 1
track_map[:, 0] = 1
track_map[:, -1] = 1
track_map[30:70, 30:70] = 1

class Car:
    def __init__(self):
        self.x = 15.0
        self.y = 15.0
        self.angle = 0.0
        self.alive = True
        self.history = [] # Para guardar o caminho (x, y)
        self.history.append((self.x, self.y))

    def get_sensors(self):
        angles = [-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2]
        readings = []
        for rel_angle in angles:
            dist = 0
            for d in range(1, 101, 5): # Passo maior (5) para ficar mais rápido
                check_x = int(self.x + d * math.cos(self.angle + rel_angle))
                check_y = int(self.y + d * math.sin(self.angle + rel_angle))
                if (check_x < 0 or check_x >= MAP_SIZE or
                    check_y < 0 or check_y >= MAP_SIZE or
                    track_map[check_x, check_y] == 1):
                    dist = d
                    break
                dist = 100
            readings.append(dist)
        return readings

    def move(self, steering_angle):
        if not self.alive: return
        rad_steer = math.radians(steering_angle)
        self.angle += rad_steer
        self.x += math.cos(self.angle)
        self.y += math.sin(self.angle)

        # Salva posição
        self.history.append((self.x, self.y))

        if (self.x < 0 or self.x >= MAP_SIZE or
            self.y < 0 or self.y >= MAP_SIZE or
            track_map[int(self.x), int(self.y)] == 1):
            self.alive = False

def create_fuzzy_system():
    s_left = ctrl.Antecedent(np.arange(0, 101, 1), 's_left')
    s_front = ctrl.Antecedent(np.arange(0, 101, 1), 's_front')
    s_right = ctrl.Antecedent(np.arange(0, 101, 1), 's_right')
    direction = ctrl.Consequent(np.arange(-45, 46, 1), 'direction')

    names_in = ['mp', 'p', 'm', 'l', 'ml']
    for s in [s_left, s_front, s_right]: s.automf(names=names_in)
    direction.automf(names=['me', 'e', 'c', 'd', 'md'])
    return s_left, s_front, s_right, direction

def fitness_func(ga_instance, solution, solution_idx):
    s_left, s_front, s_right, direction = create_fuzzy_system()
    terms_in = ['mp', 'p', 'm', 'l', 'ml']
    terms_out = ['me', 'e', 'c', 'd', 'md']
    rule_list = []

    cnt = 0
    for t1 in terms_in:
        for t2 in terms_in:
            for t3 in terms_in:
                 if cnt >= len(solution): break
                 output_idx = int(solution[cnt])
                 rule = ctrl.Rule(s_left[t1] & s_front[t2] & s_right[t3], direction[terms_out[output_idx]])
                 rule_list.append(rule)
                 cnt += 1

    if not rule_list: return 0
    try:
        driving_ctrl = ctrl.ControlSystem(rule_list)
        driver = ctrl.ControlSystemSimulation(driving_ctrl)
    except: return 0

    car = Car()
    steps = 0
    while car.alive and steps < 200:
        sensors = car.get_sensors()
        driver.input['s_left'] = sensors[1]
        driver.input['s_front'] = sensors[2]
        driver.input['s_right'] = sensors[3]
        try:
            driver.compute()
            steer = driver.output['direction']
        except: steer = 0
        car.move(steer)
        steps += 1

    return steps

ga_instance = pygad.GA(num_generations=10,
                       num_parents_mating=2,
                       fitness_func=fitness_func,
                       sol_per_pop=5,
                       num_genes=125,
                       gene_type=int,
                       init_range_low=0,
                       init_range_high=4,
                       mutation_percent_genes=10)

print("Evoluindo... Aguarde.")
ga_instance.run()

print("1. Evolução da Aptidão (Fitness):")
ga_instance.plot_fitness()

print("2. Gerando Animação do Melhor Indivíduo...")
solution, solution_fitness, _ = ga_instance.best_solution()

best_car = Car()
s_left, s_front, s_right, direction = create_fuzzy_system()
terms_in = ['mp', 'p', 'm', 'l', 'ml']
terms_out = ['me', 'e', 'c', 'd', 'md']
rule_list = []
cnt = 0
for t1 in terms_in:
    for t2 in terms_in:
        for t3 in terms_in:
            if cnt >= len(solution): break
            output_idx = int(solution[cnt])
            rule = ctrl.Rule(s_left[t1] & s_front[t2] & s_right[t3], direction[terms_out[output_idx]])
            rule_list.append(rule)
            cnt += 1
driver_ctrl = ctrl.ControlSystem(rule_list)
sim = ctrl.ControlSystemSimulation(driver_ctrl)

for _ in range(300):
    if not best_car.alive: break
    sens = best_car.get_sensors()
    sim.input['s_left'] = sens[1]
    sim.input['s_front'] = sens[2]
    sim.input['s_right'] = sens[3]
    try:
        sim.compute()
        st = sim.output['direction']
    except: st = 0
    best_car.move(st)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, MAP_SIZE)
ax.set_ylim(0, MAP_SIZE)
ax.imshow(track_map.T, cmap='Greys', origin='lower', extent=[0, MAP_SIZE, 0, MAP_SIZE])
line, = ax.plot([], [], 'r-', lw=2, label='Trajeto')
point, = ax.plot([], [], 'bo', ms=5, label='Carro')
ax.legend()

def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

def animate(i):
    if i < len(best_car.history):
        x_data = [p[0] for p in best_car.history[:i+1]]
        y_data = [p[1] for p in best_car.history[:i+1]]
        line.set_data(x_data, y_data)
        point.set_data([best_car.history[i][0]], [best_car.history[i][1]])
    return line, point

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(best_car.history), interval=50, blit=True)
HTML(anim.to_jshtml())