import random
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings

plt.rc('font', size=22)

class SOR:
    def __init__(self, entities, max_product_types, max_product_complexity, percolation_coefficient, generation_coefficient, degradation_coefficient, replication_rate, product_interference):
        self.n = entities
        self.p = max_product_types
        self.p_comp = max_product_complexity
        self.rho = percolation_coefficient
        self.lamb = generation_coefficient
        self.fi = degradation_coefficient
        self.replication = replication_rate
        self.interference = product_interference
        self.perc_degr_prob = percolation_coefficient/(percolation_coefficient+degradation_coefficient)

        self.map = [[[0 for x in range(0, self.p_comp)] for y in range(0, self.p)]for s in range(0, self.n)]
        self.time = 0
        self.all_products = 0
        self.entity_product_types = [0 for s in range(0, self.n)]
        self.entity_product_counts = [0 for s in range(0, self.n)]
        self.entity_bool = [[0 for y in range(0, self.p)] for s in range(0, self.n)]
        self.entity_int = [[0 for y in range(0, self.p)] for s in range(0, self.n)]

        self.generation_time = np.random.exponential(1/(self.lamb * self.n), 1)[0]
        self.perc_degr_time = 0

        self.graph = [[0 for y in range(0, self.n)] for s in range(0, self.n)]
        self.generate_graph(entities)

        self.next_checkpoint = 0
        self.variation = [0]
        self.number_of_product_types = [0]
        self.complexity = [0]
        self.average_length = [0]

        self.time_manipulation_count = 0

    def generate_graph(self, n):
        '''Create a well-connected directed graph. Each graph will have a cycle that contains all nodes'''
        a = 0
        b = 1
        self.graph[a][b] = 1
        self.graph[n-1][0] = 1
        for i in range(0, n-2):
            a += 1
            b += 1
            self.graph[a][b] = 1

        reps = random.randint(2*n, 5*n)
        for r in range(0, reps):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            self.graph[a][b] = 1

    def product_appears(self, entity, product, version):
        print("product appears")
        self.map[entity][product][version] = 1
        self.all_products += 1
        self.entity_product_counts[entity] += 1
        self.entity_int[entity][product] += 1
        if self.entity_bool[entity][product] == 0:
            self.entity_product_types[entity] += 1
        self.entity_bool[entity][product] = 1

    def product_disappears(self, entity, product, version):
        print("product disappears")
        self.map[entity][product][version] = 0
        self.all_products -= 1
        self.entity_product_counts[entity] -= 1
        self.entity_int[entity][product] -= 1
        if self.map[entity][product] == [0] * self.p_comp:
            self.entity_product_types[entity] -= 1
            self.entity_bool[entity][product] = 0

    def check(self):
        allcounts = 0
        for n in range(0, self.n):
            nsum = np.sum(self.map[n])
            if nsum != self.entity_product_counts[n]:
                warnings.warn("entity_product_counts is incorrect. true value is " + str(nsum) + " but entity_product_counts says " + str(self.entity_product_counts))
            for p in range(0, self.p):
                sum = np.sum(self.map[n][p])
                if bool(sum) != bool(self.entity_bool[n][p]):
                    warnings.warn("entity_bool is incorrect in entity " + str(n) + "product " + str(p) + "the real value is " + str(bool(sum)))
                if sum != self.entity_int[n][p]:
                    warnings.warn("entity_int is incorrect")
                for v in range(0, self.p_comp):
                    if self.map[n][p][v] == 1:
                        allcounts += 1
        if allcounts != self.all_products:
            warnings.warn("all_products is incorrect, there are a total of " + str(allcounts) + " products, while all_products thinks there are " + str(self.all_products))

    def monitor(self):
        self.next_checkpoint += 1
        var = 0
        check_for = []
        print("var")
        for x in range(0, self.n):  # for every entity
            if check_for:  # compare to previous types
                switches = []
                for r in range(0, len(check_for)):  # for every previous type
                    if self.entity_product_counts[x] != self.entity_product_counts[check_for[r]]:
                        continue
                    else:
                        for p in range(0, self.p):
                            if self.entity_int[x][p] != self.entity_int[check_for[r]][p]:
                                break
                        else:
                            for i in range(0, self.p):
                                if self.entity_bool[x][i] != 0:
                                    if self.map[x][i] != self.map[check_for[r]][i]:
                                        break
                            else:
                                break
                            continue
                        continue
                else:
                    check_for.append(x)  # append the new type to types
                    var += 1
            else:
                check_for.append(x)
                var += 1
        self.variation.append(var)

        comp = 0
        for i in range(0, self.n):
            for j in range(0, self.p):
                if self.entity_bool[i][j]:
                    if [i for i, x in enumerate(self.map[i][j]) if x][-1] > comp:
                        comp = [i for i, x in enumerate(self.map[i][j]) if x][-1]
        self.complexity.append(comp)

        lens = np.sum(self.entity_product_counts)
        self.average_length.append(lens / self.n)

    def repeat(self):
        self.next_checkpoint += 1
        self.variation.append(self.variation[-1])
        self.complexity.append(self.complexity[-1])
        self.average_length.append(self.average_length[-1])

    def generation(self):
        ''' Forward the chosen entity's composition, by either adding a new product lineage,
        or introducing a new version of an existing product type'''
        print("gen")
        self.time = self.generation_time
        self.generation_time = np.random.exponential(1/(self.lamb * self.n), 1)[0] + self.time
        entity = random.randint(0, self.n-1)
        decide = random.randint(0, self.entity_product_types[entity])  # randomly decide what happens
        if decide == self.entity_product_types[entity]:  # new product lineage is added
            zeros = [i for i, x in enumerate(self.entity_bool[entity]) if not x] # list of products that can newly appear
            product = zeros[random.randint(0, len(zeros)-1)] # pick one

            self.product_appears(entity, product, 0)

            if self.interference != 0:
                if random.random() < self.interference:
                    ones = [i for i, x in enumerate(self.entity_bool[entity]) if x]
                    ones.remove(product) # it cannot remove versions from its own product type
                    if len(ones) != 0:
                        prod_to_remove = ones[random.randint(0, len(ones)-1)]
                        versions = self.map[entity][prod_to_remove]
                        version = versions[random.randint(0, len(versions)-1)]
                        self.product_disappears(entity, prod_to_remove, version)

        else:  # existing product is further transformed
            ones = [i for i, x in enumerate(self.entity_bool[entity]) if x]
            product = ones[random.randint(0, len(ones)-1)]
            last_version = [index for index, item in enumerate(self.map[entity][product]) if item != 0][-1]

            self.product_appears(entity, product, last_version+1)

            if self.interference != 0:
                if random.random() < self.interference:
                    ones = [i for i, x in enumerate(self.entity_bool[entity]) if x]
                    ones.remove(product)
                    if len(ones) != 0:
                        prod_to_remove = ones[random.randint(0, len(ones) - 1)]
                        versions = self.map[entity][prod_to_remove]
                        version = versions[random.randint(0, len(versions) - 1)]
                        self.product_disappears(entity, prod_to_remove, version)


    def percolation(self):
        '''Forward a random product from a random entity to one of it's neighbours.'''
        print("perc")
        self.time = self.perc_degr_time
        entity = random.choices(list(range(0, self.n)), weights=self.entity_product_counts, k=1)[0]
        product = random.choices(list(range(0, self.p)), weights=self.entity_int[entity], k = 1)[0]
        v_ones = [i for i, x in enumerate(self.map[entity][product]) if x]
        version = v_ones[random.randint(0, len(v_ones) - 1)]

        targets = []
        for i in range(0, self.n):
            if self.graph[entity][i] == 1:  # there has to be an edge from x to k
                targets.append(i)
        target = random.choice(targets)
        if self.map[target][product][version] != 1:
            self.product_appears(target, product, version)

        if self.replication != 1:  # if there is any chance of percolation loss
            if random.random() > self.replication:
                self.product_disappears(entity, product, version)

    def degradation(self):
        self.time = self.perc_degr_time
        entity = random.choices(list(range(0, self.n)), weights=self.entity_product_counts, k=1)[0]
        product = random.choices(list(range(0, self.p)), weights=self.entity_int[entity], k=1)[0]
        v_ones = [i for i, x in enumerate(self.map[entity][product]) if x]
        version = v_ones[random.randint(0, len(v_ones) - 1)]

        self.product_disappears(entity, product, version)

    def time_development(self):
        print(self.time)
        # self.check()
        difference = self.time - self.next_checkpoint  # calculate time difference from next checkpoint
        if difference > 1:  # check if we've skipped any integers
            for y in range(0, math.floor(difference)):
                self.repeat()  # for every integer skipped, repeat the last snapshot
            self.monitor()  # take snapshot
        elif difference > 0:  # if we've passed the next checkpoint, but by less than 1
            self.monitor()  # take snapshot
        event_list = [self.generation_time, self.perc_degr_time]
        if not self.all_products:
            event_list.pop(1)
        else:
            self.perc_degr_time = np.random.exponential(1 / ((self.rho + self.fi) * self.all_products), 1)[0] + self.time
        event = event_list.index(min(event_list))
        if event == 0:
            self.generation()
        else:
            rand = random.random()
            if rand < self.perc_degr_prob:
                self.percolation()
            else:
                self.degradation()

runtime = 10000
#entities, max_product_types, max_product_complexity, percolation_coefficient, generation_coefficient, degradation_coefficient, replication_rate, product_interference

values_list = [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005,   0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

avr_end_var = [0] * len(values_list)
avr_end_comp = [0] * len(values_list)
avr_end_avr_len = [0] * len(values_list)

for run in range(0, 10):
    end_var = []
    end_comp = []
    end_avr_len = []
    for value in values_list:
        A = SOR(100, 100, 200, 0.1, 0.0001, value, 1, 0)
        while A.time < runtime:
            A.time_development()

        end_var.append(A.variation[-1])
        end_comp.append(A.complexity[-1])
        end_avr_len.append(A.average_length[-1])

        fig = plt.figure(facecolor="white", figsize=(16, 16))
        x = np.linspace(0, A.next_checkpoint+1, A.next_checkpoint+1)

        plt.plot(x, A.variation, label = "variation", linewidth=4)
        plt.plot(x, A.complexity, label = "complexity", linewidth=4)
        plt.plot(x, A.average_length, label = "average length", linewidth=4)
        plt.legend()
        title = str("p: " + str(A.rho) + "  lambda: " + str(A.lamb) + " degr: " + str(A.fi))
        plt.title(title)
        plt.savefig('/home/jakab/Documents/graphs/SOR/compact_version2/degradation/run' + str(run) + '/' + 'degradation' + str(value) + '.jpg')


    for n in range(0, len(values_list)):
        avr_end_var[n] = avr_end_var[n] + end_var[n]
        avr_end_comp[n] = avr_end_comp[n] + end_comp[n]
        avr_end_avr_len[n] = avr_end_avr_len[n] + end_avr_len[n]

    fig = plt.figure(facecolor="white", figsize=(16, 16))

    plt.gca().set_xscale('log')
    plt.plot(values_list, end_var, label = "variation", linewidth=4)
    plt.plot(values_list, end_comp, label = "complexity", linewidth=4)
    plt.plot(values_list, end_avr_len, label = "average length", linewidth=4)
    title = str("p: " + str(A.rho) + "  lambda: " + str(A.lamb) + 'runtime: ' + str(runtime))
    plt.title(title)
    plt.xlabel("degradation_rate")

    plt.savefig('/home/jakab/Documents/graphs/SOR/compact_version2/degradation/run' + str(run) + '/' + 'degradation_rate_summary' + '.jpg')

fig = plt.figure(facecolor="white", figsize=(16, 16))

plt.gca().set_xscale('log')
plt.plot(values_list, avr_end_var, label = "variation", linewidth=4)
plt.plot(values_list, avr_end_comp, label = "complexity", linewidth=4)
plt.plot(values_list, avr_end_avr_len, label = "average length", linewidth=4)
title = str("p: " + str(A.rho) + "  lambda: " + str(A.lamb) + 'runtime: ' + str(runtime))
plt.title(title)
plt.xlabel("degradation probability")



plt.savefig('/home/jakab/Documents/graphs/SOR/compact_version2/degradation/' + 'degradation_rate_averages' + '.jpg')
