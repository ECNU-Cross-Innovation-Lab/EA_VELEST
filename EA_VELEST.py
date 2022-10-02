import os
from numpy import loadtxt, array, random, hstack, arange
from initial_mpl import init_plotting_isi
import matplotlib.pyplot as plt
import mpl_toolkits

mpl_toolkits.__path__.append('/usr/lib/python2.7/dist-packages/mpl_toolkits/')
from initial_mpl import init_plotting_isi
from matplotlib.colors import LightSource
from matplotlib.cbook import get_sample_data
from matplotlib.colors import LinearSegmentedColormap
import time
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker, gridspec
import matplotlib.pyplot as plt

inp = loadtxt('syntvel.dat', dtype=str, comments='#', delimiter=':')
z = inp[16][1].strip()


def read_velestmodel():
    mods = os.listdir('/home/yin/zhangxiaoyue/tmp/out/result/mod')
    vel, vel2, elev = [], [], []
    for mod in mods:
        f = open('/home/yin/zhangxiaoyue/tmp/out/result/mod/' + mod, 'r')
        file_data = []

        line = f.readline()  #
        while line:
            file_data.append(line)  # 往列表后面插入
            line = f.readline()
        f.close

        for i in range(len(file_data)):
            if i == 1:
                layer = int(file_data[i].split(' ')[1])
        for i in range(len(file_data)):
            if i > 1 and i < layer + 2:
                vel.append(float(file_data[i].split(' ')[1]))
                elev.append(float(file_data[i].split(' ')[-5]))
            elif i > layer + 2 and i < 2 * layer + 3:
                vel2.append(float(file_data[i].split(' ')[1]))
    vel = array(vel)
    vel2 = array(vel2)
    elev = array(elev)
    vel = vel.reshape(500, 12)
    vel2 = vel2.reshape(500, 12)
    elev = elev.reshape(500, 12)
    return vel, vel2, elev


def write_cmn(ind_generation,ind):
    file = os.listdir('/home/yin/zhangxiaoyue/tmp/out/synthvel')
    f = open('/home/yin/zhangxiaoyue/velest.cmncopy', 'r')
    line = f.readline()  #
    file_data = []
    while line:
        file_data.append(line)
        line = f.readline()
    f.close

    f_out = open('/home/yin/zhangxiaoyue/velest.cmn', 'w')
    for l, k in enumerate(file_data):
        if l == 40:
            out_cmn = '/home/yin/zhangxiaoyue/tmp/out/synthvel/model_' + str(ind_generation+1) +"_"+str(ind+1) + '.mod' + '\n'
            f_out.write(out_cmn)
        else:
            f_out.write(k)
    f_out.flush()
    f_out.close()
    f.close()


def re_velest(ind_generation, ind):
    if ind_generation == 100:
        min_rms = 100
    else:
        write_cmn(ind_generation,ind)
        os.system('velest')
        os.replace('velout.mod', '/home/yin/zhangxiaoyue/tmp/out/result/mod/velout.mod_' + str(ind_generation+1) +"_"+ str(ind+1))
        os.replace('sta.COR', '/home/yin/zhangxiaoyue/tmp/out/result/cor/sta.COR_' + str(ind_generation+1) + "_"+ str(ind))
        f = open('main.OUT', 'r')
        # print(cc)
        file_data = []
        line = f.readline()
        while line:
            file_data.append(line)
            line = f.readline()
        f.close
        rms = []
        for i, s in enumerate(file_data):
            if 'RMS RESIDUAL=' in s:
                z = s.split()
                if z[-2] == 'RESIDUAL=':
                    rms.append(float(z[-1]))
        min_rms = min(rms)
    return min_rms

# set
lb = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
ub = [7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5]
DNA_SIZE = 12
POP_SIZE = 10
CROSS_RATE = 0.5
MUTATION_RATE = 0.05
N_GENERATIONS = 50


def get_fitness(ind_generation, ind):
    return re_velest(ind_generation, ind)

def select(pop, fitness):
    f = 1 / (fitness + 1e-3)
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=f / f.sum())
    return pop[idx]

def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool_)
        parent[cross_points] = pop[i_, cross_points]
    return parent

def mutate(child, ub, lb):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = lb[point] + (ub[point] - lb[point]) * np.random.rand()
    return child


def read_cor():
    mods = os.listdir('/home/yin/zhangxiaoyue/tmp/out/result/cor')
    sta, cor_p, cor_s = [], [], []
    for cc in mods:
        f = open('/home/yin/zhangxiaoyue/tmp/out/result/cor/' + cc, 'r')
        # print(cc)
        file_data = []
        line = f.readline()  #
        while line:
            file_data.append(line)  # 往列表后面插入
            line = f.readline()
        f.close
        for i, k in enumerate(file_data):
            if i > 0 and i < 19:
                sta.append(k[0:4])
                # print(k[34:39])
                cor_p.append(k[34:39])
                cor_s.append(k[41:46])
        cor = array([sta, cor_p, cor_s])

    return cor


# pvel,svel,pdep      = read_velestmodel()

pvel = [3.5, 4.0, 4.5, 5.5, 5.8, 6.0, 6.05, 6.3, 6.7, 6.72, 6.75, 7.0]
svel = [round(v / 1.75, 2) for v in pvel]
pdep = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 40]

pdmp = [1, 0.2] * 6
sdmp = [1, 0.2] * 6


class main():
    """
    Description:
    A class for running velest in automatic mode (synthetic
    model generation, control file prepration, analysing and
    plot results).

    """

    def __init__(self):

        ss = 'yxx'
        # input_dic  = read_par()

    def mk_syn_model(inp_syntvel='syntvel.dat'):
        vel_mode = inp[0][1].strip()
        dep_mode = inp[1][1].strip()
        vel_min = float(inp[2][1])
        vel_max = float(inp[3][1])
        dep_min = float(inp[4][1])
        dep_max = float(inp[5][1])
        vel_dev = float(inp[6][1])
        dep_dev = float(inp[7][1])
        nmod = int(inp[8][1].strip())
        vpvs = float(inp[9][1])
        ors = float(inp[10][1])
        grd = int(inp[11][1].strip())
        bad_evnt_herr = int(inp[12][1].strip())
        bad_evnt_zerr = int(inp[13][1].strip())
        bad_evnt_rms = float(inp[14][1].strip())
        res_max_rw = float(inp[15][1].strip())
        velmod_name = inp[16][1].strip()

        from os import path, makedirs
        from shutil import copy, rmtree, move

        if not path.exists(path.join('tmp', 'out', 'synthvel')):

            makedirs(path.join('tmp', 'out', 'synthvel'))

        else:

            rmtree(path.join('tmp', 'out', 'synthvel'))
            makedirs(path.join('tmp', 'out', 'synthvel'))

        output = path.join('tmp', 'out', 'synthvel')
        orig_velmod = array([pvel, pdep, pdmp]).T


        # --step1
        pop_layers = np.zeros((POP_SIZE, DNA_SIZE))
        for i in range(12):
            pop_layers[:, i] = lb[i] + (ub[i] - lb[i]) * np.random.rand(POP_SIZE, )

        # --step2
        fitness = np.zeros([POP_SIZE, ])

        for i in range(POP_SIZE):
            pop_list = list(pop_layers[i])
            fitness[i] = get_fitness(ind_generation=100, ind=i)
        result = []
        best_fitness = np.inf
        trace = np.zeros([N_GENERATIONS, ])

        # -- step3
        for each_generation in range(N_GENERATIONS):

            pop = select(pop_layers, fitness)
            pop_copy = pop.copy()
            for parent in pop:
                child = crossover(parent, pop_copy)
                child = mutate(child, ub, lb)
                parent = child
            for i in range(POP_SIZE):
                pop[i] = sorted(pop[i],reverse=False)


            # --step4
            for i in range(POP_SIZE):  # chr_i

                vel = list(pop[i])

                # generate random dep with no gradient

                if grd == 0:

                    if dep_mode == "u":

                        dep = array([random.uniform(i - dep_dev * (1.0 / j), i + dep_dev * (1.0 / j), nmod) for i, j in
                                     zip(pdep, pdmp)]).T

                    elif dep_mode == "n":

                        dep = array([random.normal(i, dep_dev * (1.0 / j), nmod) for i, j in zip(pdep, pdmp)]).T

                # generate random dep with positive gradient

                elif grd == 1:

                    dep = []
                    if dep_mode == "u":
                            dep.append(
                                sorted(random.uniform(i - dep_dev * (1.0 / j), i + dep_dev * (1.0 / j)) for i, j in
                                       zip(orig_velmod[:, 1], pdmp)))
                    elif dep_mode == "n":
                            dep.append(
                                sorted(random.normal(i, dep_dev * (1.0 / j)) for i, j in zip(orig_velmod[:, 1], pdmp)))
                    dep = array(dep)

                dep_org = [i[1] for i in orig_velmod]  # 1*12
                vel_org = [i[0] for i in orig_velmod]  # 1*12
                dep_org[0] = dep_min  # -3.0

                for d in range(len(dep)):
                    dep[d][0] = dep_min

                nl = len(dep[0])  # number of layers for each random model   #nl : 12


                fid = open(path.join(output, "model_" + str(each_generation + 1) + "_" + str(i + 1) + ".mod"), 'w')  # new
                fid.write('%s%d%s%d%s\n' % ("Model: model_", int(each_generation + 1),"_", i + 1, ".mod"))
                fid.write(
                    '%s%d %s\n' % (" ", len(dep[0]), "         vel,depth,vdamp,phase(f5.2,5x,f7.2,2x,f7.3,3x,a1)"))

                for j in range(len(dep[0])):

                    if j == 0:

                        fid.write('%5.2f %11.2f %8.3f %26s\n' % (
                             vel[j],dep[0][j], pdmp[j], "P-VELOCITY MODEL"))

                    else:

                        fid.write('%5.2f %11.2f %8.3f\n' % (vel[j], dep[0][j], 1.0))  # dep change!!

                fid.write('%s%s\n' % (" ", len(dep[0])))

                for j in range(len(dep[0])):

                    if j == 0:

                        fid.write(
                            '%5.2f %11.2f %8.3f %26s\n' % (
                                vel[j] / vpvs, dep[0][j], pdmp[j],
                                "S-VELOCITY MODEL"))

                    else:

                        fid.write(
                            '%5.2f %11.2f %8.3f\n' % (vel[j] / vpvs, dep[0][j], 1.0))
                fid.close()

                fitness[i] = get_fitness(ind_generation=each_generation, ind=i)

            # update

            loc_pop = pop[np.argmin(fitness)]
            for i in range(len(fitness)):
                if loc_pop == fitness[i]:
                    print('The current optimal result is: velout.mod_',each_generation + 1,'_',str(i+1),sep='')
                else:
                    pass


            loc_fitness = np.min(fitness)

            if loc_fitness < best_fitness:
                best_fitness = loc_fitness.copy()
                best_pop = loc_pop.copy()
            trace[each_generation] = best_fitness
            # The original parameter combination selected is not the best parameters being chosen,
            # we should find the corresponding file 'velout.mod_each_generation_i'
            print(each_generation + 1, 'fitness:', best_fitness, ' The original parameter combination selected is:',
                  [round(best_pop[i], 1) for i in range(DNA_SIZE)])
            result.append([round(best_pop[i], 1) for i in range(DNA_SIZE)])
        plt.figure()
        plt.plot(trace)
        plt.title('fitness curve')
        plt.savefig('fitness curve.jpg')
        plt.show()


def make_vel_velest(vel, vs, high, cor_p, cor_s):
    s = open('/home/yin/zhangxiaoyue/tomoDD.mod', 'w')
    s.write(' Output model:\n')
    s.write(' {:2}\n'.format(len(vel)))
    for i in range(len(vel)):
        s.write(' {:4.2f}{:11.1f}{}\n'.format(vel[i], high[i], '    001.00'))
    s.write(' {:2}\n'.format(len(vel)))
    for i in range(len(vel)):
        s.write(' {:4.2f}{:11.1f}{}\n'.format(vs[i], high[i], '    001.00'))

    ##  write the COR file
    f = open('/home/yin/zhangxiaoyue/sta_copy.COR', 'r')
    file_data = []
    line = f.readline()  #
    while line:
        file_data.append(line)  # 往列表后面插入
        line = f.readline()
    f.close

    s = open('/home/yin/zhangxiaoyue/sta.COR', 'w')
    s.write('(a4,f7.4,a1,1x,f8.4,a1,1x,i4,1x,i1,1x,i3,1x,f5.2,2x,f5.2)\n')
    for i in range(19):
        if i == 1:
            s.write('{}{:6.2f}{:7.2f}{}\n'.format(file_data[i][0:33], cor_p[i - 1], cor_s[i - 1],
                                                  '       lon,z,model,icc,ptcor,stcor'))
        elif i > 1:
            s.write('{}{:6.2f}{:7.2f}\n'.format(file_data[i][0:33], cor_p[i - 1], cor_s[i - 1]))


if __name__ == '__main__':
    import numpy as np

    start = main()
    start.mk_syn_model()
    time_start = time.time()
    # run_velest()
    time_end = time.time()
    time_sum = time_end - time_start
    print(time_sum)


    vel, vel2, elev = read_velestmodel()
    cor = read_cor()
    p_cor = cor[1].reshape(500, 18).astype(np.float)
    s_cor = cor[2].reshape(500, 18).astype(np.float)

    vp, vs, cor_p, cor_s = [], [], [], []
    for i in range(vel.shape[1]):
        vp.append(np.median(vel[:, i]))
        vs.append(np.median(vel2[:, i]))
    for j in range(p_cor.shape[1]):
        cor_p.append(np.median(p_cor[:, j]))
        cor_s.append(np.median(s_cor[:, j]))
    y = [-1 * d for d in elev[i]]

    init_plotting_isi(9, 9)
    plt.rcParams['axes.labelsize'] = 8
    ax = plt.subplot(111)
    for i in range(vel.shape[0]):
        ax.plot(vel[i, :], y, 'grey', linewidth=0.5)
        ax.plot(vel2[i, :], y, 'lightgreen', linewidth=0.5)
    ax.plot(vp, y, 'r-', linewidth=1.0)
    ax.plot(vs, y, 'b-', linewidth=1.0)
    ax.set_title("Final Models")
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Depth (km)")
    plt.savefig('vel.png')

    with open('p_cor.txt', 'w') as f:
        for i in range(len(p_cor)):
            for j in range(18):
                f.write(str(p_cor[i][j]) + ' ')
            f.write('\n')

    with open('s_cor.txt', 'w') as fs:
        for i in range(len(s_cor)):
            for j in range(18):
                fs.write(str(s_cor[i][j]) + ' ')
            fs.write('\n')

    make_vel_velest(vp, vs, elev[0, :], cor_p, cor_s)
