from faste import util, caches
import time
import numpy as np

class RRCache(caches.RRCache):
    def move_to_end(self, key, last=True):
        va = self._store.pop(key)
        self._store[key] = va


class FIFOCache(RRCache):

    def __init__(self, max_size, populate=None):
        super(FIFOCache, self).__init__(max_size, populate=populate)

    def __setitem__(self, key, value):
        if not util.hashable(key):
            raise TypeError("unhashable type: {0!r}".format(type(key.__class__.__name__)))

        self._store[key] = value
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)
        return value


class LRUCache(FIFOCache):

    def __init__(self, max_size, populate=None):
        super(LRUCache, self).__init__(max_size, populate=populate)

    def __getitem__(self, item):
        if item not in self._store:
            raise KeyError("key {0!r} not in cache".format(item))

        self.move_to_end(item)
        return self._store[item]

from matplotlib import pyplot as plt
class canvas_show():
    def __init__(self, x_max, y_min = None, y_max = None):
        self.x_max = x_max
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.lines, = self.ax.plot([], [], '-')
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(0, self.x_max)
        if y_min is not None and y_max is not None:
            self.ax.set_ylim(y_min, y_max)
        self.fig.show()
        self.y = [0] * self.x_max
        self.lines.set_xdata(range(0, self.x_max, 1))

    def up_va(self, va):
        self.y = self.y[1: self.x_max]
        self.y.append(va)
        self.lines.set_ydata(self.y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Filter():
    def __init__(self, max_size=3):
        self.vallF =FIFOCache(max_size)
        self.valhF = FIFOCache(max_size)
        self.i = 0

    def get_va(self, vall,valh ):
        xl = [self.vallF.get(i) for i in self.vallF.keys()]
        xh = [self.valhF.get(i) for i in self.valhF.keys()]
        self.vallF.update((self.i, vall))
        self.valhF.update((self.i, valh))
        self.i += 1
        if xl != []:
             disl = abs(vall - np.mean(xl))
             dish = abs(valh - np.mean(xh))
             return abs(dish - disl)
        else:
            return 0


class Anormal():
    def __init__(self, threshold = 0.45):
        self.threshold = threshold

    def filter(self, feas, threshold = None):
        if threshold is None:
            threshold = self.threshold

        feas = np.array(feas)
        mean_feas = np.mean(feas, axis=0)
        feas_dis = np.sqrt(np.sum(np.square(feas - mean_feas), axis=1))
        return feas[feas_dis < threshold]

# { time: {
#          'landmark': [nx,ny],
#           'pos': 'Left'
#           },
# }

class PositionStream():
    def __init__(self, ):
        self.fifo = FIFOCache(2)

    def push(self, rect, pos):
        self.fifo.update((time.time(), {'landmark': rect, 'pos': pos}))

    def resize(self, max_frame):
        self.fifo.max_size = max_frame

    def position(self):
        for k in self.fifo.keys():
            posi = self.fifo.get(k)
            yield [k, posi['landmark']]

# {name: {
#       'time': time,
#       'PosiStream': tposi
#       }
# }
class HotFaces():
    timeout = 1.0

    def __init__(self, max_person):
        self.hot_pool = LRUCache(max_person)
        self.canvas_offset = canvas_show(50, -200, 200)

    def update(self, name, landmark, pos=None):
        time_new = time.time()
        if name not in self.hot_pool.keys():
            posi = PositionStream()
            posi.push([landmark[2], landmark[7]], pos)
            self.hot_pool.update((name, {'time': time_new, 'PosiStream': posi}))
        else:
            tposi = self.hot_pool.get(name)
            posi = tposi['PosiStream']
            if time_new - tposi['time'] > HotFaces.timeout:  # the time is greater than 1s since the last update.
                self.hot_pool.pop(name)
            else:
                tposi['time'] = time_new
                posi.push([landmark[2], landmark[7]], pos)

    def remove_person(self, name):
        if name in self.hot_pool.keys():
            self.hot_pool.pop(name)

    def __hot_person(self, name, landmark, time_now, pos=None):
        if name not in self.hot_pool.keys():
            #print '*** %s no in pool' % name
            return (False,False)
        tposi = self.hot_pool.get(name)
        posi = tposi['PosiStream']
        if time_now - tposi['time'] > HotFaces.timeout:  # the time is greater than 1s since the last update.
            self.hot_pool.pop(name)
            print "*** timeout %f" % (time_now - tposi['time'])
            return (False,False)

        if posi.fifo.size < 2:
            print "*** %s fifo size is %d" % (name, posi.fifo.size)
            return (True, False)

        tl = [ k[0] for k in posi.position()]
        pl = [ k[1] for k in posi.position()]
        # tl = [time_1, time_2]
        # pl = [[nx1, ny1], [nx2, ny2]]
        # px = (nx2 - nx1)/ (time_2 - time-1) * (time_new - time_2) + nx2
        # py = (ny2 - ny1) / (time_2 - time - 1) * (time_new - time_2) + ny2
        t1 = tl[1] - tl[0]
        t2 = time_now - tl[1]
        px = ((pl[1][0] - pl[0][0]) / t1) * t2 + pl[1][0]
        py = ((pl[1][1] - pl[0][1]) / t1) * t2 + pl[1][1]
        #print("### %f %f %d %d  %f" % (abs(px - rect[0]), abs(py - rect[1]), rect[2], rect[3], t2))
        dis = np.sqrt( np.square(px - landmark[2]) + np.square(py - landmark[7]))
        self.canvas_offset.up_va(dis - 450 * t2)
        if  dis > 450 * t2:
            print ('*** dis = %f, t2=%f' % (dis, t2))
            return (False,False)
        self.update(name, landmark)
        return (True, True)

    def __hot_postion(self,landmark,time_now, pos=None):

        maybe = {}
        lax = landmark[2]
        lay = landmark[7]
        for it in self.hot_pool.keys():
            tposi = self.hot_pool.get(it)
            posi = tposi['PosiStream']
            tl = [k[0] for k in posi.position()]
            pl = [k[1] for k in posi.position()]
            if len(tl) !=2:
                continue
            # tl = [time_1, time_2]
            # pl = [[nx1, ny1], [nx2, ny2]]
            # px = (nx2 - nx1)/ (time_2 - time-1) * (time_new - time_2) + nx2
            # py = (ny2 - ny1) / (time_2 - time - 1) * (time_new - time_2) + ny2
            t1 = tl[1] - tl[0]
            t2 = time_now - tl[1]
            if t1 > 0.5 or t2 > 0.5:
                continue
            px = ((pl[1][0] - pl[0][0]) / t1) * t2 + pl[1][0]
            py = ((pl[1][1] - pl[0][1]) / t1) * t2 + pl[1][1]
            maybe[it] = [px, py, t2]

        disc = [((lax - x[0])**2 + (lay - x[1]) **2) for x in maybe.values()]
        if len(disc) == 0:
            return None
        min_disc = min(disc)
        id = maybe.keys()[disc.index(min_disc)]
        print "min_disc = %f threshold = %f" % (min_disc,(150 * maybe[id][2]) ** 2)
        if min_disc > (150 * maybe[id][2]) ** 2:
            return None

        return maybe.keys()[disc.index(min_disc)]

    def isHot(self, name, landmark, pos=None):
        time_now = time.time()

        (ishot, isful) = self.__hot_person(name, landmark, time_now, pos)

        if not ishot:
            name_v = self.__hot_postion(landmark,time_now, pos)
            if name_v is None:
                return (False, isful, "")
            else:
                return (True, isful, name_v)
        return (ishot, isful, name)

if __name__ == '__main__':
    a = LRUCache(4)
    a.update(('a',1))
    a.update(('b', 2))
    a.update(('c', 3))
    a.update(('d', 4))

    print a.get('a')
    print a.get('b')
    print a.get('c')
    print a.get('d')
    print a.get('a')
    print a.get('b')
    print a.get('d')

    a.update(('e', 5))
    print a.keys()
    print 'end'
