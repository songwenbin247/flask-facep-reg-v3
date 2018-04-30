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


# { time: {
#          'rect': [px,py,lx,ly],
#           'pos': 'Left'
#           },
# }

class PositionStream():
    def __init__(self, ):
        self.fifo = FIFOCache(2)

    def push(self, rect, pos):
        self.fifo.update((time.time(), {'rect': rect, 'pos': pos}))

    def resize(self, max_frame):
        self.fifo.max_size = max_frame

    def position(self):
        for k in self.fifo.keys():
            posi = self.fifo.get(k)
            yield [k, posi['rect']]

# {name: {
#       'time': time,
#       'PosiStream': tposi
#       }
# }
class HotFaces():
    timeout = 5.0

    def __init__(self, max_person):
        self.hot_pool = LRUCache(max_person)

    def update(self, name, rect, pos=None):
        time_new = time.time()
        if name not in self.hot_pool.keys():
            posi = PositionStream()
            posi.push(rect, pos)
            self.hot_pool.update((name, {'time': time_new, 'PosiStream': posi}))
        else:
           self.hot_pool.get(name)['time'] = time_new
           self.hot_pool.get(name)['PosiStream'].push(rect, pos)

    def isHot(self, name, rect, pos=None):
        time_now = time.time()
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
        # pl = [[px1, py1, sx1, sy1], [px2, py2, sx2, sy2]]
        # px = (px2 - px1)/ (time_2 - time-1) * (time_new - time_2) + px2
        # py = (py2 - py1) / (time_2 - time - 1) * (time_new - time_2) + py2
        t1 = tl[1] - tl[0]
        t2 = time_now - tl[1]
        px = ((pl[1][0] - pl[0][0]) / t1) * t2 + pl[1][0]
        py = ((pl[1][1] - pl[0][1]) / t1) * t2 + pl[1][1]
        #print("### %f %f %d %d  %f" % (abs(px - rect[0]), abs(py - rect[1]), rect[2], rect[3], t2))
        dis = np.sqrt( np.square(px - rect[0]) + np.square(py - rect[1]))
        #if  dis > 350 * t2:
        if  dis > 700 * t2:
            print ('*** dis = %f, t2=%f' % (dis, t2))
            return (False,False)
        self.update(name, rect)
        return (True, True)
        # if len(re) < 2:
        #     return (1, '')
        #

        # def calc_change(vtl):
        #     ret = []
        #     if len(vtl) < 2:
        #         return ret
        #     prev_v = 0
        #     prev_t = 0
        #     i = 0
        #     for v, t in vtl:
        #         if i != 0:
        #             i = 1
        #             ret.append([(v - prev_v)/(t[i] - prev_t), prev_t])
        #         prev_v = v
        #         prev_t = t
        #     return ret
        #
        # # x, y axis position
        # px_t = [[x[1][0], x[0]] for x in re]
        # py_t = [[x[1][0], x[1]] for x in re]
        #
        # # x, y axis position with test point
        # px_t_u = px_t[:]
        # py_t_u = py_t[:]
        # px_t_u.append([rect[0], time])
        # py_t_u.append([rect[0], time])
        #
        # # x, y axis speed
        # vx_t = calc_change(px_t_u)
        # vy_t = calc_change(py_t_u)
        #
        # # x, y axis acceleration
        # ax_t = calc_change(vx_t)
        # ay_t = calc_change(vy_t)
        #
        # vx = [x[0] for x in vx_t]
        # vy = [x[0] for x in vy_t]
        # ax = [x[0] for x in ax_t]
        # ay = [x[0] for x in ay_t]
        # ret = [0] * 4
        # self.lm.fit(range(0,len(vx) - 1), vx[:-1])
        # ret[0] = self.lm.predict([vx[-1]])[0]
        # self.lm.fit(range(0, len(vy) - 1), vy[:-1])
        # ret[1] = self.lm.predict([vy[-1]])[0]
        #
        # self.lm.fit(range(0, len(ay) - 1), ay[:-1])
        # ret[2] = self.lm.predict([ay[-1]])[0]
        # self.lm.fit(range(0, len(ay) - 1), ay[:-1])
        # ret[3] = self.lm.predict([ay[-1]])[0]


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
