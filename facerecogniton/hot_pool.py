import faste
from faste.util import hashable
import time
import sys
from sklearn import neighbors


class FIFOCache(faste.caches.RRCache):
    """
    First In First Out cache.
    When the max_size is reached, the cache evicts the first key set/accessed without any regard to when or
    how often it is accessed.

    Parameters
    ----------
    max_size : int
        Max size of the cache. Must be > 0
    populate : dict
        Keyword argument with values to populate cache with, in no given order. Can't be larger than max_size

    """

    def __init__(self, max_size, populate=None):
        super(FIFOCache, self).__init__(max_size, populate=populate)

    def __setitem__(self, key, value):
        if not hashable(key):
            raise TypeError("unhashable type: {0!r}".format(type(key.__class__.__name__)))

        self._store[key] = value
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)
        return value


class LRUCache(FIFOCache):
    """
    Least recently used cache implementation.
    When the max size is reached, the least recently used value is evicted from the cache.

    Parameters
    ----------
    max_size : int
        Max size of the cache. Must be > 0
    populate : dict
        Keyword argument with values to populate cache with, in no given order. Can't be larger than max_size

    """

    def __init__(self, max_size, populate=None):
        super(LRUCache, self).__init__(max_size, populate=populate)

    def __getitem__(self, item):
        if item not in self._store:
            raise KeyError("key {0!r} not in cache".format(item))

        self._store.move_to_end(item)
        return self._store[item]


# { time: {
#          'rect': [px,py,lx,ly],
#           'pos': 'Left'
#           },
# }

class PositionStream():
    def __init__(self, max_frame=None):
        if max_frame is None:
            max_frame = 30
        self.fifo = FIFOCache(max_frame)

    def push(self, rect, pos):
        self.fifo.update((time.time, {'rect': rect, 'pos': pos}))

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
    last_adjust_time = 0
    frames_sum = 0
    pps = 5
    timeout = 1.0
    instance_list = []

    def __init__(self, max_person, radius):
        self.hot_pool = LRUCache(max_person)
        HotFaces.instance_list.append(self)
        self.RNC = neighbors.RadiusNeighborsClassifier(radius=radius)
        self.labels_Y = []
        self.vac_X = []

    @classmethod
    def auto_adjust_stream_size(cls):  # called when receiving a frame.
        if cls.last_adjust_time != 0:
            cls.frames_sum += 1
        else:
            cls.last_adjust_time = time.time()
            cls.frames_sum = 0

        if cls.frames_sum > 20:
            pps = cls.frames_sum / (time.time() - cls.last_adjust_time)
            cls.last_adjust_time = time.time()
            cls.frames_sum = 0
            if abs(int(pps) - cls.pps) > 3:
                cls.pps = int(pps)
            for i in cls.instance_list:
                i.stream.resize(cls.pps * cls.timeout + 6)

    def update(self, name, rect, pos):
        time_new = time.time()
        if name not in self.hot_pool.keys():
            posi = PositionStream(HotFaces.pps * HotFaces.timeout + 4)
            posi.push(rect, pos)
            self.hot_pool.update((name, {'time': time_new, 'PosiStream': posi}))
        else:
            dic = self.hot_pool.get(name)
            dic['time'] = time_new
            dic['PosiStream'].push(rect, pos)
        self.labels_Y = []
        self.vac_X = []
        for name in self.hot_pool.keys():
            tposi = self.hot_pool.get(name)
            if time_new - tposi['time'] > HotFaces.timeout:
                self.hot_pool.pop(name)
                continue

            for posi in tposi['PosiStream'].position():
                self.labels_Y.append(name)
                self.vac_X.append([posi[1][0], posi[1][1]])

        self.RNC.fit(self.labels_Y, self.labels_Y)

    def isHot(self, name, rect, pos):
        time_now = time.time()
        if name not in self.hot_pool.keys():
            return (0, '')

        if time_now - self.hot_pool.get(name) > HotFaces.timeout:  # the time is greater than 1s since the last update.
            self.hot_pool.pop(name)
            return (0, '')

        ret = self.RNC.predict([rect[0], rect[1]])
        return self.labels_Y[ret[0]]
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
    a = frame_stream(5)


