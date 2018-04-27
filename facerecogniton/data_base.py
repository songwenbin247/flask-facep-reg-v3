import sqlite3
import pickle


class FeaturesDB():

    def __init__(self, db_path='./models/features_128D.db.42p'):
        def adapt_list(dat):
            return pickle.dumps(dat)

        def convert_list(s):
            return pickle.loads(s)

        sqlite3.register_adapter(list, adapt_list)
        sqlite3.register_converter("list", convert_list)

        self.con = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.db_path = db_path
        self.cur =  self.con.cursor()

        self.cur.execute('create table if not exists Left (name text, feature list,  line integer primary key)')
        self.cur.execute('create table if not exists Right (name text, feature list, line integer primary key)')
        self.cur.execute('create table if not exists Center (name text, feature list, line integer primary key)')

    def __del__(self):
        self.cur.close()
        self.con.close()

    def get_dict(self, pos=None):
        dict = {}
        cmd = []
        if pos is None:
            poss = ['Left', 'Right', 'Center']
            for p in poss:
                cmd.append("select name, feature from {pos}".format(pos=p))
        else:
            cmd.append("select name, feature from {pos}".format(pos=pos))

        for c in cmd:
            self.cur.execute(c)
            for row in self.cur:
                try:
                    dict[row[0]].append(row[1])
                except KeyError:
                    dict[row[0]] = [row[1]]
        return dict

    def get_names(self, pos=None):
        cmd = []
        names = []
        if pos is None:
            poss = ['Left', 'Right', 'Center']
            for p in poss:
                cmd.append("select distinct name from {pos}".format(pos=p))
        else:
                cmd.append("select distinct name from {pos}".format(pos=pos))

        for c in cmd:
            self.cur.execute(c)
            names += [x for x in self.cur]
        ln = list(set(names))
        return [x[0] for x in ln]

    def features(self, person=None, pos=None):
        cmd = []
        if pos is None:
            poss = ['Left', 'Right', 'Center']
            for p in poss:
                if person is None:
                    cmd.append("select feature from {pos}".format(pos=p))
                else:
                    cmd.append("select feature from {pos} where name='{name}'".format(pos = p, name=person))
        else:
            if person is None:
                cmd.append("select feature from {pos}".format(pos=pos))
            else:
                cmd.append("select feature from {pos} where name='{name}'".format(pos=pos, name=person))

        for s in cmd:
            self.cur.execute(s)
            while True:
                try:
                    yield self.cur.next()[0]
                except StopIteration:
                    break
        raise StopIteration

    def add_feature(self, person, feature, pos):
        self.cur.execute('insert into %s (name, feature) values(?, ?)' % pos, (person, feature))
        self.con.commit()

    def add_features(self, person, features, pos):
        for feature in features:
            self.cur.execute('insert into %s (name, feature) values(?, ?)' % pos, (person, feature))
        self.con.commit()

    def del_person(self, person, pos = None):
        if pos is None:
            poss = ['Left', 'Right', 'Center']
            for p in poss:
                self.cur.execute('delete from %s where name=:name' % p, {'name': person})
        else:
            self.cur.execute('delete from %s where name=:name' % pos, {'name': person})
        self.con.commit()


class Batches(FeaturesDB):
    poss = ['Left', 'Right', 'Center']

    def __init__(self,db_path='./models/features_128D.db', person=None, pos=None):
        FeaturesDB.__init__(self, db_path=db_path)
        self.person = person
        self.pos = pos
        self.cmd = {}
        self.poss_nums = {'Left': [0, 0, 0], 'Right': [0, 0, 0], 'Center': [0, 0, 0]}
        self.sum_feature = 0
        self.reset(person, pos)

    def reset(self,person=None, pos=None):
        self.cmd.clear()
        if pos is None:
            for p in self.poss:
                if person is None:
                    self.cmd[p] = "select count() from {pos}".format(pos=p)
                else:
                    self.cmd[p] = "select count() from {pos} where name='{name}'".format(pos = p, name=person)
        else:
            if person is None:
                self.cmd[pos] = "select count() from {pos}".format(pos=pos)
            else:
                self.cmd[pos] = "select count() from {pos} where name='{name}'".format(pos=pos, name=person)

        self.sum_feature = 0
        for p in self.cmd.keys():
            self.cur.execute(self.cmd[p])
            self.poss_nums[p][0] = self.cur.fetchone()[0]
            self.sum_feature += self.poss_nums[p][0]

        self.cmd.clear()
        if pos is None:
            for p in self.poss:
                if person is None:
                    self.cmd[p] = "select feature from {pos}".format(pos=p)
                else:
                    self.cmd[p] = "select feature from {pos} where name='{name}'".format(pos=p, name=person)
        else:
            if person is None:
                self.cmd[pos] = "select feature from {pos}".format(pos=pos)
            else:
                self.cmd[pos] = "select feature from {pos} where name='{name}'".format(pos=pos, name=person)

    def get_sum(self):
        return self.sum_feature

    def get_batch_start(self, num):
        self.reset(self.person, self.pos)
        return self.get_next_batch(num)

    def get_next_batch(self, num):
        res = []
        i = 0
        num_n = 0
        if num > self.sum_feature:
            return [[]]
        for p in self.cmd.keys():
            if self.poss_nums[p][2] == 0:
                self.cur.execute(self.cmd[p])
                self.poss_nums[p][2] = 1
                self.poss_nums[p][1] = self.poss_nums[p][0]
            if self.poss_nums[p][1] >= num:
                self.poss_nums[p][1] -= num
                num_n = num
                num = 0
            else:
                num_n = self.poss_nums[p][1]
                num -= num_n
                self.poss_nums[p][1] = 0

            res += [self.cur.next()[0] for _ in range(num_n)]
            if num == 0:
                break
        self.sum_feature -= len(res)
        return res


if __name__ == '__main__':
    import numpy as np
    import json
    fdb= FeaturesDB("../models/features_128D.db")
    f = open('../models/facerec_128D_12p.txt', 'rb')
    feature_data_set = json.loads(f.read())
    f.close()
    for name in feature_data_set.keys():
        for p in ['Left', 'Right', 'Center']:
            fdb.add_features(name, feature_data_set[name][p], p)



    print 'End'
    # feature_dirc = {}
    # features_all = {}
    # features = np.random.random([20,128]).tolist()
    # feature = features[0]
    #
    # # open face95 database, it will be created if no exist.
    # fdb = FeaturesDB('./face95.db')
    #
    # # get all person name in this database
    # name = fdb.get_names()
    #
    # # feature_dirc= {'name2' : [[feature1], [feature2],...], 'name2': [], }
    # # get the 'Left' features  with the corresponding person name as the keys
    # feature_dirc = fdb.get_dict('Left')
    # # get the all angles features with the corresponding person name as the keys
    # feature_dirc = fdb.get_dict()
    #
    # # add a feature
    # fdb.add_feature('name', feature, 'Left')
    # # add a lot of features
    # fdb.add_features('name', features, 'Left')
    #
    # # get all angles features of the person named "songwb"
    # features_all = [x for x in fdb.features(person='songwb')]
    # # get the "Left" features of the person named "songwb"
    # features_all['5'] = [x for x in fdb.features(person='songwb', pos='Left')]
    # # get all person's all angles feature
    # features_all = [x for x in fdb.features()]
    # # get all person's 'Left' feature
    # features_all = [x for x in fdb.features(pos='Left')]
    #
    # #delete a person's 'Left' features
    # fdb.del_person('songwb', 'Left')
    # # delete a persong's all angles features
    # fdb.del_person('songwb')
    #
    #
    # bdb = Batches('./face95.db')
    # num = bdb.get_sum()







