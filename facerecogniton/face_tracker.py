SCORE_MAX_COUNT = 4
TRACE_PIXELS = 40
FRAME_KEEP_LIMIT = 10
FACE_ID_MAX = 1000

class FaceWindow:
    def __init__(self, x, y, face_id, frame_seq):
        self.x = x
        self.y = y
        self.face_id = face_id
        self.frame_seq = frame_seq
        self.scores = []
        self.name = None
        self.recog_count = 0
        self.phash = None

    def add_score(self, score):
        if len(self.scores) == SCORE_MAX_COUNT:
            del(self.scores[0])
        self.scores.append(score)

    def get_avg_score(self):
        return float(sum(self.scores)) / len(self.scores)

    def set_name(self, name):
        if name is not None and name != " ":
            self.name = name
        self.recog_count += 1

    def get_name(self):
        return self.name

    def set_phash(self, phash):
        self.phash = phash

    def get_distance(self, phash):
        if self.phash == None:
            return 100
        else:
            return sum([ch1 != ch2 for ch1, ch2 in zip(phash, self.phash)])

class FaceTracker:
    def __init__(self):
        self.current_id = 0
        self.current_frame_count = 0
        self.face_list = []

    def drop_timeout_face(self):
        for face in self.face_list:
            if face.frame_seq + FRAME_KEEP_LIMIT < self.current_frame_count:
                self.face_list.remove(face)
                print("remove timeout face")

    def get_new_face_id(self):
        self.current_id += 1
        return  self.current_id % FACE_ID_MAX

    def increase_frame(self):
        self.current_frame_count += 1

    def get_face_by_position(self, rect, image):
        found = []
        x = rect[0] + rect[2] / 2
        y = rect[1] + rect[3] / 2

        for face in self.face_list:
            offset_x = abs(face.x - x)
            offset_y = abs(face.y - y)
            if (offset_x < TRACE_PIXELS and offset_y < TRACE_PIXELS 
                   and face.frame_seq + FRAME_KEEP_LIMIT >= self.current_frame_count):
                found.append(face)
            else:
                print(offset_x,offset_y,face.frame_seq,FRAME_KEEP_LIMIT,self.current_frame_count)
        if (len(found) != 1):
            if (len(found) > 1):
                for face in found:
                    self.face_list.remove(face)
            face_id = self.get_new_face_id();
            newface = FaceWindow(x, y, face_id, self.current_frame_count)
            self.face_list.append(newface)
            print("No found")
        else:
            found[0].x = x
            found[0].y = y
            found[0].frame_seq = self.current_frame_count
            newface = found[0]
            
        #phash = str(imagehash.phash(Image.fromarray(image)))
        #print("get distance with before...\n", found[0].get_distance(phash))
        #found.set_phash(phash)

        return newface
