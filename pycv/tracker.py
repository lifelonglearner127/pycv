import numpy as np
from scipy.spatial import distance as dist


class TrackableObject:

    def __init__(self, object_id, centroid):
        self.object_id = object_id
        self.centroids = [centroid]
        self.counted = False


class CentroidTracker:

    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.max_disappeared = max_disappeared
        self.objects = {}
        self.disappeared = {}

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        rects_num = len(rects)
        if rects_num == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            return self.objects

        input_centroids = np.zeros((rects_num, 2), dtype="int")

        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            cx = (start_x + end_x) // 2
            cy = (start_y + end_y) // 2
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(0, rects_num):
                self.register(input_centroids[i])

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] > D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects
