# https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, threshAppeared = 10, distThresh = 25, maxDisappeared=5):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        # number of consecutive frames object has appeared
        self.appeared = OrderedDict()
        # number of consecutive frames object has been marked as "disappeared"
        self.disappeared = OrderedDict()

        # detected object in more than "threshAppeared" consecutive frames
        self.confirmed = False

        self.threshAppeared = threshAppeared
        self.maxDisappeared = maxDisappeared
        self.distThresh = distThresh

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        if len(bbox) > 0:
            self.bboxes[self.nextObjectID] = bbox
        self.appeared[self.nextObjectID] = 0
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        if objectID in self.bboxes: 
            del self.bboxes[objectID]
        del self.appeared[objectID]
        del self.disappeared[objectID]

    def update(self, centroids, bboxes):
        if len(centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for i in range(len(centroids)):
                if len(bboxes) > 0:
                    self.register(centroids[i], bboxes[i])
                else:
                    self.register(centroids[i],[])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively
            D = dist.cdist(np.array(objectCentroids), centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                if D[row,col] < self.distThresh:
                    self.objects[objectID] = centroids[col]
                    if len(bboxes) > 0:
                        self.bboxes[objectID] = bboxes[col]
                    self.appeared[objectID] += 1
                    self.disappeared[objectID] = 0

                    usedRows.add(row)
                    usedCols.add(col)
                
                if self.appeared[objectID] > self.threshAppeared:
                    self.confirmed = True
                    self.confirmedCentroid = centroids[col]

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.appeared[objectID] = 0
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    if len(bboxes) > 0:
                        self.register(centroids[col], bboxes[col])
                    else:
                        self.register(centroids[col],[])

        return self.objects