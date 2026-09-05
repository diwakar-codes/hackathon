VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
PERSON_CLASS = "person"

MOVE_THRESHOLD_PX = 5


class SimpleTracker:
    """Nearest-centroid tracker -- just enough to time how long an object
    has stayed put, which is what the stopped-vehicle/loitering signals need.
    Not built for crowded-scene accuracy; Stage 2's VLM is what actually
    judges whether a flagged clip is a real anomaly.
    """

    def __init__(self, max_distance=50, max_missed=10):
        self.tracks = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, detections, t):
        unmatched = list(range(len(detections)))

        for track in self.tracks.values():
            track["matched"] = False

        for tid, track in list(self.tracks.items()):
            best_j, best_d = None, self.max_distance
            for j in unmatched:
                cx, cy, _cls = detections[j]
                tx, ty = track["centroid"]
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if d < best_d:
                    best_j, best_d = j, d

            if best_j is not None:
                cx, cy, cls_name = detections[best_j]
                if best_d > MOVE_THRESHOLD_PX:
                    track["stationary_since"] = t
                track["centroid"] = (cx, cy)
                track["class"] = cls_name
                track["missed"] = 0
                track["matched"] = True
                unmatched.remove(best_j)
            else:
                track["missed"] += 1
                if track["missed"] > self.max_missed:
                    del self.tracks[tid]

        for j in unmatched:
            cx, cy, cls_name = detections[j]
            self.tracks[self.next_id] = {
                "centroid": (cx, cy),
                "class": cls_name,
                "missed": 0,
                "stationary_since": t,
                "matched": True,
            }
            self.next_id += 1

        return self.tracks
