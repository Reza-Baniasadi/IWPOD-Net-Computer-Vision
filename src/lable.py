import numpy as np
from os.path import isfile


# ---------------- Label Class -----------------
class BoundingBox:
    """
    Represents a labeled bounding box with optional probability.
    Coordinates are stored as top-left and bottom-right points.
    """

    def __init__(self, class_id=-1, top_left=np.array([0., 0.]), bottom_right=np.array([0., 0.]), probability=None):
        self._top_left = top_left
        self._bottom_right = bottom_right
        self._class_id = class_id
        self._probability = probability

    def __str__(self):
        return f"Class: {self._class_id}, top_left(x:{self._top_left[0]},y:{self._top_left[1]}), bottom_right(x:{self._bottom_right[0]},y:{self._bottom_right[1]})"

    def copy(self):
        return BoundingBox(self._class_id, self._top_left.copy(), self._bottom_right.copy(), self._probability)

    # ----- Properties -----
    def width_height(self):
        return self._bottom_right - self._top_left

    def center(self):
        return self._top_left + self.width_height() / 2

    def top_left(self):
        return self._top_left

    def bottom_right(self):
        return self._bottom_right

    def top_right(self):
        return np.array([self._bottom_right[0], self._top_left[1]])

    def bottom_left(self):
        return np.array([self._top_left[0], self._bottom_right[1]])

    def class_id(self):
        return self._class_id

    def area(self):
        return np.prod(self.width_height())

    def probability(self):
        return self._probability

    # ----- Setters -----
    def set_class_id(self, class_id):
        self._class_id = class_id

    def set_top_left(self, top_left):
        self._top_left = top_left

    def set_bottom_right(self, bottom_right):
        self._bottom_right = bottom_right

    def set_width_height(self, wh):
        c = self.center()
        self._top_left = c - 0.5 * wh
        self._bottom_right = c + 0.5 * wh

    def set_probability(self, prob):
        self._probability = prob

    # ----- File I/O -----
    @staticmethod
    def read_labels(file_path):
        """
        Read bounding boxes from file
        """
        if not isfile(file_path):
            return []

        boxes = []
        with open(file_path, 'r') as f:
            for line in f:
                vals = line.strip().split()
                class_id = int(vals[0])
                ccx, ccy = float(vals[1]), float(vals[2])
                w, h = float(vals[3]), float(vals[4])
                prob = float(vals[5]) if len(vals) == 6 else None
                center = np.array([ccx, ccy])
                wh = np.array([w, h])
                boxes.append(BoundingBox(class_id, center - wh / 2, center + wh / 2, prob))
        return boxes

    @staticmethod
    def write_labels(file_path, labels, write_probs=True):
        """
        Write bounding boxes to file
        """
        with open(file_path, 'w') as f:
            for label in labels:
                cc = label.center()
                wh = label.width_height()
                class_id = label.class_id()
                prob = label.probability()
                if prob is not None and write_probs:
                    f.write(f"{class_id} {cc[0]} {cc[1]} {wh[0]} {wh[1]} {prob}\n")
                else:
                    f.write(f"{class_id} {cc[0]} {cc[1]} {wh[0]} {wh[1]}\n")


# ---------------- Shape Class -----------------
class Shape:
    """
    Represents a shape defined by points and optional text label.
    """

    def __init__(self, points=np.zeros((2, 0)), max_sides=4, text=''):
        self.points = points
        self.max_sides = max_sides
        self.text = text

    def is_valid(self):
        return self.points.shape[1] > 2

    # ----- File I/O -----
    def write(self, file_obj):
        file_obj.write(f"{self.points.shape[1]},")
        flat_pts = self.points.flatten()
        file_obj.write(','.join([f"{v}" for v in flat_pts]))
        file_obj.write(f",{self.text},\n")

    def read(self, line):
        data = line.strip().split(',')
        num_pts = int(data[0])
        values = data[1:(num_pts * 2 + 1)]
        text = data[num_pts * 2 + 1] if len(data) >= num_pts * 2 + 2 else ''
        self.points = np.array([float(v) for v in values]).reshape((2, num_pts))
        self.text = text

    @staticmethod
    def read_shapes(file_path):
        shapes = []
        with open(file_path, 'r') as f:
            for line in f:
                s = Shape()
                s.read(line)
                shapes.append(s)
        return shapes

    @staticmethod
    def write_shapes(file_path, shapes):
        if len(shapes):
            with open(file_path, 'w') as f:
                for s in shapes:
                    if s.is_valid():
                        s.write(f)
