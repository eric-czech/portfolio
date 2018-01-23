
from myo import DeviceListener

class Listener(DeviceListener):

    def __init__(self):
        super(Listener).__init__()
        self.poses = []
        self.orientations = []
        self.events = []

    def on_pair(self, myo, timestamp, firmware_version):
        print("Hello, Myo!")

    def on_unpair(self, myo, timestamp):
        print("Goodbye, Myo!")

    def on_pose(self, myo, timestamp, pose):
        print('[Pose]', pose)

    def on_orientation_data(self, myo, timestamp, quat):
        self.orientations.append((myo, timestamp, quat))
        print("Orientation:", quat.x, quat.y, quat.z, quat.w)


