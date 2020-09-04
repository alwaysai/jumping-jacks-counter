import argparse
import time
from collections import deque
import csv
import edgeiq


def get_last_value(values, default):
    try:
        val = values[-1]
    except IndexError:
        val = default
    return val


class SmoothedValue:
    def __init__(self, smooth_len):
        self._smooth_len = smooth_len
        self.raw = []
        self.smoothed = []

    def append(self, value):
        self.raw.append(value)
        self.smoothed.append(
                sum(self.raw[-self._smooth_len:]) /
                min(len(self.raw), self._smooth_len))

    @property
    def smoothed_value(self):
        return get_last_value(self.smoothed, 0)


class ZeroCrossDownValue:
    def __init__(self, check_span, center, prediction_factor=1):
        self.center = center
        self._prediction_factor = prediction_factor
        self.history = deque(maxlen=check_span)
        self.raw = []
        self.centered = []

    def append(self, value):
        self.raw.append(value)
        if value == -1:
            value = self._get_prediction()
        else:
            value = self.center.smoothed_value - value
        complete = value <= 0 and get_last_value(self.centered, 0) > 0
        self.centered.append(value)
        self.history.appendleft(complete)

    def _get_prediction(self):
        if len(self.centered) >= 2:
            return self.centered[-1] + self._prediction_factor * (
                    self.centered[-1] - self.centered[-2])
        else:
            return 0

    def clear_state(self):
        try:
            while True:
                self.history.pop()
        except IndexError:
            pass

    @property
    def has_changed(self):
        return True in self.history


class Wrists:
    def __init__(
            self, center_smooth_samples, memory_samples, prediction_factor):
        self.center = SmoothedValue(center_smooth_samples)
        self.left_wrist = ZeroCrossDownValue(
                memory_samples, self.center, prediction_factor)
        self.right_wrist = ZeroCrossDownValue(
                memory_samples, self.center, prediction_factor)

    def update(self, pose):
        # Save only `y` values for wrists
        self.center.append(pose.key_points['Neck'][1])
        self.left_wrist.append(pose.key_points['Left Wrist'][1])
        self.right_wrist.append(pose.key_points['Right Wrist'][1])

        return self.left_wrist.has_changed and self.right_wrist.has_changed

    def clear_state(self):
        self.left_wrist.clear_state()
        self.right_wrist.clear_state()

    @property
    def header(self):
        return (
                    'Left (raw)',
                    'Left (centered)',
                    'Right (raw)',
                    'Right (centered)',
                    'Center (raw)',
                    'Center (smoothed)')

    @property
    def points(self):
        data = zip(
                self.left_wrist.raw,
                self.left_wrist.centered,
                self.right_wrist.raw,
                self.right_wrist.centered,
                self.center.raw,
                self.center.smoothed)
        return data


class JumpingJacksTracker:
    def __init__(self):
        self.wrists = Wrists(
                center_smooth_samples=10,
                memory_samples=4,
                prediction_factor=0.5)
        self.count = 0

    def update(self, pose):
        if self.wrists.update(pose) is True:
            self.count += 1
            self.wrists.clear_state()

    def save_history(self):
        filename = 'data-{}.csv'.format(time.strftime('%Y%m%d-%H%M%S'))
        print('Saving data to {}'.format(filename))
        with open(filename, mode='w') as f:
            writer = csv.writer(
                    f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(self.wrists.header)
            for point in self.wrists.points:
                writer.writerow([str(x) for x in point])


def main(cam=0, video_file=None, debug=False):
    if video_file is not None:
        video_stream_class = edgeiq.FileVideoStream
        kwargs = {'path': video_file, 'play_realtime': True}
    else:
        video_stream_class = edgeiq.WebcamVideoStream
        kwargs = {'cam': cam}

        edgeiq.WebcamVideoStream.more = lambda x: True

    if edgeiq.is_jetson_xavier_nx():
        model = "alwaysai/human_pose_xavier_nx"
    else:
        model = "alwaysai/human-pose"

    pose_estimator = edgeiq.PoseEstimation(model)

    if edgeiq.is_jetson():
        pose_estimator.load(engine=edgeiq.Engine.TENSOR_RT)
    else:
        pose_estimator.load(engine=edgeiq.Engine.DNN)

    print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    print("Engine: {}".format(pose_estimator.engine))
    print("Accelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()

    jj = JumpingJacksTracker()

    try:
        with video_stream_class(**kwargs) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while video_stream.more():
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)
                # Generate text to display on streamer
                text = ["Model: {}".format(pose_estimator.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                if len(results.poses) > 0:
                    jj.update(results.poses[0])
                else:
                    text.append('No poses found')
                text.append("Count: {}".format(jj.count))

                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        if debug is True:
            jj.save_history()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Jumping Jacks Counter')
    parser.add_argument(
            '--camera', type=int, default=0,
            help='Set the camera index. (default: 0)')
    parser.add_argument(
            '--video-file', type=str,
            help='Perform counting on a video file')
    parser.add_argument(
            '--debug', action='store_true',
            help='Save the data to a CSV file')
    args = parser.parse_args()
    main(cam=args.camera, video_file=args.video_file, debug=args.debug)
