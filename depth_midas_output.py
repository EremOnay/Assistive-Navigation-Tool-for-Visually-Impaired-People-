import cv2
import numpy as np
import time
import argparse
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import load_delegate


class MidasDepthEstimator:
    def __init__(self, model_path: str):
        # Load Edge TPU–compatible TFLite interpreter
        delegates = []
        try:
            delegates = [load_delegate('libedgetpu.so.1.0')]
        except Exception:
            pass
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=delegates
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Quantization params and input size
        self.scale, self.zero_point = self.input_details[0]['quantization']
        _, self.height, self.width, _ = self.input_details[0]['shape']

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        # Convert BGR->RGB, resize, normalize, quantize
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.width, self.height))
        data = resized.astype(np.float32) / 255.0
        data = data / self.scale + self.zero_point
        data = np.clip(data, 0, 255).astype(np.uint8)
        return data.reshape(1, self.height, self.width, 3)

    def infer(self, frame: np.ndarray) -> (np.ndarray, float):
        input_data = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        start = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start) * 1000.0

        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        depth_map = np.squeeze(output)
        # Normalize to 0-255 and resize to frame dimensions
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        return depth_map, inference_time

    def calculate_average_depth(self, depth_map: np.ndarray,
                                x1: int, y1: int,
                                x2: int, y2: int) -> float:
        # Compute mean of valid depth pixels in ROI
        roi = depth_map[y1:y2, x1:x2]
        valid = roi[(roi > 0) & (roi < 255)]
        return float(np.mean(valid)) if valid.size > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Live Midas depth estimation on a video stream with depth map display.'
    )
    parser.add_argument('--model', default='models/Midas-V2_2_edgetpu.tflite',
                        help='Path to the TFLite Midas model')
    parser.add_argument('--bbox', nargs=4, type=int, required=True,
                        metavar=('x1', 'y1', 'x2', 'y2'),
                        help='Bounding box coordinates')
    parser.add_argument('--gst', type=str,
                        help='GStreamer pipeline string (optional)')
    parser.add_argument('--device', type=str, default='/dev/video0',
                        help='Video device path (e.g., /dev/video0)')
    args = parser.parse_args()

    # Initialize video capture
    if args.gst:
        cap = cv2.VideoCapture(args.gst, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(args.device)

    if not cap.isOpened():
        print(f"Error: cannot open video source {args.gst or args.device}")
        return

    estimator = MidasDepthEstimator(args.model)
    x1, y1, x2, y2 = args.bbox

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        depth_map, inference_time = estimator.infer(frame)
        avg_depth = estimator.calculate_average_depth(depth_map, x1, y1, x2, y2)

        # Colorize depth map for better visualization
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

        # Overlay bbox and info on RGB frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Depth: {avg_depth:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'{inference_time:.1f} ms', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display both RGB and depth map
        cv2.imshow('RGB Stream', frame)
        cv2.imshow('Depth Map', depth_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
