import argparse
import os
import re
import time
from collections import defaultdict, deque, namedtuple
import numpy as np
import cv2
import gstreamer

import common
from tracker import ObjectTracker
from depth_midas_output import MidasDepthEstimator
from object_scorer import ObjectScorer
from text_generator import TextGenerator
from audio import say

from gstreamer import GstPipeline

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Constants
FRAME_COUNT = 10
CONF_THRESHOLD = 0.3

emergency_flag = False

# Bounding box data structure
BBox = namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])
BBox.__slots__ = ()

# Track and depth history storage
track_history = defaultdict(lambda: deque(maxlen=10))

depth_history = defaultdict(lambda: deque(maxlen=1))
depth_history_mean = defaultdict(lambda: deque(maxlen=1))
depth_history_con = defaultdict(lambda: deque(maxlen=10))


last_said_track_ids = deque(maxlen=2)
last_line = deque(maxlen=2)

accumulated_objects = {}  # track_id -> object

class TermColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Detected object data structure with memory optimization
class DetectedObject:
    __slots__ = ['id', 'conf', 'bbox', 'depth', 'prev_depth', 'dx', 'dy', 'score', 'track_id']

    def __init__(self, id, conf, bbox, depth=0.0, prev_depth=0.0, dx=0.0, dy=0.0, score=0.0, track_id = 0):
        self.id = id  # class ID or tracking ID
        self.conf = conf
        self.bbox = bbox
        self.depth = depth
        self.prev_depth = prev_depth
        self.dx = dx
        self.dy = dy
        self.score = score
        self.track_id = track_id  

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# Load labels from file
def load_labels(path):
    pattern = re.compile(r"\s*(\d+)(.+)")
    labels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            num, text = pattern.match(line).groups()
            labels[int(num)] = text.strip()
    return labels

# Extract output tensors from interpreter
def get_output(interpreter, confi_threshold, top_k):
    boxes = common.output_tensor(interpreter, 0)
    classes = common.output_tensor(interpreter, 1)
    confs = common.output_tensor(interpreter, 2)
    results = []
    for i in range(top_k): 
        if confs[i] < CONF_THRESHOLD: # confi_threshold
            continue
        ymin, xmin, ymax, xmax = boxes[i]
        bbox = BBox(
            xmin=max(0.0, xmin), ymin=max(0.0, ymin),
            xmax=min(1.0, xmax), ymax=min(1.0, ymax)
        )
        results.append(DetectedObject(id=int(classes[i]), conf=float(confs[i]), bbox=bbox))
    return results

########## GStreamer pipeline for video processing ###########
def _run_opencv_video(path, user_function, src_size, inf_size, tracker=None, fps_counter=None):
    w, h = src_size
    inf_w, inf_h = inf_size

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video:", path)
        return

    # Prepare video writers
    video_writer = cv2.VideoWriter('detection_output_phone1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
    depth_writer = cv2.VideoWriter('depth_output_phone1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (inf_w, inf_h))
        objects, labels, depth_map = user_function(frame_resized, (w, h), tracker)

        # Visualize and write detection output
        vis_frame = frame.copy()
        for obj in objects:
            if obj.conf > 0.4:  # Ensure track_id is valid

                x0 = int(obj.bbox.xmin * w)
                y0 = int(obj.bbox.ymin * h)
                x1 = int(obj.bbox.xmax * w)
                y1 = int(obj.bbox.ymax * h)
                label = labels.get(obj.id, str(obj.id))
                depth = obj.depth
                dx, dy = obj.dx, obj.dy

                # label_text = f"{obj.conf:.2f} {label} {depth:.1f}u dx:{dx:.1f} dy:{dy:.1f}"
                # label_text = f"{depth:.1f}, {obj.track_id}"
                label_text = f"{obj.conf:.2f} {label} d:{depth:.1f} tid:{obj.track_id}"

                cv2.rectangle(vis_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(vis_frame, label_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                arrow_scale = 1
                
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                start_point = (int(cx - dx * arrow_scale/ 2), int(cy - dy * arrow_scale / 2))
                end_point = (int(cx + dx * arrow_scale / 2), int(cy + dy * arrow_scale / 2))
                cv2.arrowedLine(vis_frame, start_point, end_point, line_type=cv2.LINE_AA, color=(216, 230, 176), thickness=2, tipLength=0.2)

         # --- Add FPS display here ---
        fps = next(fps_counter)
        fps_text = f"FPS: {fps:.2f}"
        # Position: bottom left (10 px from left, 30 px from bottom)
        cv2.putText(vis_frame, fps_text, (10, vis_frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        # ---------------------------

        video_writer.write(vis_frame)

        # Show processed frame
        # cv2.imshow("Processed Frame", vis_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Visualize and write depth output
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_color = cv2.resize(depth_color, (640, 480))

        for obj in objects:
            if obj.conf > 0.4:

                x0 = int(obj.bbox.xmin * w)
                y0 = int(obj.bbox.ymin * h)
                x1 = int(obj.bbox.xmax * w)
                y1 = int(obj.bbox.ymax * h)
                label = labels.get(obj.id, str(obj.id))
                depth = obj.depth
                dx, dy = obj.dx, obj.dy

                # Calculate center of bounding box
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2

                # Only show depth, at center
                label_text = f"{depth:.1f}"

                cv2.rectangle(depth_color, (x0, y0), (x1, y1), (255, 255, 255), 2)
                cv2.putText(depth_color, label_text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        depth_writer.write(depth_color)


    cap.release()

    video_writer.release()
    depth_writer.release()

###### Convert GStreamer buffer to numpy array ########
def gstbuffer_to_ndarray(buf, appsink_size):
    data = buf.extract_dup(0, buf.get_size())
    arr = np.frombuffer(data, dtype=np.uint8)
    w, h = appsink_size
    try:
        return arr.reshape(h, w, 3)
    except ValueError:
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


####### Main function #######
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=os.path.join('..', 'models', 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'))
    parser.add_argument('--labels', default=os.path.join('..', 'models', 'coco_labels.txt'))
    parser.add_argument('--top_k', type=int, default= 5)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--use_camera', action='store_true', help='Use camera instead of video file')
    args = parser.parse_args()

    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)
    inf_w, inf_h, _ = common.input_image_size(interpreter)
    inference_size = (inf_w, inf_h)

    fps_counter = common.avg_fps_counter(30)
    depth_estimator = MidasDepthEstimator('models/Midas-V2_2_edgetpu.tflite')
    scorer = ObjectScorer()
    textgen = TextGenerator()
    tracker = ObjectTracker("sort").trackerObject.mot_tracker

    # List of allowed object categories
    allowed = {
        "person", "cup", "tv", "chair", "laptop", "bed", "backpack", "couch", "handbag",
        "dining table", "traffic light", "oven", "microwave", "toilet", "refrigerator", "knife",
        "cat", "motorcycle", "car", "bicycle", "bottle", "bus", "stop sign"
    }
    
    frame_counter = 0

    alert_given = False
    
    # debug
    print("System ready to run")

    ### Function to handle each frame
    def user_callback(input_tensor, src_size, mot_tracker):
        nonlocal fps_counter, frame_counter, alert_given

        """
         # --- Manual FPS measurement ---
        if not hasattr(user_callback, "last_time"):
            user_callback.last_time = time.time()
        now = time.time()
        fps = 1.0 / (now - user_callback.last_time) if user_callback.last_time else 0
        user_callback.last_time = now
        print(f"[Manual FPS] {fps:.2f}")
        # ------------------------------
        """

        frame_counter += 1

        # Prepare input for the model 
        # this was using for video input 
        # input_data = input_tensor.astype(np.uint8).reshape(1, *input_tensor.shape)
        # interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        
        # this is for camera input
        if args.use_camera:
            common.set_input(interpreter, input_tensor)
        else:
            input_data = input_tensor.astype(np.uint8).reshape(1, *input_tensor.shape)
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        

        interpreter.invoke()
        objects = get_output(interpreter, args.threshold, args.top_k)
        objects = [o for o in objects if labels.get(o.id) in allowed]

        if not args.use_camera:
            frame = input_tensor
        else:
            frame = gstbuffer_to_ndarray(input_tensor, inference_size)

        if frame is None or frame.size == 0:
            return frame, [], labels

        ## Depth estimation
        depth_map, _ = depth_estimator.infer(frame)
        

        for obj in objects:
            x0 = int(obj.bbox.xmin * src_size[0])
            y0 = int(obj.bbox.ymin * src_size[1])
            x1 = int(obj.bbox.xmax * src_size[0])
            y1 = int(obj.bbox.ymax * src_size[1])
            
            depth_map2 = cv2.resize(depth_map, (640, 480))
            obj.depth = depth_estimator.calculate_average_depth(depth_map2, x0, y0, x1, y1)

            # --- NEW LOGIC: rolling mean of previous 10 depths ---
            history = depth_history_con[obj.track_id]
            if len(history) > 0:
                obj.prev_depth = np.mean(history)
            else:
                obj.prev_depth = obj.depth  # or 0.0 if you prefer
            # -----------------------------------------------------

            history.append(obj.depth)

        ## Tracking and displacement (motion)
        dets = np.array([[o.bbox.xmin, o.bbox.ymin, o.bbox.xmax, o.bbox.ymax, o.conf] for o in objects])
        displacements = {}

        if dets.size and mot_tracker:
            track_data = mot_tracker.update(dets)

            for x0, y0, x1, y1, tid in track_data:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                track_history[tid].append((cx, cy))

                ## Calculate accumulated displacement for each track_id
                centers = list(track_history[tid])
                dx_sum = 0.0
                dy_sum = 0.0
                for i in range(1, len(centers)):
                    prev_cx, prev_cy = centers[i-1]
                    curr_cx, curr_cy = centers[i]
                    dx_sum += (curr_cx - prev_cx) * src_size[0]
                    dy_sum += (curr_cy - prev_cy) * src_size[1]
                displacements[int(tid)] = (dx_sum, dy_sum)

            
            ids_in_frame = set()
            used_tids = set()
            for obj in objects:
                best_iou = 0
                best_track = None
                for x0, y0, x1, y1, tid in track_data:
                    if tid in used_tids:
                        continue
                    iou_val = iou([obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax], [x0, y0, x1, y1])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_track = (x0, y0, x1, y1, tid)
                if best_track is not None:
                    x0, y0, x1, y1, tid = best_track
                    dx, dy = displacements.get(tid, (0.0, 0.0))
                    obj.dx = dx
                    obj.dy = dy
                    obj.track_id = tid
                    obj.score = scorer.compute_object_score(labels.get(obj.id, 'unknown'), obj.depth, dx, dy)
                    used_tids.add(tid)
                    # --- Debug: check for duplicate IDs in the same frame ---
                    if obj.track_id in ids_in_frame:
                        print(f"Warning: Duplicate track_id {obj.track_id} in frame!")
                    ids_in_frame.add(obj.track_id)
                    # -------------------------------------------
            
            objects = [obj for obj in objects if obj.track_id != 0 and obj.score > 0.0]
            
            

        # Update objects and continuous depth history 
        for obj in objects:
            accumulated_objects[obj.track_id] = obj
            depth_history_con[obj.track_id].append(obj.depth)

        # --- ALERT LOGIC ---
        if not alert_given:
            for obj in objects:
                if (obj.score > 0.9 and obj.depth > 100):  # obj.score > 1.0 
                    if len(depth_history[obj.track_id]) == 0:
                        depth_history[obj.track_id].append(obj.depth)
                    else:
                        obj.prev_depth = depth_history[obj.track_id][0]
                        depth_history[obj.track_id].clear()
                        depth_history[obj.track_id].append(obj.depth)

                    description = textgen.generate([obj], labels)[0]

                    last_said_track_ids.append(obj.track_id)
                    last_line.append(description)

                    print(f"{TermColor.HEADER}{TermColor.BOLD}─" * 30 + TermColor.ENDC)
                    print(f"{TermColor.BOLD}FRAME COUNT: {frame_counter}{TermColor.ENDC}")
                    print(f"{TermColor.OKCYAN}[Text Descriptions]{TermColor.ENDC}")
                    

                    print(
                        f"\n{TermColor.BOLD}{TermColor.OKGREEN}{labels.get(obj.id, 'unknown'):>8}{TermColor.ENDC} | "
                        f"score: {TermColor.OKBLUE}{obj.score:6.3f}{TermColor.ENDC} | "
                        f"tid: {TermColor.OKCYAN}{int(obj.track_id):3}{TermColor.ENDC} | "
                        f"depth: {TermColor.OKBLUE}{obj.depth:6.2f}{TermColor.ENDC} | "
                        f"prev_depth: {TermColor.OKBLUE}{obj.prev_depth:6.2f}{TermColor.ENDC} | "
                        f"dx: {TermColor.WARNING}{obj.dx:5.1f}{TermColor.ENDC} | "
                        f"dy: {TermColor.WARNING}{obj.dy:5.1f}{TermColor.ENDC}"
                    )
                    print(f"{TermColor.FAIL}{TermColor.BOLD}Alert: {description}{TermColor.ENDC}")
                    say(f"Alert {description}", interrupt=True, is_alert=True)
                    alert_given = True
                    break  # Only alert once per window


        ### Process every FRAME_COUNT frames
        if frame_counter % FRAME_COUNT == 0 and alert_given == False:

            all_objects = list(accumulated_objects.values())  # objects with unique track_id
            accumulated_objects.clear()  

            sorted_objects = sorted(all_objects, key=lambda o: -o.score)[:5]

            """
            # Update mean depth history
            for obj in sorted_objects:
                history = depth_history_con[obj.track_id]
                if len(history) > 0:
                    mean_depth = np.mean(history)
                    depth_history_mean[obj.track_id].clear()
                    depth_history_mean[obj.track_id].append(mean_depth)
                else:
                    # If no history, clear 
                    depth_history_mean[obj.track_id].clear()

            # Update depth history
            for obj in sorted_objects:
                if len(depth_history[obj.track_id]) == 0:
                    depth_history[obj.track_id].append(depth_history_mean[obj.track_id][0])
                else:
                    obj.prev_depth = depth_history[obj.track_id][0]
                    depth_history[obj.track_id].clear()
                    depth_history[obj.track_id].append(depth_history_mean[obj.track_id][0])
            """


            ## okey untill here
            descriptions = textgen.generate(sorted_objects, labels)

            print(f"{TermColor.HEADER}{TermColor.BOLD}─" * 30 + TermColor.ENDC)
            print(f"{TermColor.BOLD}FRAME COUNT: {frame_counter}{TermColor.ENDC}")
            print(f"{TermColor.OKCYAN}[Text Descriptions]{TermColor.ENDC}")

            for obj in sorted_objects:
                 print(
                    f"{TermColor.OKGREEN}{labels.get(obj.id, 'unknown'):>8}{TermColor.ENDC} | "
                    f"score: {TermColor.OKBLUE}{obj.score:6.3f}{TermColor.ENDC} | "
                    f"tid: {TermColor.OKCYAN}{int(obj.track_id):3}{TermColor.ENDC} | "
                    f"depth: {TermColor.OKBLUE}{obj.depth:6.2f}{TermColor.ENDC} | "
                    f"prev_depth: {TermColor.OKBLUE}{obj.prev_depth:6.2f}{TermColor.ENDC} | "
                    f"dx: {TermColor.WARNING}{obj.dx:5.1f}{TermColor.ENDC} | "
                    f"dy: {TermColor.WARNING}{obj.dy:5.1f}{TermColor.ENDC}"
                )

            # print(f"{TermColor.HEADER}{TermColor.BOLD}─" * 30 + TermColor.ENDC)            

            prev_id = None
            for obj, line in zip(sorted_objects, descriptions):
                if obj.score > 0.8:
                    if obj.track_id in last_said_track_ids:
                        print(f"{TermColor.WARNING}SKIPPED (repeated tid {obj.track_id}): {line}{TermColor.ENDC}")
                        prev_id = obj.id
                        continue
                    elif line in last_line:
                        print(f"{TermColor.WARNING}SKIPPED (repeated line): {line}{TermColor.ENDC}")
                        prev_id = obj.id
                        continue
                    else:
                        if prev_id != obj.id:                        
                            last_said_track_ids.append(obj.track_id)
                            last_line.append(line)
                            print(f"\n{line}")
                            say(line)
                            break
        
        if frame_counter % FRAME_COUNT == 0:
            # Reset alert flag every FRAME_COUNT frames
            alert_given = False

        
        return objects, labels, depth_map
    

    video_writer = None
    depth_writer = None
    if args.use_camera:
        video_writer = cv2.VideoWriter('camera_detection_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 8, (640, 480))
        depth_writer = cv2.VideoWriter('camera_depth_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 8, (640, 480))
        pass
    
    def gs_callback(input_tensor, src_size, mot_tracker):
        # Call the original user_callback
        objects, labels, depth_map = user_callback(input_tensor, src_size, mot_tracker)

        
        common.set_input(interpreter, input_tensor)

        interpreter.invoke()

        # Visualize and write detection output (same as in _run_opencv_video)
        frame = gstbuffer_to_ndarray(input_tensor, (inf_w, inf_h))
        if frame is None:
            print("Frame is None or empty")
            return objects, labels, depth_map 
        
        if (frame.size == 0):
            print("Frame size is 0")
            return objects, labels, depth_map  # or just return
        
        frame_resized = cv2.resize(frame, (640, 480))
        
        vis_frame = frame_resized.copy()
        """
        for obj in objects:
            if obj.conf > 0.5:
                x0 = int(obj.bbox.xmin * src_size[0])
                y0 = int(obj.bbox.ymin * src_size[1])
                x1 = int(obj.bbox.xmax * src_size[0])
                y1 = int(obj.bbox.ymax * src_size[1])
                label = labels.get(obj.id, str(obj.id))
                depth = obj.depth
                dx, dy = obj.dx, obj.dy
                label_text = f"{obj.conf:.2f} {label} d:{depth:.1f}"
                cv2.rectangle(vis_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(vis_frame, label_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                arrow_scale = 1
                cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                start_point = (int(cx - dx * arrow_scale/ 2), int(cy - dy * arrow_scale / 2))
                end_point = (int(cx + dx * arrow_scale / 2), int(cy + dy * arrow_scale / 2))
                cv2.arrowedLine(vis_frame, start_point, end_point, line_type=cv2.LINE_AA, color=(216, 230, 176), thickness=2, tipLength=0.2)
        """

        if video_writer is not None:
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)  
            video_writer.write(vis_frame)

        # Depth output
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        depth_color = cv2.resize(depth_color, (640, 480))
        if depth_writer is not None:
            depth_writer.write(depth_color)

        return objects, labels, depth_map

    if args.use_camera:
        try:
            gstreamer.run_pipeline(
                user_function=gs_callback,
                src_size=(640, 480),
                appsink_size=inference_size,
                trackerName="sort",
                videosrc='/dev/video0',
                videofmt='raw'
            )
        finally:
            if video_writer is not None:
                video_writer.release()
            if depth_writer is not None:
                depth_writer.release()
    
    else:
        # Use OpenCV pipeline for video file
        _run_opencv_video("image_video.mp4", user_callback, (640, 480), inference_size, tracker=tracker, fps_counter=fps_counter)
 
# Launch script
if __name__ == '__main__':
    main()