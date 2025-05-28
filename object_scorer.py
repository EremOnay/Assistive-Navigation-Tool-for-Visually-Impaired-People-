import numpy as np

class ObjectScorer:
    def __init__(self):
        self.object_importance = {
			"person": 0.95, 
			"car": 1.0,
			"motorcycle": 1.0,
			"bicycle": 0.90,
			"dog": 0.8,
			"chair": 0.9, #just for trial, old version was 0.5
			"table": 0.4,        # Not in your label set, can remove if not used
			"tree": 0.3,         # Kept as you requested
			#"background": 0.1,
			"tv": 0.6,
			"cup": 0.3,
			"laptop": 0.7,
			"bed": 0.6,
			"backpack": 0.5,
			"couch": 0.6,
			#"handbag": 0.4,
			"dining table": 0.4,
			"traffic light": 0.8,
			"stop sign": 0.8,
			#"oven": 0.3,
			#"microwave": 0.3,
			#"toilet": 0.2,
			#"refrigerator": 0.3,
			#"knife": 0.2,
			"cat": 0.8,
			"bus": 0.85,
			#"book": 0.3,           # Optional extra if needed
			#"remote": 0.4,         # Optional extra
			#"keyboard": 0.4        # Optional extra
		}

    def compute_depth_score(self, depth_val, max_depth_val=255.0):
        norm_depth = 1 - (depth_val / max_depth_val) # one minus since bigger values are closer
        return 6 / (1 + (norm_depth * 5.0) ** 2)  # stronger near weighting

    def compute_motion_score(self, dx, dy, depth, prev_depth, alpha=0.1):
        depth_diff = abs(depth - prev_depth)
        if depth_diff < 15:
            flow_vector = np.array([dx, dy])
            speed = np.linalg.norm(flow_vector)
            base_score = 1 / (1 + np.exp(-alpha * speed))
            motion_score = base_score * 1.5
            print(f"motion_score: {motion_score}, dx: {dx}, dy: {dy}")
            return motion_score
        else: 
            base_score = 1 / (1 + np.exp(-alpha * depth_diff))
            motion_score = base_score * 1.5
            print(f"dx: {dx}, dy: {dy}, depth_motion_score: {motion_score}, depth_diff: {depth_diff}")
            return motion_score
       

    def compute_object_score(self, class_name, depth_val, dx, dy, prev_depth=None):
        w1, w2, w3 = 0.4, 0.3, 0.3 # weights for importance, depth, and motion
        speed = np.linalg.norm([dx, dy])

        I = self.object_importance.get(class_name, 0.1) / max(self.object_importance.values())
        D = self.compute_depth_score(depth_val)

        if prev_depth is None:
            prev_depth = depth_val
        M = self.compute_motion_score(dx, dy, depth_val, prev_depth)


        final_score = w1 * I + w2 * D + w3 * M
        return round(final_score, 3)

    def rank_objects(self, objects, labels, displacements, top_n=3): 
        scored_objects = []
        for obj in objects:
            class_name = labels.get(obj.id, "unknown")
            dx, dy = displacements.get(obj.id, (0.0, 0.0))
            score = self.compute_object_score(class_name, obj.depth, dx, dy)
            obj.score = score 
        return sorted(objects, key=lambda o: -o.score)[:top_n]


# ---------- MAIN TESTING ----------

if __name__ == "__main__":
    scorer = ObjectScorer()

    # Sample test cases (as would be used from detect.py)
    test_objects = [
        {"name": "person", "depth": 10, "dx": 5.0, "dy": 2.0,},
        {"name": "person", "depth": 20, "dx": 5.0, "dy": 2.0},
        {"name": "person", "depth": 50, "dx": 5.0, "dy": 2.0},
        {"name": "person", "depth": 120, "dx": 5.0, "dy": 2.0},
        {"name": "person", "depth": 150, "dx": 5.0, "dy": 2.0},
        {"name": "person", "depth": 180, "dx": 5.0, "dy": 2.0},
        {"name": "person", "depth": 200, "dx": 5.0, "dy": 2.0},
        #{"name": "car", "depth": 35, "dx": 3.0, "dy": 2.0},
        #{"name": "motorcycle", "depth": 35, "dx": 3.0, "dy": 2.0},
        {"name": "person", "depth": 50, "dx": 10.0, "dy": 10.0},
        {"name": "person", "depth": 50, "dx": 20.0, "dy": 10.0},
        {"name": "person", "depth": 50, "dx": 60.0, "dy": 10.0},
        {"name": "person", "depth": 50, "dx": 60.0, "dy": 10.0},
        {"name": "person", "depth": 50, "dx": 20.0, "dy": 10.0},
        {"name": "car", "depth": 50, "dx": 60.0, "dy": 10.0},
        {"name": "dog", "depth": 50, "dx": 60.0, "dy": 10.0},
        {"name": "traffic light", "depth": 50, "dx": 60.0, "dy": 10.0},
        {"name": "stop sign", "depth": 50, "dx": 60.0, "dy": 10.0},
        {"name": "tree", "depth": 50, "dx": 60.0, "dy": 10.0},

        #{"name": "tv", "depth": 80, "dx": 0.5, "dy": -0.5},
        #{"name": "tv", "depth": 80, "dx": 0.5, "dy": -0.5},
        #{"name": "cup", "depth": 15, "dx": 1.0, "dy": 1.0},

        {"name": "person", "depth": 50, "dx": 10.0, "dy": 10.0, "prev_depth": 30},
        {"name": "person", "depth": 50, "dx": 20.0, "dy": 10.0, "prev_depth": 50},
        #{"name": "person", "depth": 50, "dx": 60.0, "dy": 10.0, "prev_depth": 50}
    ]

    print("Testing Object Scorer with sample inputs:")
    for obj in test_objects:
        score = scorer.compute_object_score(
            obj["name"], obj["depth"], obj["dx"], obj["dy"], obj.get("prev_depth")
        )
        print(f"Object: {obj['name']} | Depth: {obj['depth']} | dx/dy: ({obj['dx']}, {obj['dy']}) => Score: {score}")
