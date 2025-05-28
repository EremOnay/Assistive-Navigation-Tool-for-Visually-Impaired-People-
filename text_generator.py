import numpy as np

class TextGenerator:
    def __init__(self, score_threshold=0.90):
        self.score_threshold = score_threshold
        
        self.speed_map = {
            "slow": (0.0, 15.0),
            "at moderate speed": (15.0, 40.0),
            "fast": (40.0, float('inf')),
        }

        self.distance_map = {
            "far": (0, 30),
            "at moderate distance": (30, 100),
            "close": (100, 200),
            "very close": (200, 255),
        }

    def _map_value(self, val, mapping):
        for k, (low, high) in mapping.items():
            if low <= val < high:
                return k
        return list(mapping.keys())[-1]

    def _direction(self, dx, dy):
        if abs(dx) < 10:
            return None  # not moving
        return "right" if dx > 0 else "left" 

    def _movement_trend(self, current_depth, previous_depth):
        delta = current_depth - previous_depth
        if abs(delta) < 15: # old version was 20
            return "holding"
        return "approaching" if delta > 0 else "moving away"
    
    def _side(self, obj, frame_width=640):
        # Assume obj has bbox.xmin and bbox.xmax as normalized (0-1)
        center_x = (obj.bbox.xmin + obj.bbox.xmax) / 2
        if center_x < 0.33:  
            return  "on the left" # "left" 
        elif center_x > 0.66:
            return "on the right" # "right"
        else:
            return "in the center" # "center"

    def generate(self, objects, labels):
        descriptions = []

        for obj in objects:
            name = labels.get(obj.id, "unknown")
            depth = obj.depth
            prev_depth = getattr(obj, "prev_depth", depth)
            dx = getattr(obj, "dx", 0.0)
            dy = getattr(obj, "dy", 0.0)
            score = obj.score

            distance_text = self._map_value(depth, self.distance_map)

            direction = self._direction(dx, dy)

            trend = self._movement_trend(depth, prev_depth)

            if trend == "holding":
                speed = np.linalg.norm([dx, dy])
            else:
                speed = abs(depth - prev_depth) / 1 # 1 can be adjusted

            speed_text = self._map_value(speed, self.speed_map)

            side = self._side(obj)
                
            if direction is None and trend == "holding":
                sentence = f"A {name} is {distance_text} and stationary {side}."
            elif trend == "holding":
                sentence = f"A {name} is {distance_text}, moving {direction} {speed_text} {side}."
            else:
                sentence = f"A {name} is {distance_text} {trend} {speed_text} {side}."

            descriptions.append((score, sentence))

        descriptions.sort(key=lambda x: -x[0])
        top_sentences = [s for _, s in descriptions[:5]]

        return top_sentences

    def generate_warning_for_first(self, objects, labels):
        if not objects:
            return []

        obj = objects[0]
        name = labels.get(obj.id, "unknown")
        depth = obj.depth
        prev_depth = getattr(obj, "prev_depth", depth)
        dx = getattr(obj, "dx", 0.0)
        dy = getattr(obj, "dy", 0.0)

        distance_text = self._map_value(depth, self.distance_map)
        speed = np.linalg.norm([dx, dy])
        speed_text = self._map_value(speed, self.speed_map)
        direction = self._direction(dx, dy)
        trend = self._movement_trend(depth, prev_depth)

        if direction is None:
            sentence = f"A {name} is {distance_text} stationary."
        elif trend == "holding":
            sentence = f"A {name} is {distance_text}, moving {direction} at {speed_text} speed."
        else:
            sentence = f"A {name} is {distance_text}, {trend}, at {speed_text} speed."
            
        return ["WARNING", sentence]
