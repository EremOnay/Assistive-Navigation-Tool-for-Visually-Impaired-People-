# sample multiprocessing architecture which is for both yoloLukas and depthEstimation run

p1 = multiprocessing.Process(target=object_detection, args=(queue,))
p2 = multiprocessing.Process(target=depth_estimation, args=(queue,))

p1.start()
p2.start()

p1.join()
p2.join()
