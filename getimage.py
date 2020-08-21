import pyzed.sl as sl
import cv2
import os
import sys

def main():
    
    #argv stand for class number
    class_num = int(sys.argv[1])

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Capture 50 frames and stop
    i = 0
    image = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()
    while i < 1:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image, sl.VIEW.LEFT)
            #cv2.imshow("ZED", image.get_data())
            cv2.imwrite('/home/xcy/workspace/read.jpg', image.get_data())
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get the timestamp at the time the image was captured
            print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
                  timestamp.get_milliseconds()))
            i = i + 1
            #cv2.waitKey(0)


    # Close the camera
    zed.close()
    
    #run the detection program
    os.system('python3 detect.py --source /home/xcy/workspace/read.jpg --targetclass %s' %(class_num))  #'/home/xcy/workspace/read.jpg'

if __name__ == "__main__":
    main()
