#!/usr/bin/env python3
#coding:utf-8

import rospy
import std_msgs.msg
import geometry_msgs.msg

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import math
import numpy as np
import os
import time

import pyrealsense2 as rs
import cv2

# initialize apriltag module
from scipy.spatial.transform import Rotation as Rota
from pupil_apriltags import Detector


class DetectObject:
    def __init__(self):
        self.targetclass = opt.targetclass
        self.__target_class_sub_ = rospy.Subscriber('detect_item', std_msgs.msg.Int32, self.targetClassCB, queue_size=5)
        self.__target_position_pub_ = rospy.Publisher('detect_item_result', geometry_msgs.msg.PointStamped,
                                                      queue_size=5)
        # self.__apriltag_switch_sub_ = rospy.Subscriber('apriltag_switch', std_msgs.msg.Bool, self.apriltag_switch_,
        #                                                queue_size=5)
        # self.__target_position_pub_ = rospy.Publisher('apriltag_pose_result', geometry_msgs.msg.TransformStamped,
        #                                               queue_size=5)
        # TODO: Wish to fix
        self.__pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # Start streaming
        self.__pipe_profile = self.__pipeline.start(config)

    def __del__(self):
        # Close the camera
        self.__pipeline.stop()

    def targetClassCB(self, msg):
        # receive target class index and start to detect
        self.targetclass = msg.data
        rospy.loginfo('got target class index:%d', self.targetclass)
        with torch.no_grad():
            self.__detect()
        return

    def __detect(self, save_img=False):
        # Declare pointcloud object, for calculating pointclouds and texture mappings
        pc = rs.pointcloud()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        frames = self.__pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # time.sleep(8)
        img_color = np.array(color_frame.get_data())
        img_depth = np.array(depth_frame.get_data())
        cv2.imwrite(opt.source, img_color)

        # detection section
        img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
        out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # Initialize model
        model = Darknet(opt.cfg, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Eval mode
        model.to(device).eval()

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Export mode
        if ONNX_EXPORT:
            model.fuse()
            img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
            f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
            torch.onnx.export(model, img, f, verbose=False, opset_version=11)

            # Validate exported model
            import onnx
            model = onnx.load(f)  # Load the ONNX model
            onnx.checker.check_model(model)  # Check that the IR is well formed
            print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
            return

        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=img_size)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=img_size)

        # Get names and colors
        names = load_classes(opt.names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        _ = model(torch.zeros((1, 3, img_size, img_size), device=device)) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            t2 = torch_utils.time_synchronized()

            # to float
            if half:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                       multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                position_result=geometry_msgs.msg.PointStamped()
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # stop_num = 0
                    # print('c = ',len(det))
                    target_detected=False
                    # Write results
                    for *xyxy, conf, cls in det:
                        # c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        # c0 = ((int(xyxy[0])+int(xyxy[2]))/2, (int(xyxy[1])+int(xyxy[3]))/2)
                        if save_txt:  # Write to file
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                        # print(c0)

                        if int(cls) == self.targetclass:
                            x_center = int((int(xyxy[0])+int(xyxy[2]))/2)
                            y_center = int((int(xyxy[1])+int(xyxy[3]))/2)

                            # Intrinsics and Extrinsics
                            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                            depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)

                            # Depth scale: units of the values inside a depth frame, how to convert the value to units of 1 meter
                            depth_sensor = self.__pipe_profile.get_device().first_depth_sensor()
                            depth_scale = depth_sensor.get_depth_scale()

                            # Map depth to color
                            depth_pixel = [240, 320]   # Random pixel
                            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)

                            color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
                            color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)

                            pc.map_to(color_frame)
                            points = pc.calculate(depth_frame)
                            vtx = np.array(points.get_vertices())
                            tex = np.array(points.get_texture_coordinates())
                            pix_num = 1280 * y_center + x_center


                            point_cloud_value = [np.float(vtx[pix_num][0]),np.float(vtx[pix_num][1]),np.float(vtx[pix_num][2])]
                            print('point_cloud_value:',point_cloud_value,names[int(cls)],int(cls))
                            if np.float(vtx[pix_num][2])>0.11:
                                position_result.header.frame_id='camera_color_optical_frame'
                                position_result.point.x=np.float(vtx[pix_num][0])
                                position_result.point.y=np.float(vtx[pix_num][1])
                                position_result.point.z=np.float(vtx[pix_num][2])
                                self.__target_position_pub_.publish(position_result)#publish the result
                                target_detected=True
                                rospy.loginfo('The target has been detected!')
                                break   # only the target class
                                # os.system('rostopic pub -1 /goal_pose geometry_msgs/PointStamped [0,[0,0],zed_left_camera_optical_frame] [%s,%s,%s]' %(np.float(vtx[pix_num][0]),np.float(vtx[pix_num][1]),np.float(vtx[pix_num][2]))
                                # os.system('rostopic pub -1 /start_plan std_msgs/Bool 1')
                            # else:
                                # os.system('rostopic pub -1 /start_plan std_msgs/Bool 0')
                                # print("Can't estimate point cloud at this position.")
                        # stop_num += 1
                        # if stop_num >= len(det):
                        #     os.system('rostopic pub -1 /start_plan std_msgs/Bool 0')
                    if not target_detected:
                        position_result.header.frame_id='empty'
                        self.__target_position_pub_.publish(position_result)  # publish failure topic
                        rospy.logwarn('Fail to detect the target!')
                else:
                    position_result.header.frame_id='empty'
                    self.__target_position_pub_.publish(position_result)  # publish failure topic
                    rospy.logwarn('Fail to detect any target!')

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + out + ' ' + save_path)

        print('Done. (%.3fs)' % (time.time() - t0))
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--targetclass', type=int, default=0, help='targetclass')  # input class which needs to be detected
    opt = parser.parse_args()
    print(opt)
    rospy.init_node('detect_node')
    detect_node=DetectObject()
    rospy.loginfo('detect_node is activated.')
    rospy.spin()

