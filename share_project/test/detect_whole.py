import argparse

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-3cls.cfg', help='*.cfg path')
parser.add_argument('--names', type=str, default='data/daodan.names', help='*.names path')
# parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
parser.add_argument('--weights', type=str, default='weights/best.pt', help='weights path')
# parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
parser.add_argument('--source', type=str, default='data/daodan/val', help='source')

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
opt = parser.parse_args()
def detect_whole(im0,model,save_path,x_top=0,y_top=0):
    a = []
    device = torch_utils.select_device('0')
    # img = letterbox(img0, new_shape=self.img_size)[0]
    img = im0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    img_ = torch.zeros((1, 3, 512, 512), device=device)  # init img
    # _ = model(img_.float())
    img = torch.from_numpy(img).to(device)
    img =img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    
    t1 = torch_utils.time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    t2 = torch_utils.time_synchronized()
   

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

   
    # Process detections
    t5 = time.time()
    for i, det in enumerate(pred):  
       
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] 
        if det is not None and len(det):
            # Rescale boxes from imgsz to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in det:
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                #         file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                # if save_img or view_img:  # Add bbox to imagey_top
                #     label = '%s %.2f' % (names[int(cls)], conf)
                #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                
                label = '%s %.2f' % (names[int(cls)], conf)
                c1, c2 = (int(xyxy[0])+x_top, int(xyxy[1])+y_top), (int(xyxy[2])+x_top, int(xyxy[3])+y_top)
                a.append((label,c1,c2))
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])


        # Print time (inference + NMS)
        # print('%sDone. (%.3fs)' % (s, t2 - t1))

       

        # Save results (image with detections)
        # if save_img:
        #     if dataset.mode == 'images':
      
        cv2.imwrite(save_path, im0)
    return a
       
# def detect_all(img_dir,weights):
#     weights = 'weights/best.pt'
#     for root_dir, dir1, path in os.walk(img_dir):
#         for p in path:
#             if p.endswith('tif'):
#                 image_path  = os.path.join(root_dir, p)
#                 img0 = cv2.imread(image_path)
#                 new_image_name = p[:-4]+ '_'+ image_path.split('/')[-2]+ '.tif'
#                 new_image_path = os.path.join('/home/zoucg/data/all_data_detect',new_image_name)
#                 detect_whole(img0,weights,new_image_path)


def main():
    image_path= '/home/zoucg/cv_project/yolov3/data/daodan/val/japan12_20190123_19__7680__7680.jpg'
    weights = 'weights/1_or_best.pt'
    save_path = './test.jpg'
    img0 = cv2.imread(image_path)
    detect_whole(img0,weights,save_path)


    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-3cls.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/daodan.names', help='*.names path')
    # parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='weights path')
    # parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='data/daodan/val', help='source')
    
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
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)
    main()

    # with torch.no_grad():
    #     detect()
