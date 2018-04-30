import os,time,cv2,argparse,multiprocessing
import numpy as np
from numpy.linalg import inv
from numpy.linalg import multi_dot
from mvnc import mvncapi as mvnc
from skimage.transform import resize
from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
dim=(448,448)
threshold = 0.2
iou_threshold = 0.5
num_class = 20
num_box = 2
grid_size = 7

img_kalman = np.zeros((600,800))
first_capture = 1

#Variables for Kalman Filter - State variable is Four-Vector
# X = [x, y, vx, vy]'
x_hat_minus = np.zeros((4,1)) #Predicted state for the tracked object
x_hat       = np.zeros((4,1)) #Corrected state for the tracked object

sampling_time = 0;            #This time is calculated as time difference betweent two consecutive frames

A_state     = np.array([[1,0,sampling_time,0],[0,1,0,sampling_time],[0,0,1,0],[0,0,0,1]]) #State transition matrix
H_meas      = np.array([[1,0,0,0],[0,1,0,0]])                                             #Only positions can be measured from consective frames

sigma_model = 1 
Q_process   = np.eye(4,4) * (sigma_model**2)       #Process Noise
P_varminus  = np.zeros((4,4))                     #Prediction Covariance        
P_var       = np.zeros((4,4))                     #Computation Covariance

sigma_meas  = 1.4
R_process   = np.eye(2,2) * (sigma_meas**2)       #Process Noise

kalman_estimate = np.zeros((3,1))
estimate = np.zeros((3,1))

def kalman_iter (x_measured, y_measured):
    
    global sampling_time, x_hat, x_hat_minus, P_varminus, P_var
    
    A_state     = np.array([[1,0,sampling_time,0],[0,1,0,sampling_time],[0,0,1,0],[0,0,0,1]])
    
    z_measured = np.array([[x_measured],[y_measured]])
    
    #Prediction State
    x_hat_minus = np.dot(A_state,x_hat)
    P_varminus  = multi_dot([A_state,P_var,A_state.T]) + Q_process
    
    #Gain calulation
    K_gain_intr = multi_dot([H_meas,P_varminus,H_meas.T]) + R_process
    #print("K_gain_intr : \n", K_gain_intr)
    K_gain = multi_dot([P_varminus,H_meas.T,inv(K_gain_intr)])
    
    #Correction State
    state_residual = z_measured - H_meas.dot(x_hat_minus)  
    x_hat = x_hat_minus + K_gain.dot(state_residual)
    P_var = P_varminus.dot((np.eye(4,4) - K_gain.dot(H_meas)))
    #print("Sampling Time :", sampling_time) 
    #print("A_state :\n",A_state)
    print("New state estimate :\n", x_hat)



def show_results(img, results, img_width, img_height):
    global img_kalman, first_capture, estimate, kalman_estimate  
    
    img_cp = img
    #disp_console = False
    disp_console = True
    imshow = True
    for i in range(len(results)):
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3])//2
        h = int(results[i][4])//2
        if disp_console : print ('    class : ' + results[i][0] + ' , [x,y,w,h]=[' + str(x) + ',' + str(y) + ',' + str(int(results[i][3])) + ',' + str(int(results[i][4]))+'], Confidence = ' + str(results[i][5]) )
        xmin = x-w
        xmax = x+w
        ymin = y-h
        ymax = y+h
        if xmin<0:
        	xmin = 0
        if ymin<0:
        	ymin = 0
        if xmax>img_width:
        	xmax = img_width
        if ymax>img_height:
        	ymax = img_height
        if  imshow:
        	cv2.rectangle(img_cp,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        	#print ((xmin, ymin, xmax, ymax))
        	cv2.rectangle(img_cp,(xmin,ymin-20),(xmax,ymin),(125,125,125),-1)
        	cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    cv2.imshow('YOLO detection',img_cp)
#    if first_capture == 1:
#        img_kalman = img
#        first_capture = 2
       
    
    #Kalman Tracking results
    if len(results) > 0:
        if results[0][0] == "person" :
            xlocal = int(results[0][1])
            ylocal = int(results[0][2])
            kalman_iter(xlocal,ylocal)
            f = 595.0
            z = float(frame_d[ylocal][xlocal][0])            
            xd=(xlocal%640 - (640-1)/2) * z / f
            #yd=(xlocal/640 - (480-1)/2) * z / f
            yd=(ylocal/480 - (480-1)/2) * z / f
            
            zkalman = float(frame_d[int(x_hat[1])][int(x_hat[0])][0])
            xdkalman=(int(x_hat[0])%640 - (640-1)/2) * zkalman / f
            #ydkalman=(int(x_hat[0])/640 - (480-1)/2) * zkalman / f
            ydkalman=(int(x_hat[1])/480 - (480-1)/2) * zkalman / f
            
#           cv2.line(img_kalman,(x_hat[0][1],x_hat[1][1]),(x_hat[0][1],x_hat[1][1]),(255,0,0),5)
#           cv2.line(img_kalman,(int(results[i][1]),int(results[i][2])),(int(results[i][1]),int(results[i][2])),(0,255,0),5)
            #cv2.line(img_kalman,(x_hat[0],x_hat[1]),(x_hat[0],x_hat[1]),(255,0,0),5)
            #cv2.line(img_kalman,(int(results[0][1]),int(results[0][2])),(int(results[0][1]),int(results[0][2])),(0,255,0),5)
            #cv2.imshow('Kalman & Object Detection',img_kalman)
            print("Estimate [x y z]", xd,yd,z)            
            estimate = np.append(estimate,[[xd],[yd],[z]],axis=1)            
            
            print("Kalman [x y z]", xdkalman,ydkalman,zkalman)            
            kalman_estimate = np.append(kalman_estimate,[[xdkalman],[ydkalman],[zkalman]],axis=1)
            
    cv2.waitKey(1)

#Check this function in detail
def interpret_output(output, img_width, img_height):
    w_img = img_width
    h_img = img_height
    probs = np.zeros((7,7,2,20))
    class_probs = (np.reshape(output[0:980],(7,7,20)))
    #print("Class_probs",class_probs)    
    scales = (np.reshape(output[980:1078],(7,7,2)))
    #print(scales)
    boxes = (np.reshape(output[1078:],(7,7,2,4)))
    offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))
    #boxes.setflags(write=1)
    boxes[:,:,:,0] += offset
    boxes[:,:,:,1] += np.transpose(offset,(1,0,2))
    boxes[:,:,:,0:2] = boxes[:,:,:,0:2] / 7.0
    boxes[:,:,:,2] = np.multiply(boxes[:,:,:,2],boxes[:,:,:,2])
    boxes[:,:,:,3] = np.multiply(boxes[:,:,:,3],boxes[:,:,:,3])

    boxes[:,:,:,0] *= w_img
    boxes[:,:,:,1] *= h_img
    boxes[:,:,:,2] *= w_img
    boxes[:,:,:,3] *= h_img

    for i in range(2):
    	for j in range(20):
    		probs[:,:,i,j] = np.multiply(class_probs[:,:,j],scales[:,:,i])
    #print (probs)
    filter_mat_probs = np.array(probs>=threshold,dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
    	if probs_filtered[i] == 0 : continue
    	for j in range(i+1,len(boxes_filtered)):
    		if iou(boxes_filtered[i],boxes_filtered[j]) > iou_threshold :
    			probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered>0.0,dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
    	result.append([classes[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

    return result

def iou(box1,box2):
	tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
	lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
	if tb < 0 or lr < 0 : intersection = 0
	else : intersection =  tb*lr
	return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

def worker(graph, input_q, output_q):
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        graph.LoadTensor(resize(frame/255.0,dim,1)[:,:,(2,1,0)].astype(np.float16), 'user object')
        out, userobj = graph.GetResult()
        results = interpret_output(out.astype(np.float32), frame.shape[1], frame.shape[0])
        output_q.put((frame, results, frame.shape[1], frame.shape[0]))
        #output_q.put((frame, [], frame.shape[1], frame.shape[0]))
        #output_q.put(frame)
    #
    fps.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    # configuration NCS
    network_blob = 'graph'
    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 0)
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
    	print('No devices found')
    	quit()
    device = mvnc.Device(devices[0])
    device.OpenDevice()
    opt = device.GetDeviceOption(mvnc.DeviceOption.OPTIMISATION_LIST)
    # load blob
    with open(network_blob, mode='rb') as f:
    	blob = f.read()
    graph = device.AllocateGraph(blob)
    graph.SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
    iterations = graph.GetGraphOption(mvnc.GraphOption.ITERATIONS)
    #
    pool = Pool(args.num_workers, worker, (graph, input_q, output_q))
    #
    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).stop()
    fps = FPS().start()
    
    global sampling_time, estimate, kalman_estimate
     
    while True:  # fps._numFrames < 120
        for itr in range(30,50):        
            frame = cv2.imread('../dataset1/img/image%s.png' %itr)
            frame_d=cv2.imread('../dataset1/img_d/image%s.png' %itr)
            #print('Lengths of files iteration no %s:' % itr,np.shape(frame), np.shape(frame_d))            
            input_q.put(frame)
            t = time.time()
            (img, results, img_width, img_height) = output_q.get()
            sampling_time = time.time() - t
            show_results(img, results, img_width, img_height)
            #cv2.imshow('Video', output_q.get())
            #cv2.imshow('Video', output_q.get())
            fps.update()
            #print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        
        print("Kalman Estimate size:", np.shape(kalman_estimate))
        print("Kalman Estimate :", kalman_estimate)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        constant_value = 3;
        #ax.plot3D(kalman_estimate[0,:], kalman_estimate[2,:], kalman_estimate[1,:], 'green')
        ax.plot3D(kalman_estimate[0,:], kalman_estimate[2,:], constant_value, 'red')        
        #ax.plot3D(estimate[0,:], estimate[2,:], estimate[1,:], 'gray')
        ax.plot3D(estimate[0,:], estimate[2,:], constant_value, 'gray')        
        plt.xlabel("X Axis")        
        plt.ylabel("Z Axis")        
        #plt.zlabel("Y Axis")        
        #np.savetxt('kalman_estimate.out', kalman_estimate,delimiter=',', newline='\n', fmt='%10.2f')    
        #np.savetxt('estimate.out', estimate,delimiter=',', newline='\n', fmt='%10.2f')
        plt.show()
        
        file = open("kalman_estimate.out",'w')
        file1 = open("estimate.out",'w')        
        file.write("\n".join(str(elem) for elem in kalman_estimate))
        file1.write("\n".join(str(elem) for elem in estimate))        
        file.close()
        file1.close()
        cv2.waitKey(0)        
        break        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
    graph.DeallocateGraph()
    device.CloseDevice()

#file.write("\n".join(str(elem) for elem in a))
