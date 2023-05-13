import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# danger = False
# first = False
cap = cv2.VideoCapture("C:\\Coding\\python_programming\\Projects\\bridge suicide\\Scene.mp4")
def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter("C:\\Coding\\python_programming\\Projects\\bridge suicide\\Sceneoutput.avi" , fourcc, 30.0, (640,352))       
area_1 = [(0,143),(631,145),(614,160),(0,156)]
area_2 = [(0,143-20),(631,145-20),(614,160+20),(0,156+20)]
area2 = {}
inframe = []
fallen = {}
log = {}
# area1=set()
cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)
f=0
from deep_sort_realtime.deepsort_tracker import DeepSort
movement = {}
# object_tracker = DeepSort(max_age=90,
#                           n_init=2,
#                           nms_max_overlap=0.5,
#                           max_cosine_distance=0.3,
#                           nn_budget=None,
#                           override_track_class=None,
#                           embedder="mobilenet",
#                           half=True,
#                           bgr=True,
#                           embedder_gpu=True,
#                           embedder_model_name=None,
#                           embedder_wts=None,
#                           polygon=False,
#                           today=None)
object_tracker = DeepSort(max_age=30,
                          n_init=5,
                          nms_max_overlap=0.5,
                          max_cosine_distance=0.12,
                          nn_budget=None,
                          override_track_class=None,
                          embedder="mobilenet",
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None)

import time


while cap.isOpened():
    ret, frame = cap.read()
    if(ret==False) :
        print("Video ended or errored occured")
        break
    print(f)

    f+=1
    start = time.perf_counter()
    frame = cv2.resize(frame,(640,352))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    
        
    
       
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # cv2.polylines(image,[np.array(area_2,np.int32)],True,(0,255,0),0)
    try:
        det_results=model(image)
        # print(det_results.pandas().xyxy[0])
        detections=[]
        for index,row in det_results.pandas().xyxy[0].iterrows():
            # print(row)
            x1=int(row[ 'xmin' ])
            y1=int (row[ 'ymin' ])
            x2=int(row[ 'xmax' ])
            y2=int (row[ 'ymax' ])
            conf = row['confidence']
            b=str(row['name'])
            if 'person' in b:
                detections.append(([x1,y1,int(x2-x1),int(y2-y1)],conf,b))
            
        tracks = object_tracker.update_tracks(detections,frame=image)
        inframe.clear()
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            inframe.append(track_id)
            bbox = ltrb
            if any([True for k,v in movement.items() if k == track_id]):
                # change in x
                if movement[track_id][2]=='started':
                    if int((bbox[0]+bbox[2])/2)>movement[track_id][0]:
                        movement[track_id][2]='right'
                        movement[track_id][0]=int((bbox[0]+bbox[2])/2)
                        movement[track_id][5][1]=int((bbox[0]+bbox[2])/2)+10
                        movement[track_id][5][0]=int((bbox[0]+bbox[2])/2)-10
                    elif int((bbox[0]+bbox[2])/2)<movement[track_id][0]:
                        movement[track_id][2]='left'
                        movement[track_id][0]=int((bbox[0]+bbox[2])/2)
                        movement[track_id][5][1]=int((bbox[0]+bbox[2])/2)+10
                        movement[track_id][5][0]=int((bbox[0]+bbox[2])/2)-10
                else:
                    if movement[track_id][2]=='left':
                        if int((bbox[0]+bbox[2])/2)>movement[track_id][5][1]:
                            movement[track_id][2]='right'
                            movement[track_id][0]=int((bbox[0]+bbox[2])/2)
                            movement[track_id][4]=movement[track_id][4]+1
                            movement[track_id][5][1]=int((bbox[0]+bbox[2])/2)+10
                            movement[track_id][5][0]=int((bbox[0]+bbox[2])/2)-10
                        elif int((bbox[0]+bbox[2])/2)<movement[track_id][5][0]:
                            movement[track_id][0]=int((bbox[0]+bbox[2])/2)
                            movement[track_id][5][1]=int((bbox[0]+bbox[2])/2)+10
                            movement[track_id][5][0]=int((bbox[0]+bbox[2])/2)-10
                        else:
                            movement[track_id][0]=int((bbox[0]+bbox[2])/2)
                    else:
                        if int((bbox[0]+bbox[2])/2)<movement[track_id][5][0]:
                            movement[track_id][2]='left'
                            movement[track_id][0]=int((bbox[0]+bbox[2])/2)
                            movement[track_id][4]=movement[track_id][4]+1
                            movement[track_id][5][1]=int((bbox[0]+bbox[2])/2)+10
                            movement[track_id][5][0]=int((bbox[0]+bbox[2])/2)-10
                        elif int((bbox[0]+bbox[2])/2)>movement[track_id][5][1]:
                            movement[track_id][0]=int((bbox[0]+bbox[2])/2)
                            movement[track_id][5][1]=int((bbox[0]+bbox[2])/2)+10
                            movement[track_id][5][0]=int((bbox[0]+bbox[2])/2)-10
                        else:
                            movement[track_id][0]=int((bbox[0]+bbox[2])/2)
                # change in y
                if movement[track_id][3]=='started':
                    if int(bbox[3])>movement[track_id][1]:
                        movement[track_id][3]='down'
                        movement[track_id][1]=int(bbox[3])
                        movement[track_id][6][1]=int(bbox[3])+10
                        movement[track_id][6][0]=int(bbox[3])-10
                    elif int(bbox[3])<movement[track_id][1]:
                        movement[track_id][3]='up'
                        movement[track_id][1]=int(bbox[3])
                        movement[track_id][6][1]=int(bbox[3])+10
                        movement[track_id][6][0]=int(bbox[3])-10
                else:
                    if movement[track_id][3]=='up':
                        if int(bbox[3])>movement[track_id][6][1]:
                            movement[track_id][3]='down'
                            movement[track_id][1]=int(bbox[3])
                            movement[track_id][7]=movement[track_id][7]+1
                            movement[track_id][6][1]=int(bbox[3])+10
                            movement[track_id][6][0]=int(bbox[3])-10
                        elif int(bbox[3])<movement[track_id][6][0]:
                            movement[track_id][1]=int(bbox[3])
                            movement[track_id][6][1]=int(bbox[3])+10
                            movement[track_id][6][0]=int(bbox[3])-10
                        else:
                            movement[track_id][1]=int(bbox[3])
                    else:
                        if int(bbox[3])<movement[track_id][6][0]:
                            movement[track_id][3]='up'
                            movement[track_id][1]=int(bbox[3])
                            movement[track_id][7]=movement[track_id][7]+1
                            movement[track_id][6][1]=int(bbox[3])+10
                            movement[track_id][6][0]=int(bbox[3])-10
                        elif int(bbox[3])>movement[track_id][6][1]:
                            movement[track_id][1]=int(bbox[3])
                            movement[track_id][6][1]=int(bbox[3])+10
                            movement[track_id][6][0]=int(bbox[3])-10
                        else:
                            movement[track_id][1]=int(bbox[3])
            else:
                movement.update({track_id: [int((bbox[0]+bbox[2])/2),int(bbox[3]),'started','started',0,[int((bbox[0]+bbox[2])/2)-10,int((bbox[0]+bbox[2])/2)+10],[int(bbox[3])-10,int(bbox[3])+10],0,time.perf_counter(),f]})
            
            if(movement[track_id][1]<=160):
                if any([True for k,v in area2.items() if k == track_id]):
                    pass
                else:
                    area2.update({track_id : ['danger',time.perf_counter(),0,f]})
                    log.update({track_id : ['danger']})

            if any([True for k,v in area2.items() if k == track_id]):
                if(movement[track_id][1]<=180):
                  cv2.putText(image,"Person with ID: "+str(track_id)+" is suspicious",(50,300), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
                  area2[track_id][2]=f-area2[track_id][3]
                  cv2.putText(image,"Person with ID: "+str(track_id)+" is in danger for "+str(round(area2[track_id][2]/30,2)),(50,320), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
                #   print("Person with ID: "+str(track_id)+" is suspicious")
                #   print("Person with ID: "+str(track_id)+" is in danger for "+str(area2[track_id][2]))
                else:
                    cv2.putText(image,"Person with ID: "+str(track_id)+" is at safe postition now",(50,320), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)  
                    print("Person with ID: "+str(track_id)+" is at safe postition now")
                    del area2[track_id]
                    log[track_id][0]='safe'
            
            cv2.rectangle(image,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255 if movement[track_id][1]<210 else ((85 if movement[track_id][4]>10 else (movement[track_id][4]*85)/10)+(85 if movement[track_id][7]>10 else (movement[track_id][7]*85)/10)+(40 if((f-movement[track_id][9])/30>300) else ((f-movement[track_id][9])/30)*40/300))),2)
            mul = ((85 if movement[track_id][4]>10 else (movement[track_id][4]*85)/10)+(85 if movement[track_id][7]>10 else (movement[track_id][7]*85)/10)+(40 if((f-movement[track_id][9])/30>300) else ((f-movement[track_id][9])/30)*40/300))
            prob = round(mul/255,2)
            cv2.putText(image,"Prob "+str(prob),(int(bbox[0]),int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,mul),2)
            # cv2.putText(image,"XChange: "+str(movement[track_id][4]),(int(bbox[0]),int(bbox[1] - 25)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            # cv2.putText(image,"YChange: "+str(movement[track_id][7]),(int(bbox[0]),int(bbox[1] - 35)), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
            # result=cv2.pointPolygonTest(np.array(area_2),(movement[track_id][0],movement[track_id][1]),False)
            # if(result>0):
            #     cv2.putText("ID: "+str(track_id)+" is in danger zone",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            
        # for dan in inframe:
        #     if any([True for k,v in area2.items() if k == dan]):

        for k,v in area2.items():
            flag = False
            for l in inframe:
                if(l==k):
                    area2[k][3]=0
                    flag = True
                    break
            if(flag):
                continue
            else:
                if(area2[k][3]>30):
                    print("Person with id {} has commited suicide".format(k))
                    fallen.update({k:[area2[k][0],area2[k][1],area2[k][2],area2[k][3]]})
                    log[track_id][0] = 'suicide'
                    del area2[k]
                else:
                    area2[k][3]+=1

        
        end = time.perf_counter()
        totalTime = end - start
        fps = 1 / totalTime

        cv2.putText(image,"FPS: "+str(fps),(10,10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        # for box_id in boxes_ids:
        #     x,y,w,h,id=box_id
        #     # cv2.rectangle(image,(x,y),(w,h),(255,0,255),2)
        #     # cv2.putText(image,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0,),2)
        #     if(danger==True):
        #         if(h<227+50 and h>227-50):
        #             pass
        #         else:
        #             if(h>227+50):
        #                 danger=False
        #                 cv2.putText(image,  "Person has exited danger place", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
        #                 print("Person has exited danger place")
                       
    except:
        end = time.perf_counter()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.polylines(image,[np.array(area_1,np.int32)],True,(0,255,0),0)
        cv2.putText(image,"FPS: "+str(fps),(10,10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        video.write(image)
        cv2.imshow('FRAME', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        continue
        
        
       
        
          
        # cv2.putText(image, str(rheel), 
        #                    tuple(np.multiply(rheel, [640, 352]).astype(int)), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        #                         )
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                         mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
        #                         mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
        #                          )               

        

    # if(danger==True):
    #     cv2.putText(image,  "Some one is standing on railing", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
    # if(danger==False and first==True):
    #     cv2.putText(image,  "Person has exited danger place", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
        
    #     cv2.polylines(image,[np.array(area_1,np.int32)],True,(0,255,0),2)
    #     if(danger==False):
    #         print("Person has entered danger place")
    #         first=True
    #     danger=True
    cv2.polylines(image,[np.array(area_1,np.int32)],True,(0,255,0),0)
    cv2.imshow('FRAME', image)
    video.write(image)
    height, width, channels = image.shape
    # print(height,width)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()