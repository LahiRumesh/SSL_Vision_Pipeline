import os
import cv2
import time
from datetime import datetime

class Image_Capture():

    def __init__(self,cameraID):

        self.cameraID = cameraID

    def crop_to_box(self,img, size):
        shape = img.shape
        if shape[0] > shape[1]:
            start = int((shape[0]-shape[1])/2)
            img = img[start:start+shape[1], :]
        else:
            start = int((shape[1]-shape[0])/2)
            img = img[:, start:start+shape[0]]
        img = cv2.resize(img, (size, size))
        return img

    def capture_image(self,
                        img_size,
                        out_dir,
                        rotate90=True,rotate180=True,rotate270=True,scale=True,scale_val=0.2):
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        cam_feed = cv2.VideoCapture(self.cameraID)

        while True:
            timestr = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            ret, frame = cam_feed.read()
            
            frame = self.crop_to_box(frame,img_size)
            if not ret:
                print("Stream is not working")
                break
            cv2.imshow("Image_Capture", frame)

            k = cv2.waitKey(1)
            if k%256 == 27:
                # press ESC to Close Window
                print("Terminate process...")
                break
            elif k%256 == 32:
                # press SPACE to capture image
                img_name = "{}.jpg".format(timestr)
                cv2.imwrite(os.path.join(out_dir,img_name), frame)
                print("{} image saved!".format(img_name))
                if rotate90:
                    img_name90 = "{}90.jpg".format(timestr)
                    frame90 = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(os.path.join(out_dir,img_name90), frame90)
                    print("{} image saved!".format(img_name90))
                if rotate180:
                    img_name180 = "{}180.jpg".format(timestr)
                    frame180 = cv2.rotate(frame, cv2.ROTATE_180)
                    cv2.imwrite(os.path.join(out_dir,img_name180), frame180)
                    print("{} image saved!".format(img_name180))
                if rotate270:
                    img_name270 = "{}270.jpg".format(timestr)
                    frame270 = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.imwrite(os.path.join(out_dir,img_name270), frame270)
                    print("{} image saved!".format(img_name270))
                if scale:
                    img_namescle = "{}scle.jpg".format(timestr)
                    sclae_img = cv2.resize(frame, (int(frame.shape[1] * scale_val),int(frame.shape[0] * scale_val)))
                    cv2.imwrite(os.path.join(out_dir,img_namescle), sclae_img)
                    print("{} image saved!".format(img_namescle))


        cam_feed.release()

        cv2.destroyAllWindows()


if __name__=='__main__':

    cap = Image_Capture(0)
    '''
    img_dir : Image folder 
    img_size : Image size
    '''
    img_dir = 'train_images'
    img_size = 512

    cap.capture_image(img_size,img_dir,rotate90=True,rotate180=True,rotate270=True,scale=True,scale_val=0.2)