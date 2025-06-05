from ultralytics import YOLO 
import cv2 
import math 
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.image import MIMEImage 
import torch 
import threading 
from twilio.rest import Client 
 
print(f'PyTorch version: {torch.__version__}') 
print('*'*10) 
print(f'_CUDA version: ') 
print('*'*10) 
print(f'CUDNN version: {torch.backends.cudnn.version()}') 
print(f'Available GPU devices: {torch.cuda.device_count()}') 
print(f'Device Name: {torch.cuda.get_device_name()}') 
 
model = YOLO("best.pt") 
model.to('cuda') 
classNames = {0:'Lion',1:'Tiger',2:'Boar',3:'Elephant',4:'Human',5:'Dog',6:'Wolf',7:'Bear',8:'Leopard'} 
 
cap = cv2.VideoCapture(0) 
cap.set(3, 640) 
cap.set(4, 480) 
 
counter = 0  # Counter for naming the saved images 
 
def send_emails(email_list, image_filenames, cls): 
    smtp_port = 587                 # Standard secure SMTP port 
    smtp_server = "smtp.gmail.com"  # Google SMTP Server 
    email_from = "animaldet35@gmail.com"       
    pswd = "bwnx wkjw cpyr rkdd"           
    subject = "ALERT!!!"   
 
    for person in email_list: 
        try: 
            #Body of the email 
            body = f"DETECTED {classNames[cls]}" 
 
            # make a MIME object to define parts of the email 
            msg = MIMEMultipart() 
            msg['From'] = email_from 
            msg['To'] = person 
            msg['Subject'] = subject 
 
            # Attach the body of the message 
            msg.attach(MIMEText(body, 'plain')) 
                                                                                                                              
            # Attach the detected images 
            for filename in image_filenames: 
                with open(filename, 'rb') as f: 
                    img_data = f.read() 
                    image = MIMEImage(img_data, name=filename) 
                    msg.attach(image) 
 
            # Cast as string 
            text = msg.as_string() 
 
            # Connect with the server and send email 
            with smtplib.SMTP(smtp_server, smtp_port) as TIE_server: 
                TIE_server.starttls() 
                TIE_server.login(email_from, pswd) 
                TIE_server.sendmail(email_from, person, text) 
 
            print(f"Email sent to: {person}") 
        except Exception as e: 
            print(f"Error occurred while sending email: {e}") 
 
def send_sms(message_body, twilio_phone_number, recipient_phone_number, account_sid, 
auth_token): 
    try: 
        client = Client(account_sid, auth_token) 
        message = client.messages.create( 
            body=message_body, 
            from_=twilio_phone_number, 
            to=recipient_phone_number 
        ) 
        print("SMS sent successfully. SID:", message.sid) 
    except Exception as e: 
        print(f"Error occurred while sending SMS: {e}") 
 
def detect_objects_and_notify(): 
    global counter 
    while True: 
        success, img = cap.read() 
    
        results = model(img, stream=True, verbose=False) 
        detected_image_filenames = [] 
 
        for r in results: 
            boxes = r.boxes 
            for box in boxes: 
                confidence = math.ceil((box.conf[0] * 100)) / 100 
                if confidence > 0.8: 
                    cls = int(box.cls[0]) 
                    if cls in classNames: 
                        x1, y1, x2, y2 = box.xyxy[0] 
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values 
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3) 
                        org = (x1, y1 - 10)                                                                                                                           
                        font = cv2.FONT_HERSHEY_SIMPLEX 
                        fontScale = 1 
                        color = (0, 255, 0) 
                        thickness = 2 
                        text = f"{classNames[cls]}:{confidence}" 
                        cv2.putText(img, text, org, font, fontScale, color, thickness) 
                        
                        # Extract and save the detected object 
                        cropped_img = img#[y1:y2, x1:x2] 
                        filename = f'detected_{counter}.jpg' 
                        cv2.imwrite(filename, cropped_img) 
                        detected_image_filenames.append(filename) 
 
        # Email 
        if counter % 10 == 0 and detected_image_filenames: 
            threading.Thread(target=send_emails, args=(["adarshv8045@gmail.com"], 
detected_image_filenames, cls)).start() 
 
        # SMS 
        #if counter % 20 == 0 and detected_image_filenames: 
            #threading.Thread(target=send_sms, args=(f"Detected {classNames[cls]}", "+14086693482", "+917306058764", "ACf3a9b3b8735c034418ae35f5e4ee7683", 
                                                     #"3f96072d1db80fcfb9c34164767533d5")).start() 
 
        cv2.imshow('Result', img) 
        if cv2.waitKey(1) == ord('q'): 
            break 
 
        counter += 1 
 
    cap.release() 
    cv2.destroyAllWindows() 
 
# Call the function to start object detection and notification 
detect_objects_and_notify() 