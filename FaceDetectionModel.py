import glob
import os
import face_recognition
import cv2
import numpy as np

# IMAGES_PATH = r'C:\Users\Anaconda\OneDrive\Desktop\face_detection\Student_dataset'
IMAGES_PATH = os.path.join(os.getcwd(),"Student_dataset")

MAX_DISTANCE = 30


def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]
        
    # Run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)
    
    # Run the embedding model to get face embeddings for the supplied locations
    face_embeddings = face_recognition.face_encodings(image, face_locations)
    
    return face_locations, face_embeddings

def setup_database():
    """
    Load reference images and create a database of their face encodings
    """
    database = {}
    
    for filename in glob.glob(os.path.join(IMAGES_PATH,'*.jpeg')):
        # Load image
        image_rgb = face_recognition.load_image_file(filename)
        
        # Use the name in the filename as the identity key
        identity = os.path.splitext(os.path.basename(filename))[0]
        
        # Get the face encoding and link it to the identity
        locations, encodings = get_face_embeddings_from_image(image_rgb)
        database[identity] = encodings[0]
    
    return database

def paint_detected_face_on_image(frame, location, name=None):
    """
    Paint a rectangle around the face and write the name
    """
    # unpack the coordinates from the location tuple
    top, right, bottom, left = location
    
    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        color = (0, 128, 0)  # dark green for recognized face
        
    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom + 15), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)



def run_face_recognition(database,video):
    """
    Start the face recognition via webcam
    """
    
    
    global face_locations
    global frame
    global name
    global loc_name_dict

    # Open a connection to the camera
    video_capture = cv2.VideoCapture(video)
    matchFound = {}
    
    
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    # The face_recognition library uses keys and values of your database separately
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())
    j = 0
    
    while video_capture.isOpened():
        
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        
        if int(j/fps)!=j/fps:
            j+=1
            continue
        else:
            j += 1

            if not ret:
                break
            frame = cv2.resize(frame, None, fx=1, fy=1)


            # Run detection and embedding models
            face_locations, face_encodings = get_face_embeddings_from_image(frame, convert_to_rgb=True)

            # Build a dictionary to pair the location with the name of the detected face 
            loc_name_dict = dict()

            # Loop through each face in this frame of video and see if there's a match
            for location, face_encoding in zip(face_locations, face_encodings):

                # get the distances from this encoding to those of all reference images
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                # select the closest match (smallest distance) if it's below the threshold value
                if np.any(distances <= MAX_DISTANCE):
                    best_match_idx = np.argmin(distances)
                    name = known_face_names[best_match_idx]

                    timeStamp = j//fps
                    if name not in matchFound.keys():
                        matchFound[name] = [timeStamp]
                    else:
                        matchFound[name].append(timeStamp)
                else:
                    name = None

                # Pair the location with the name of the detected face inside a dictionary
                loc_name_dict[location] = name

    # When everything done, release the capture
    video_capture.release()
    return matchFound