# API to detect blood on face

#### Use case of the API
1. We can use this API to detect the severity of accident by checking if the driver is bleeding
2. We can use the API in CCTV cameras to identify if someone is bleeding
3. Can be used for other face detection system

### Tech stack used
1. Python Machine learning model with Tensorflow keras framework
2. Python Flask frame work for deployment
3. Heroku for hosting the API
4. Postman to test the POST request for the API

### How does the API work

The API accepts only POST request as a JSON format:

{

"url" : "Input the image url here"

}

The response given by the API is alson in Json format with a boolean value for the key "Blood detected":

{

    "Blood detected": 0 if No blood is detect, 1 if Blood is detected 

}

#### Check out the machine learning model used,[FaceBloodIdentifier](https://github.com/Shakthi-Dhar/FaceBloodIdentifier).

#### Test the API on postman with POST request, API url: [https://faceblood-detection-api.herokuapp.com/](https://faceblood-detection-api.herokuapp.com/).
