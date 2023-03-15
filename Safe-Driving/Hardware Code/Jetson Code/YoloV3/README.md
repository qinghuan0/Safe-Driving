# Laptop Test:

To test the model you can enter our Jupyter Notebook in Google Colab and everything is ready to run the notebook.

Link: 

## Model:

In this case we have used an existing model called YoloV3 used for object detection in addition to its own algorithm to detect the distance of objects with a simple camera.

The model creator page: https://pjreddie.com/darknet/yolo/

# How does it work:

In this algorithm we use the detection of objects using PyTorch, YoloV3 and OpenCV which allows the use of CDNN.

<img src="https://i.ibb.co/S3Wz9Sc/process2.jpg" width="600">

In this case, I created an algorithm to calculate the approximate distance at which an object is located using YOLOV3, the algorithm is as follows.

We calculate the aperture and distance of our camera, in my case it is 19 centimeters in aperture at a distance of 16 cm.

It means that an object 19 cm wide can be at least 16 cm away from the camera so that it can fully capture it, in this case the calculation of the distance of any object becomes a problem of similar triangles.

<img src="https://i.ibb.co/cC8wbb6/esquemadist.png" width="1000">

In this case, since our model creates boxes where it encloses our detected objects, we will use that data to obtain the apparent width of the object, in this case the approximate width will be stored in a variable during the calculation.

<img src="https://i.ibb.co/jRf5xV3/exmaple.png" width="1000">

If we combine both terms we get the following equation.

<img src="https://i.ibb.co/5R4dKvG/Screenshot-from-2021-11-01-19-03-59.png" width="600">

In this case for the objects that we are going to detect, we approximate the following widths in centimeters.

    {
        person:70,
        car:220,
        motorcycle:120,
        dogs:50 
    }

Here is an example of its operation with the same image.

<img src="https://i.ibb.co/wdf45wq/prockess2.jpg" width="600">

Testing the algorithm with the camera.

<img src="https://i.ibb.co/d2dzt60/process.jpg" width="600">
<img src="https://i.ibb.co/qjTc7rX/process1.jpg" width="600">

Once having this distance, we will filter all distances that are greater than 2 meters, this being a safe distance to the car at the blind spot.

<img src="https://i.ibb.co/mzNdqVp/Whats-App-Image-2020-03-16-at-13-57-36.jpg" width="600">