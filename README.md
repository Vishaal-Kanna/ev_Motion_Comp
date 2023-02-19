# Event Motion Compensation
This repository contains the code for Event camera motion compensation. We predict the motion of the camera by maximizing the image contrast, thereby removing motion blur for better detections and also the change in pose of the camera for visual odometry. Unlike the current methods that use a stream of x, y, t data for motion compensation, this algorithm uses the input representation of the events in the form of a discretized volume that maintains the temporal distribution of the events, proposed by Kostas Daniilidis et al.

## Results
The below image shows data captured by an event camera before and after motion compensation.

<p align="center">
  <img width="240" height="180" src="[http://www.fillmurray.com/460/300](https://github.com/Vishaal-Kanna/ev_Motion_Comp/blob/main/Sample_data/sample_op_before_comp.png)">
  <img width="240" height="180" src="[http://www.fillmurray.com/460/300](https://github.com/Vishaal-Kanna/ev_Motion_Comp/blob/main/Sample_data/sample_op_after_comp.png)">
</p>



