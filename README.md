# opencv-stabilization-demo

A video stabilization demo using openCV and vidStab with python.
This mainly involves reducing the effect of motion due to shaking, rotation or any movement in camera.
The video source is streamed using Gstreamer and can be used with an ip source, a local stream or a videofile (all using gstreamer).
In this, a video stabilization library is used, because it is adequate for motion stabilization.

# Main properties
This demonstration is aimed mainly at webcams for desktop  or laptop, but can be easily be used with previously captured video by webcam, smartphone, camera etc.
The raw video is 
It also extrapolates the data of the trejectory for the stabilized video and the transformation it performs, to use for analisys at a later stage.
The output video is shown side by side with the raw input video, and saved in the same manner as output.avi.

# Etceteras
The simple options added to the project are a few, mainly:
/n1. The output.avi (saves the file to the project directory)
2. Graphs for the trajectory and transforms
3. The ability to stream from a few sources
4. Option to change parameters such as scaling of the video, border amount, smoothness of video
