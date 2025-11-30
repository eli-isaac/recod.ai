1. [optional] use a binary classifier to decide if image has forgery in it or not
2. use sam3 to segment the image into objects
3. pass croped objects indidvually to a binary classifier to decide if it is a forgery or not:
    - this can be an ensemble of classifiers
    - we can train our own neural network for this
    - we can look for things like if the object overlaps another one
4. [optional] we filter out some objects that are either too small or too large or using NMS
5. for each of the forgery images we look for similar objects in the image and group them together:
    - this can be done using an ensemble approach
    - we can include embedding from various models like CLIP, DYNOv2, etc.
    - we can include some manually computed statitics like hu moments


next up:
convert current code to lable objects with numbers in visualization, then test all current methods but have them receive an object number as input and their goal is to find the closes object, then we move to a thrshold and they find all objects within it, but maybe we make it an adaptive threshold.