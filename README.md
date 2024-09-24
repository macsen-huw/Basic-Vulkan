# Basic Vulkan

## Introduction
This application uses the Vulkan graphics API to render the famous Sponza scene

## Compilation
Run the exe file.
If this doesn't work, the project can be built using Visual Studio 2022.

This application was created using C++ and Vulkan. Specifically, the Volk loader was used, which can be found [here](https://github.com/zeux/volk)

Execute the premake file  

    .\premake5.exe vs2022


## Controls
W - Move Camera Forward
S - Move Camera Backward
A - Move Camera to the Left
D - Move Camera to the Right
E - Move Camera Upwards
Q - Move Camera Downwards

Right Click - Toggle Mouse

## Interface
The interface allows you to change two settings:
- Toggle Anisotropoic Filtering
- Change Render Modes

#### Changing Render Modes
- Mipmap Levels - Visualize the texture mipmapping
- Fragment Depth - Visualize the depth value of the fragments
- Partial Fragment Depth - Visualize the derivative of the fragment depth


![Image](/assets/basic%20vulkan%203.jpg)