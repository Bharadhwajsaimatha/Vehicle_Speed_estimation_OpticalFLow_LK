# Vehicle Speed Estimation using Optical Flow

## Project Overview

This project focuses on estimating the speed of vehicles using optical flow techniques. Optical flow refers to the pattern of apparent motion of objects, surfaces, and edges in a visual scene, caused by the relative motion between an observer and the scene. By analyzing these motion patterns, we aim to accurately estimate the speed of moving vehicles.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
In this project, Lucas-Kanade optical flow method [link](https://www.cse.unr.edu/~bebis/CS474/Handouts/Lucas_Kanade.pdf) is employed to capture and analyze the movement of vehicles in a video feed. The main steps involved in the process are as follows:

1. **Capturing Video Frames**: The first step involves capturing video frames from a pre-recorded video file or a real-time video feed. This provides the raw data needed for further processing.

2. **Perspective Transformation**: To obtain a bird's eye view of the scene, a perspective transformation is applied to the captured video frames. This transformation helps in reducing the distortion caused by the camera angle and provides a top-down view of the vehicles, making it easier to analyze their motion.

3. **Optical Flow Calculation**: Optical flow algorithms, such as the Farneback or Lucas-Kanade method, are then applied to the transformed video frames. These algorithms compute the motion vectors of the pixels in the frames, which represent the apparent motion of objects in the scene.

4. **Speed Estimation**: Using the motion vectors obtained from the optical flow calculation, the speed of the vehicles is estimated. This involves analyzing the magnitude and direction of the motion vectors to determine the velocity of each vehicle in the scene.

By following these steps, we can accurately estimate the speed of moving vehicles in a video feed using optical flow techniques.

![Flowchart](asets/project_flowchart.png)

## Features

- **Real-time Speed Estimation**: Process video feeds in real-time to estimate vehicle speed.
- **Optical Flow Algorithms**: Utilize advanced optical flow techniques like Farneback, Lucas-Kanade, etc.
- **Visualization**: Visual representation of vehicle speeds and motion vectors.

## Installation

To get started with this project, follow the steps below:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/vehicle-speed-estimation.git
    cd vehicle-speed-estimation
    ```

2. **Install dependencies:**
    Ensure you have Python installed, then run:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Here's a basic example of how to use this project:

1. **Prepare your video feed**: Ensure you have a video file or a real-time video feed ready.
2. **Run the script**: Execute the main script to start processing the video feed.
    ```sh
    python vel_LK_OF_V1_1.py --_input_file path/to/your/video.mp4
    ```

## Examples

Below are some example use cases and results:

- **Example 1**: 
- **Example 2**:

## Contributing

Contributions are welcome! If you have any improvements or suggestions, please feel free to submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

No license. Open sourced.

## Contact

For any inquiries or feedback, please reach out to:

- [Email](mailto:bharadhwaj2299@gmail.com)

---

*This README is a work in progress and will be updated with more details soon.*
