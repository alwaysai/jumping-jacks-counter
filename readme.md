# Jumping Jacks Counter

This app counts jumping jacks for a single person on a real-time video stream or a video file. The app defaults to the `TensorRT` engine on NVIDIA Jetson devices, and the `DNN` engine elsewhere.

## Setup
This app requires an alwaysAI account. Head to the [Sign up page](https://www.alwaysai.co/dashboard) if you don't have an account yet. Follow the instructions to install the alwaysAI tools on your development machine.

Next, create an empty project to be used with this app. When you clone this repo, you can run `aai app configure` within the repo directory and your new project will appear in the list.

## Usage
Once the alwaysAI tools are installed on your development machine (or edge device if developing directly on it) you can run the following CLI commands:

To set up the target device and install path

```
aai app configure
```

To build and deploy the docker image of the app to the target device

```
aai app install
```

To start the app with a video stream

```
aai app start
```

The app has additional command line options:

```
$ aai app start -- --help
usage: app.py [-h] [--camera CAMERA] [--video-file VIDEO_FILE] [--debug]

Jumping Jacks Counter

optional arguments:
  -h, --help            show this help message and exit
  --camera CAMERA       Set the camera index. (default: 0)
  --video-file VIDEO_FILE
                        Perform counting on a video file
  --debug               Save the data to a CSV file
```

These options can be placed in the `aai app start` command after a `--` indicating that the following flags are to be passed from the CLI down to the app. For example:

```
aai app start -- --video-file jumping-jacks.mp4
```
