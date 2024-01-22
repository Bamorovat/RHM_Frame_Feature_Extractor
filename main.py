"""
Author: Mohammad Hossein Bamorovat Abadi
Project: RHM Dataset Frame Extraction for Human Activity Recognition
Description: This script processes videos from the RHM dataset for human activity recognition.
             It extracts specific frames and applies manual feature extraction techniques.
Date: 2024/01/05
Contact: m.bamorovvat@gmail.com
Version: 1.0.0

License:  GNU General Public License (GPL)

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

If you have questions about the GPL or need further clarification,
you can refer to the full text of the license at:
http://www.gnu.org/licenses/gpl-3.0.html
"""

import os
import cv2
import pandas as pd
import numpy as np

# Path to the RHM dataset
RHMPath = "/media/abbas/Elements/Dataset/RHM_Full"

# Number of frames to extract from each video.
ExtractFrameNumber = 17

# Debug mode: if True, frames will be displayed during processing
Debug = False
Show_Debug = False

# Feature extraction flags. Set to True to extract the feature and False to skip it.
MotionAggregation = True
FrameVariationMapper = True
DifferentialMotionTrajectory = True
Normal = True
Subtract = True
OpticalFlow = True
MotionHistoryImages = True

# Initialize list to hold names of features to be extracted
FeatureFrameExtractionList = []

# Populate the feature list based on the set flags
if MotionAggregation:
    FeatureFrameExtractionList.append("MotionAggregation")
if FrameVariationMapper:
    FeatureFrameExtractionList.append("FrameVariationMapper")
if DifferentialMotionTrajectory:
    FeatureFrameExtractionList.append("DifferentialMotionTrajectory")
if Normal:
    FeatureFrameExtractionList.append("Normal")
if Subtract:
    FeatureFrameExtractionList.append("Subtract")
if OpticalFlow:
    FeatureFrameExtractionList.append("OpticalFlow")
if MotionHistoryImages:
    FeatureFrameExtractionList.append("MotionHistoryImages")
# Print the list of features to be extracted
if Debug:
    print("FeatureFrameExtractionList: ", FeatureFrameExtractionList)

if Debug:
    # Check the openCV version
    print("openCV version: ", cv2.__version__)


# Function to create the folder structure for the extracted frames
def folder_check(split, view, action_class, video_name):
    """
    Check and create necessary folders for storing extracted frames.

    Args:
    split (str): Split category of the dataset.
    view (str): Viewpoint category.
    action_class (str): Action class category.
    video_name (str): Name of the video.

    Returns:
    pd.DataFrame: DataFrame containing paths for each feature extraction type.
    """

    # save addresses
    RHMViewPath = os.path.join(RHMPath, "RHHAR_" + view)

    Frame_path = os.path.join(RHMViewPath, "Frames_Features_Extraction")
    if not os.path.exists(Frame_path):
        os.makedirs(Frame_path)
        print(f"Folder '{Frame_path}' created successfully.")
    else:
        print(f"Folder '{Frame_path}' already exists.")

    # Create a list to hold the paths of the features to be extracted
    Paths_list = []

    # Create the folder structure for the extracted frames
    for feature in FeatureFrameExtractionList:

        FeaturePath = os.path.join(Frame_path, "rhhar_" + view + "_" + feature)
        SplitPath = os.path.join(FeaturePath, split)
        ActionPath = os.path.join(SplitPath, action_class)
        VideoPath = os.path.join(ActionPath, action_class + "_" + view + "_" + video_name)

        # Check if the folder already exists
        if not os.path.exists(FeaturePath):
            # Create the folder
            os.makedirs(FeaturePath)
            print(f"Folder '{FeaturePath}' created successfully.")
        else:
            print(f"Folder '{FeaturePath}' already exists.")

        # Check if the folder already exists
        if not os.path.exists(SplitPath):
            # Create the folder
            os.makedirs(SplitPath)
            print(f"Folder '{SplitPath}' created successfully.")
        else:
            print(f"Folder '{SplitPath}' already exists.")

        # Check if the folder already exists
        if not os.path.exists(ActionPath):
            # Create the folder
            os.makedirs(ActionPath)
            print(f"Folder '{ActionPath}' created successfully.")
        else:
            print(f"Folder '{ActionPath}' already exists.")

        # Check if the folder already exists
        if not os.path.exists(VideoPath):
            # Create the folder
            os.makedirs(VideoPath)
            print(f"Folder '{VideoPath}' created successfully.")
        else:
            print(f"Folder '{VideoPath}' already exists.")

        # Append the path to the list
        if os.path.exists(VideoPath):
            Paths_list.append([feature, VideoPath])

    # Return the list of paths in a dataframe
    return pd.DataFrame(Paths_list, columns=['Feature', 'Path'])


# Function to extract the frames from the videos
def get_frame(video_path, split, view, action_class, video_name):

    """
    Extract frames from the videos and save them in the appropriate folders.
    args:
    video_path: path to the video
    split:  e.g. train, test, validation
    view: e.g. FrontView, BackView, OmniView, RobotView
    action_class:  Bending, Cleaning, Drinking, OpeningCan, Reaching, StairsClimbingDown, StandingUp
    CarryingObject, ClosingCan, LiftingObject, PuttingDownObjects, SittingDown, StairsClimbingUp, Walking
    video_name: e.g. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

    return:
    """

    # Initialize the paths to the folders
    MotionAggregation_path = None
    FrameVariationMapper_path = None
    DifferentialMotionTrajectory_path = None
    Normal_path = None
    Subtract_path = None
    OpticalFlow_path = None
    MotionHistoryImages_path = None
    # MotionEnergyImages_path = None

    # Get the paths to the folders
    FramesPath_df = folder_check(split, view, action_class, video_name)

    if Debug:
        print("FramesPath_df: ", FramesPath_df)

    # Check if the feature is to be extracted
    if FramesPath_df[FramesPath_df['Feature'] == 'MotionAggregation'].shape[0] > 0:
        MotionAggregation = True
        # Get the path to the folder
        MotionAggregation_path = FramesPath_df[FramesPath_df['Feature'] == 'MotionAggregation']['Path'].values[0]
        if Debug:
            print("MotionAggregation_path: ", MotionAggregation_path)
    else:
        MotionAggregation = False

    # Check if the feature is to be extracted
    if FramesPath_df[FramesPath_df['Feature'] == 'FrameVariationMapper'].shape[0] > 0:
        FrameVariationMapper = True
        # Get the path to the folder
        FrameVariationMapper_path = FramesPath_df[FramesPath_df['Feature'] == 'FrameVariationMapper']['Path'].values[0]
        if Debug:
            print("FrameVariationMapper_path: ", FrameVariationMapper_path)
    else:
        FrameVariationMapper = False

    # Check if the feature is to be extracted
    if FramesPath_df[FramesPath_df['Feature'] == 'DifferentialMotionTrajectory'].shape[0] > 0:
        DifferentialMotionTrajectory = True
        # Get the path to the folder
        DifferentialMotionTrajectory_path = FramesPath_df[FramesPath_df['Feature'] == 'DifferentialMotionTrajectory']['Path'].values[0]
        if Debug:
            print("DifferentialMotionTrajectory_path: ", DifferentialMotionTrajectory_path)
    else:
        DifferentialMotionTrajectory = False

    # Check if the feature is to be extracted
    if FramesPath_df[FramesPath_df['Feature'] == 'Normal'].shape[0] > 0:
        NormalFrame = True
        # Get the path to the folder
        Normal_path = FramesPath_df[FramesPath_df['Feature'] == 'Normal']['Path'].values[0]
        if Debug:
            print("Normal_path: ", Normal_path)
    else:
        NormalFrame = False

    # Check if the feature is to be extracted
    if FramesPath_df[FramesPath_df['Feature'] == 'Subtract'].shape[0] > 0:
        Subtract = True
        # Get the path to the folder
        Subtract_path = FramesPath_df[FramesPath_df['Feature'] == 'Subtract']['Path'].values[0]
        if Debug:
            print("Subtract_path: ", Subtract_path)
    else:
        Subtract = False

    # Check if the feature is to be extracted
    if FramesPath_df[FramesPath_df['Feature'] == 'OpticalFlow'].shape[0] > 0:
        OpticalFlow = True
        # Get the path to the folder
        OpticalFlow_path = FramesPath_df[FramesPath_df['Feature'] == 'OpticalFlow']['Path'].values[0]
        if Debug:
            print("OpticalFlow_path: ", OpticalFlow_path)
    else:
        OpticalFlow = False

    # Check if the feature is to be extracted
    if FramesPath_df[FramesPath_df['Feature'] == 'MotionHistoryImages'].shape[0] > 0:
        MotionHistoryImages = True
        # Get the path to the folder
        MotionHistoryImages_path = FramesPath_df[FramesPath_df['Feature'] == 'MotionHistoryImages']['Path'].values[0]
        if Debug:
            print("MotionHistoryImages_path: ", MotionHistoryImages_path)
    else:
        MotionHistoryImages = False

    # Prepare the parameters for the motion history images
    if MotionHistoryImages:
        MHI_DURATION = 50
        DEFAULT_THRESHOLD = 32

    # Open the video capture
    video_capture = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = ExtractFrameNumber
    # Calculate the number of steps to take
    steps = int(total_frames / frame_number)
    if 24 < total_frames < 34:
        steps = 2
    elif total_frames <= 24:
        steps = 1

    if Debug:
        print("total_frames: ", total_frames)
        print("steps: ", steps)

    # Initialize the variables
    frame_count = 0
    FirstFrame = None
    OldFrame = None
    DMTSubOld = None
    prvs = None
    prev_frame = None
    OldFrameSubtract = None
    FrameVariationMapperCalculation = None
    MotionAggregationCalculation = None
    NormalCalculation = None
    DMTCalculation = None
    SubtractCalculation = None
    OpticalFlowCalculation = None
    MHICalculator = None

    # Loop through the video
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Check if the video capture was successful
        if not ret:
            break

        # Get the current frame number
        frame_number = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))

        # Check if the frame number is 1 for the first frame
        if frame_number == 1:
            FirstFrame = frame
            OldFrame = frame
            OldFrameSubtract = frame

            # Prepare the parameters for the Optical Flow
            if OpticalFlow:
                # Convert the frame to grayscale
                prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = np.zeros_like(frame)
                hsv[..., 1] = 255

            # Prepare the parameters for the motion history images
            if MotionHistoryImages:
                h, w = frame.shape[:2]
                prev_frame = frame.copy()
                motion_history = np.zeros((h, w), np.float32)
                timestamp = 0

        FirstGroupImage = None

        # Loop through the steps
        for i in range(steps):

            ret, frame = video_capture.read()

            # Check if the video capture was successful
            if not ret:
                break

            if frame_number == 1:
                if i == 0:
                    if DifferentialMotionTrajectory:
                        DMTSubOld = cv2.absdiff(OldFrame, frame)
                        OldFrame = frame

                if i == 1:
                    if DifferentialMotionTrajectory:
                        DMTSub = cv2.absdiff(OldFrame, frame)
                        DMTCalculation = cv2.addWeighted(DMTSubOld, 0.85, DMTSub, 0.45, 0.0)
                        OldFrame = frame

                if i >= 2:
                    if DifferentialMotionTrajectory:
                        DMTSub = cv2.absdiff(OldFrame, frame)
                        DMTCalculation = cv2.addWeighted(DMTCalculation, 0.85, DMTSub, 0.45, 0.0)
                        OldFrame = frame

                if i > 0:
                    if OpticalFlow:
                        # Convert the frame to grayscale
                        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Calculate the dense optical flow
                        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        # Obtain the flow magnitude and direction angle
                        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        # Update the color image
                        hsv[..., 0] = ang * 180 / np.pi / 2
                        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                        # Convert the HSV image to RGB
                        OpticalFlowCalculation = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        # Update the previous frame
                        prvs = next

                    if MotionHistoryImages:
                        frame_diff = cv2.absdiff(frame, prev_frame)
                        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
                        ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
                        timestamp += 1
                        # update motion history
                        cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)
                        # normalize motion history
                        MHICalculator = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)

            if frame_number != 1:
                if DifferentialMotionTrajectory:
                    DMTSub = cv2.absdiff(OldFrame, frame)
                    DMTCalculation = cv2.addWeighted(DMTCalculation, 0.85, DMTSub, 0.45, 0.0)
                    OldFrame = frame

                if OpticalFlow:
                    # Convert the frame to grayscale
                    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Calculate the dense optical flow
                    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    # Obtain the flow magnitude and direction angle
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    # Update the color image
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    # Convert the HSV image to RGB
                    OpticalFlowCalculation = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    # Update the previous frame
                    prvs = next

                if MotionHistoryImages:
                    frame_diff = cv2.absdiff(frame, prev_frame)
                    gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
                    ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
                    timestamp += 1
                    # update motion history
                    cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)
                    # normalize motion history
                    MHICalculator = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)

            if i == 0:
                if MotionAggregation:
                    FirstGroupImage = frame
            if i == 1:
                if MotionAggregation:
                    MotionAggregationCalculation = cv2.addWeighted(FirstGroupImage, 0.5, frame, 0.5, 0.0)
            if i > 1:
                if MotionAggregation:
                    MotionAggregationCalculation = cv2.addWeighted(MotionAggregationCalculation, 0.5, frame, 0.5, 0.0)

            if i == steps-1:
                if FrameVariationMapper:
                    FrameVariationMapperCalculation = cv2.absdiff(FirstFrame, frame)
                if NormalFrame:
                    NormalCalculation = frame

            if Subtract:
                SubtractCalculation = cv2.absdiff(OldFrameSubtract, frame)
                OldFrameSubtract = frame

        # name the frame for saving
        frame_filename = f'{frame_count}.jpg'

        if Show_Debug:
            # Display the resulting frame
            if FrameVariationMapper:
                cv2.imshow('FrameVariationMapper', FrameVariationMapperCalculation)
            if MotionAggregation:
                cv2.imshow('MotionAggregation', MotionAggregationCalculation)
            if NormalFrame:
                cv2.imshow('Normal', NormalCalculation)
            if DifferentialMotionTrajectory:
                cv2.imshow('DifferentialMotionTrajectory', DMTCalculation)
            if Subtract:
                cv2.imshow('Subtract', SubtractCalculation)
            if OpticalFlow:
                cv2.imshow('OpticalFlow', OpticalFlowCalculation)
            if MotionHistoryImages:
                cv2.imshow('MotionHistoryImages', MHICalculator)

            # Press Q on keyboard to  exit
            cv2.waitKey(0)
            # Break the loop
            cv2.destroyAllWindows()

        # Save the frames to the appropriate folders
        if MotionAggregation:
            MA_frame_path = os.path.join(MotionAggregation_path, frame_filename)
            cv2.imwrite(MA_frame_path, MotionAggregationCalculation)

        # Save the frames to the appropriate folders
        if FrameVariationMapper:
            FVM_frame_path = os.path.join(FrameVariationMapper_path, frame_filename)
            cv2.imwrite(FVM_frame_path, FrameVariationMapperCalculation)
        # Save the frames to the appropriate folders
        if NormalFrame:
            Normal_frame_path = os.path.join(Normal_path, frame_filename)
            cv2.imwrite(Normal_frame_path, NormalCalculation)

        # Save the frames to the appropriate folders
        if DifferentialMotionTrajectory:
            DMT_frame_path = os.path.join(DifferentialMotionTrajectory_path, frame_filename)
            cv2.imwrite(DMT_frame_path, DMTCalculation)

        # Save the frames to the appropriate folders
        if Subtract:
            Sub_frame_path = os.path.join(Subtract_path, frame_filename)
            cv2.imwrite(Sub_frame_path, SubtractCalculation)

        # Save the frames to the appropriate folders
        if OpticalFlow:
            OF_frame_path = os.path.join(OpticalFlow_path, frame_filename)
            cv2.imwrite(OF_frame_path, OpticalFlowCalculation)

        # Save the frames to the appropriate folders
        if MotionHistoryImages:
            MHI_frame_path = os.path.join(MotionHistoryImages_path, frame_filename)
            cv2.imwrite(MHI_frame_path, MHICalculator)

        # Increment the frame count
        frame_count += 1

    # When everything done, release the capture
    video_capture.release()
    # Close all the frames
    cv2.destroyAllWindows()


# Main function
def main():
    # ViewPoints = FrontView, BackView, OmniView, RobotView
    ViewPoints = os.listdir(RHMPath)
    ViewList = [view for view in ViewPoints if view != "RHHAR"]
    # RHM_view_list = ["RHHAR_FrontView", "RHHAR_BackView", "RHHAR_OmniView", "RHHAR_RobotView"]

    if Debug:
        print("ViewList: ", ViewList)
        print("%" * 90)

    for folder in ViewList:
        # Get the path to the folder
        ViewFolderPath = os.path.join(RHMPath, folder)
        # Split the name by underscore
        parts = folder.split('_')
        # Get the second part
        view = parts[1]
        if Debug:
            print("view Name: ", view)

        # Get the path of splits
        SplitPath = os.path.join(ViewFolderPath, "Rhhar_" + view)
        # Change the current working directory
        os.chdir(SplitPath)
        # Verify the current working directory
        if Debug:
            print("current Path is:", os.getcwd())

        # Get the list of splits
        SplitList = os.listdir(SplitPath)
        if Debug:
            print("SplitList: ", SplitList)
            print("$" * 70)

        # Loop through the splits
        for split in SplitList:
            if Debug:
                print("-" * 50)
                print("List: ", split)
            ActionPath = os.path.join(SplitPath, split)
            ActionList = os.listdir(ActionPath)
            if Debug:
                print("ActionList: ", ActionList)

            for action in ActionList:
                if Debug:
                    print("Action: ", action)
                VideoPath = os.path.join(ActionPath, action)
                VideoList = os.listdir(VideoPath)
                if Debug:
                    print("VideoList: ", VideoList)
                for video in VideoList:
                    if Debug:
                        print("video: ", video)
                    if not video:
                        continue
                    VideoPath = os.path.join(VideoPath, video)

                    if Debug:
                        print("VideoPath: ", VideoPath)
                        print("View: ", view)
                        print("Split:", split)
                        print("Action: ", action)
                        print("Video: ", video)
                    # Split the name by underscore
                    video_name = video.split('_')
                    video_name = video_name[2]
                    video_name = video_name.split('.')
                    video_name = video_name[0]
                    if Debug:
                        print("Video Name: ", video_name)
                    get_frame(VideoPath, split, view, action, video_name)


# Call the main function
if __name__ == '__main__':
    main()
