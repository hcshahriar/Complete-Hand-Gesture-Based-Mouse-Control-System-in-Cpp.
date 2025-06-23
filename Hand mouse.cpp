#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <windows.h>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Configuration parameters
const int SMOOTHING_FACTOR = 5;       // Higher = smoother but slower cursor
const int CLICK_THRESHOLD = 30;       // Distance threshold for click detection
const double MIN_CONTOUR_AREA = 5000; // Minimum hand area to consider
const double MAX_CONTOUR_AREA = 30000; // Maximum hand area to consider

// Global variables for smoothing
vector<Point> cursorHistory(SMOOTHING_FACTOR);
int historyIndex = 0;

// Function to simulate mouse movement with smoothing
void moveMouseSmooth(int x, int y) {
    // Store current position in history
    cursorHistory[historyIndex] = Point(x, y);
    historyIndex = (historyIndex + 1) % SMOOTHING_FACTOR;
    
    // Calculate average position
    int avgX = 0, avgY = 0;
    for (const auto& p : cursorHistory) {
        avgX += p.x;
        avgY += p.y;
    }
    avgX /= SMOOTHING_FACTOR;
    avgY /= SMOOTHING_FACTOR;
    
    SetCursorPos(avgX, avgY);
}

// Function to simulate left mouse click
void leftClick() {
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &input, sizeof(INPUT));
    
    ZeroMemory(&input, sizeof(INPUT));
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &input, sizeof(INPUT));
}

// Function to simulate right mouse click
void rightClick() {
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
    SendInput(1, &input, sizeof(INPUT));
    
    ZeroMemory(&input, sizeof(INPUT));
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_RIGHTUP;
    SendInput(1, &input, sizeof(INPUT));
}

// Function to detect fingers based on convexity defects
int detectFingers(const vector<Point>& contour, vector<Point>& hull, vector<int>& hullIndices) {
    if (hullIndices.size() < 3) return 0;
    
    vector<Vec4i> defects;
    convexityDefects(contour, hullIndices, defects);
    
    int fingerCount = 0;
    for (const auto& defect : defects) {
        int startIdx = defect[0];
        int endIdx = defect[1];
        int farIdx = defect[2];
        float depth = defect[3] / 256.0f;
        
        // Filter defects based on depth and angle
        Point startPt = contour[startIdx];
        Point endPt = contour[endIdx];
        Point farPt = contour[farIdx];
        
        double a = norm(endPt - farPt);
        double b = norm(startPt - farPt);
        double c = norm(startPt - endPt);
        
        double angle = acos((a*a + b*b - c*c) / (2 * a * b)) * 180 / CV_PI;
        
        if (depth > 20 && angle < 90) {
            fingerCount++;
        }
    }
    
    return fingerCount;
}

// Main gesture detection function
void detectGestures(Mat &frame, bool &clickDetected) {
    static Point prevTip(-1, -1);
    static int noHandFrames = 0;
    clickDetected = false;
    
    // Convert to HSV color space for better skin detection
    Mat hsv;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    
    // Skin color threshold (adjust these values for different skin tones)
    Mat mask;
    inRange(hsv, Scalar(0, 30, 60), Scalar(20, 150, 255), mask);
    
    // Morphological operations to remove noise
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    
    // Find contours
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (!contours.empty()) {
        // Find largest contour (assuming it's the hand)
        auto largestContour = max_element(contours.begin(), contours.end(),
            [](const vector<Point>& a, const vector<Point>& b) {
                return contourArea(a) < contourArea(b);
            });
        
        double area = contourArea(*largestContour);
        
        if (area > MIN_CONTOUR_AREA && area < MAX_CONTOUR_AREA) {
            noHandFrames = 0;
            
            // Get convex hull
            vector<Point> hull;
            vector<int> hullIndices;
            convexHull(*largestContour, hull, false);
            convexHull(*largestContour, hullIndices, false);
            
            // Detect fingers
            int fingerCount = detectFingers(*largestContour, hull, hullIndices);
            
            // Find the highest point (finger tip)
            Point tip = *min_element(hull.begin(), hull.end(),
                [](const Point& a, const Point& b) {
                    return a.y < b.y;
                });
            
            // Map hand position to screen coordinates
            int screenX = tip.x * GetSystemMetrics(SM_CXSCREEN) / frame.cols;
            int screenY = tip.y * GetSystemMetrics(SM_CYSCREEN) / frame.rows;
            
            // Move mouse
            moveMouseSmooth(screenX, screenY);
            
            // Detect click (finger down then up)
            if (prevTip != Point(-1, -1)) {
                double dist = norm(tip - prevTip);
                if (fingerCount <= 2 && dist < CLICK_THRESHOLD) {
                    leftClick();
                    clickDetected = true;
                }
            }
            prevTip = tip;
            
            // Visual feedback
            drawContours(frame, vector<vector<Point>>{hull}, -1, Scalar(0, 255, 0), 2);
            circle(frame, tip, 10, Scalar(0, 0, 255), -1);
            putText(frame, format("Fingers: %d", fingerCount), Point(10, 30), 
                   FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        } else {
            noHandFrames++;
        }
    } else {
        noHandFrames++;
    }
    
    // Reset previous tip if hand is lost for several frames
    if (noHandFrames > 10) {
        prevTip = Point(-1, -1);
    }
}

int main() {
    // Open webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open camera." << endl;
        return -1;
    }
    
    // Set camera resolution
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    
    Mat frame;
    namedWindow("Gesture Control", WINDOW_AUTOSIZE);
    
    cout << "Gesture Control System Started" << endl;
    cout << "Move your hand in front of the camera to control the mouse" << endl;
    cout << "Close your hand to click" << endl;
    cout << "Press any key to exit" << endl;
    
    // Initialize cursor history
    POINT initialPos;
    GetCursorPos(&initialPos);
    fill(cursorHistory.begin(), cursorHistory.end(), Point(initialPos.x, initialPos.y));
    
    bool clickDetected = false;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        flip(frame, frame, 1); // Mirror the image
        
        detectGestures(frame, clickDetected);
        
        // Visual feedback for click
        if (clickDetected) {
            rectangle(frame, Rect(0, 0, frame.cols, frame.rows), Scalar(0, 0, 255), 10);
        }
        
        imshow("Gesture Control", frame);
        
        if (waitKey(30) >= 0) break;
    }
    
    return 0;
}
