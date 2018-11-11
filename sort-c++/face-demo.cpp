#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "SORTtracker.h"

#include <iostream>

using namespace std;
using namespace cv;

#define VIDEO_WIDTH 		               360
#define VIDEO_HEIGHT 		               240
#define BB_MIN_FACE		               15
#define BB_MAX_FACE		               150
#define NUM_COLORS 				20

int main()
{
  //setup tracker
  int max_age = 10;
  int min_hits = 10;
  SORTtracker tracker(max_age, min_hits, 0.05);
  
  //flag used to init tracker
  bool first_detections = true;
  
  //colors for faces ids
  RNG rng(0xFFFFFFFF);
  Scalar_<int> randColor[NUM_COLORS];
  for (int i = 0; i < NUM_COLORS; i++)
    rng.fill(randColor[i], RNG::UNIFORM, 0, 256);
  
  //open camera
  VideoCapture cap;
  if(!cap.open(0))
  {
    cout<<"Cannot open camera with OpenCV!"<<endl;
    return 0;
  }
  cout << "Connected to camera\n";
  
  //Init face detector
  CascadeClassifier detector;
  if(!detector.load("./lbpcascade_frontalface.xml"))
  {
    cout<<"Cannot load detector"<<endl;
    return 0;
  }
  cout<<"Face detector loaded"<<endl;
  
  //detections and tracked faces
  vector<Rect> detections;
  vector<Rect_<float> > tmp_det;
  vector< TrackingBox > tracked_faces;
  
  for(;;)
  {
    // get a new frame from camera
    Mat frame, showframe;
    cap >> frame; 
    
    //process frame
    resize(frame, frame, Size(VIDEO_WIDTH,VIDEO_HEIGHT));
    flip(frame, frame, 1);
    frame.copyTo(showframe);
    cvtColor(frame, frame, CV_BGR2GRAY);
    
    //detect faces
    detections.clear();
    tmp_det.clear();
    detector.detectMultiScale(frame, detections, 1.1, 3, 0, Size(BB_MIN_FACE,BB_MIN_FACE), Size(BB_MAX_FACE,BB_MAX_FACE));
    for (int i=0; i<detections.size(); i++)
    {
      Rect_<float> r = detections[i];
      tmp_det.push_back(r);
    }
    
    //if first detections ever: init tracker if possible
    if (detections.size()>0 && first_detections)
    {
      tracker.init(tmp_det);
      first_detections = false;
      cout<<"INIT\n";
    }
    
    //if tracker initialized, track faces
    if (!first_detections)
    {
      tracked_faces.clear();
      tracker.step(tmp_det, tracked_faces);
    }
    
    
    //draw rectangles
    for (int i=0; i<tracked_faces.size(); i++)
    {
      //face "age" shows how many frames this face was missing
      //face color darkens with increasing age to black
      double alpha = ((float)max_age-tracked_faces[i].age)/max_age;
      Scalar_<int> intcol = randColor[tracked_faces[i].id % NUM_COLORS];
      Scalar col = Scalar(intcol[0],intcol[1],intcol[2]);
      col = alpha*col + (1-alpha)*Scalar(0,0,0);
      rectangle(showframe, tracked_faces[i].box, col, 3); 
    }
    
    //show frame
    imshow("tracking", showframe);
    if(cvWaitKey(1)!=-1)
    {
        break;
    }
  }
  
  cout<<"OK\n";
}
