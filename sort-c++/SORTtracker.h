///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//  
//  This is a C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//  Cong Ma, mcximing@sina.cn, 2016
//  Rewritten by Beloborodov Dmitri (BeloborodovDS), 2017
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////

#ifndef SORT_TRACKER
#define SORT_TRACKER

#include "Hungarian.h"
#include "KalmanTracker.h"

#include "opencv2/core/core.hpp"
#include <set>

using namespace std;

// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);


struct TrackingBox
{
  int frame;
  int id;
  int age;
  Rect_<float> box;
};


class SORTtracker{
  
public:
  
  int frame_count;
  int max_age;
  int min_hits;
  double iouThreshold;
  
  vector<KalmanTracker> trackers;
  
  vector<Rect_<float> > predictedBoxes;
  vector<vector<double> > iouMatrix;
  vector<int> assignment;
  set<int> unmatchedDetections;
  set<int> unmatchedTrajectories;
  set<int> allItems;
  set<int> matchedItems;
  vector<cv::Point> matchedPairs;
  unsigned int trkNum;
  unsigned int detNum;
  
  SORTtracker(int maxage=1, int minhits=3, float iou_thresh=0.3);
  
  ~SORTtracker();
  
  void init(vector<Rect_<float> > detections);
  
  void step(vector<Rect_<float> > detections, vector<TrackingBox> &result);
  
  
};

#endif