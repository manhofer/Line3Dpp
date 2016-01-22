/*
 * lsd_wrap.hpp
 *
 *  Created on: May 17, 2013
 *      Author: gary
 *
 *  Just a simple wrap of LSD for now. If LSD is used extensively, it needs a deep re-write.
 *  published
 *   2012-03-24
 *   reference
 *     Rafael Grompone von Gioi, Jérémie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
 *     LSD: a Line Segment Detector, Image Processing On Line, 2012. http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd
 *
 *  Notes
 *  5/18/2013  Author of LSD, rafael grompone von gioi <grompone@gmail.com>, has allowed the code to be BSD for OpenCV. I will
 *             allocate Google Summer of Code students on this.
 *
 *    Calling example snippet:
 *    =======================
  namedWindow("ImageL");
  namedWindow("ImageS");
  namedWindow("Segs");
  String sseg("Segs");
  int num_images = file.size();
  Mat I,Isegs;
  int64 t_;
  double sum_dt_ = 0, tick_freq_ = (double)getTickFrequency();
  vector<seg> segmentsSub,segmentsWhole;
//  Mat foo;
  for(int i = 0; i<num_images; ++i)
  {
      printf("Image %d of %d\n",i,num_images);
      I = imread(file[i],CV_LOAD_IMAGE_GRAYSCALE );
      t_ = getTickCount();
//      int NL = ls.lsdw(I,segmentsWhole);
      int NS = ls.lsd_subdivided(I,segmentsSub, 7);
      sum_dt_ += (double)(getTickCount() - t_);
      printf("NS = %d\n",NS);
//      ls.imshow_segs(string("ImageL"),I,segmentsWhole);
      ls.imshow_segs(string("ImageS"),I,segmentsSub);
//      I.copyTo(Isegs);
//      Isegs.setTo(Scalar::all(0));
// 	  	if(i > 0) ls.CompareSegs(foo,segmentsSub,string("Segs"),&Isegs); //Blue L, Red S
//      foo  = ls.segments_to_image8UC1(segmentsWhole,I.size());
//      ls.CompareSegs(segmentsWhole,segmentsSub,I.size(),string("Segs"),&Isegs); //Blue L, Red S

      //HANDLE KEY INPUT AND PAUSE
      int k = 0xFF&waitKey(0);
 *
 *    ================================
 *
 */

#ifndef LSD_WRAP_HPP_
#define LSD_WRAP_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

//#pragma hdrstop


namespace lsdwrap {

/**
 * seg Line segment structure
 * A double array of size 7 x n_out, containing the list
 *    of line segments detected. The array contains first
 *    7 values of line segment number 1, then the 7 values
 *    of line segment number 2, and so on, and it finish
 *    by the 7 values of line segment number n_out.
 *    The seven values are:
 *    - x1,y1,x2,y2,width,p,-log10(NFA)
 *    .
 *    for a line segment from coordinates (x1,y1) to (x2,y2),
 *        NOTE: This is an oriented dark to light edge from pt1 -> pt2. +90degrees from the angle pt1->pt2 is the dark side
 *    a width 'width',
 *    an angle precision of p in (0,1) given by angle_tolerance/180 degree, and
 *    Number of False Alarms (NFA) 0 <= NFA <= 2. Smaller is better
 */
typedef struct seg_s
{
	float x1;		//Start point of a segment
	float y1;
	float x2;		//End point of a segment.  +90degree from the line from pt1 -> pt2 points into dark
	float y2;
	float width;	//The width of the rectangle that bounds the segment. Smaller is better
	float p;		//Angle tollerance, default is 22.5 deg/180 deg
	float NFA;		//Number of False Alarms (NFA) 0 <= NFA <= 2. Smaller is better
} seg;


/**
 * For rotation invariance, this class calls BarPose 3 times: 0 deg, 45 deg and 90 deg
 */

///////////////////////////////////////////////////////////////////////////////////
// BARCODE DETECTION CLASS ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

class LsdWrap
{
public:
	cv::Mat I64F;	 //Will convert lsd(I,...) to a gray 64 bit image here. If not allocated or the wrong size, will reallocate
	cv::Mat Igray;   //Convert or point to gray. On multiple use, if same size, it saves on allocation

	/**
	 * Find line segments
	 *
	 * @param I			Input image, will be converted to gray, CV_64C1
	 * @param segments	Return: Vector of found line segments "seg"
	 * @param Roi		Optional region of interest to compute segments in. Smaller the image, finer the segments found
	 * @return			Return number of segments found
	 */
	int lsdw(cv::Mat &I, std::vector<seg> &segments, cv::Rect *Roi = 0);

	/**
	 * Subdivide image and call by parts
	 * Basically, this just creates Rois to break up the image into div_factor x div_factor pieces, then calls lsd(I,segmetns,ROI) above
	 *
	 * @param I				Input image, will be converted to gray, CV_64C1
	 * @param segments		Return: Vector of found line segments "seg"
	 * @param div_factor	How many parts should image be divided into in each dimension? 2 => 4 parts, 3=>9 etc. [1,I.rows()/2]
	 * @return				Return number of segments found
	 */
	int lsd_subdivided(cv::Mat &I, std::vector<seg> &segments, int div_factor = 1);

	/**
	 * Visualize segments
	 *
	 * @param name		Name of window to display in when you declared it outside with void namedWindow(const string& winname)
	 * @param gray		Image, CV_8UC1.  Will convert it to color image so that it can paint segments in red
	 * @param segments  Segments found from a call to lsd(...)
	 */
	void imshow_segs(const std::string &name, cv::Mat &gray, std::vector<seg> &segments);

	/**
	 * Pass in segments and image size, return segments drawn on 8UC1 image
	 * @param seg1		Vector of segments to be drawn
	 * @param size		width & height of image
	 * @return			Drawn image segments on CV_8UC1 image
	 */
	cv::Mat segments_to_image8UC1(std::vector<seg> &seg1, cv::Size size);

	/**
	 * To help in unit tests, compare 2 different segments, one from a gray image, 1 from a list & give number of differing pixels
	 * @param Ig		CV_8UC1 image of drawn segments (drawn in blue)
	 * @param seg2		Vector2 of segments (drawn in red)
	 * @param size		Size of images that were used to find the segment
	 * @param name		Optional name of window to display results in
	 * @param I			Optional pointer. If not null, then draw image in namedWindow(name).
	 * @return			How many points do not overlap between segments in seg1 and seg2
	 */
	int CompareSegs(cv::Mat &Ig, std::vector<seg> &seg2, const std::string &name, cv::Mat *I = 0);


	/**
	 * To help in unit tests, compare 2 different segment finds
	 * @param seg1		Vector1 of segments (drawn in blue)
	 * @param seg2		Vector2 of segments (drawn in red)
	 * @param size		Size of images that were used to find the segment
	 * @param name		Optional name of window to display results in
	 * @param I			Optional pointer. If not null, then draw image in namedWindow(name).
	 * @return			How many points do not overlap between segments in seg1 and seg2
	 */
	int CompareSegs(std::vector<seg> &seg1,std::vector<seg> &seg2,
			cv::Size size, const std::string &name, cv::Mat *I = 0);

}; //end class

} //Namespace lsdwrap

#endif /* LSD_WRAP_HPP_ */


