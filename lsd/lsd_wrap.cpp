/*
 * lsd_wrap.cpp
 *
 *  Created on: May 20, 2013
 *      Author: gary
 */
#include "lsd_wrap.hpp"
#include "lsd.hpp"

using namespace cv;
using namespace std;
using namespace lsdwrap;
/**
 * Find line segments
 *
 * @param I			Input image. CV_8UC1 or C3 (gray or color) allowed, will be converted to gray, CV_64C1
 * @param segments	Return: Vector of found line segments "seg"
 * @param Roi		Optional region of interest to compute segments in. Smaller the image, finer the segments found
 * @return			Return number of segments found (-1 => error)
 */
int LsdWrap::lsdw(Mat &I, std::vector<seg> &segments, cv::Rect *Roi)
{
	//CHECKS AND CONVERSTIONS
	if(I.empty())
		return -1;
	if(I.channels() == 3)
	{
		cv::cvtColor(I, Igray, CV_BGR2GRAY);
	}
	else if (I.channels() == 1)
	{
		Igray = I;  //Just point to it
	}
	else
		return -2;
	if(Roi)
	{
		Mat Itmp = Igray(*Roi).clone();
		Igray = Itmp;
	}
	Igray.convertTo(I64F,CV_64FC1);

	//DO LSD
    double *dp = I64F.ptr<double>(0);
    int X = I64F.cols;
    int Y = I64F.rows;
    int n;
    double *out = lsd(&n,dp,X,Y);
    double *op = out;

    //RECORD THE RESULTS:
    if(0 == n) { segments.clear(); return 0;}
    segments.resize(n);
    vector<seg>::iterator it = segments.begin(), eit = segments.end();
    if(Roi) //Taking a subset of the image
    {
    	float ox = (float)(Roi->x);//offsets to the original boundaries
    	float oy = (float)(Roi->y);
    	for(; it != eit; ++it)
    	{
    		(*it).x1 = (float)(*op++) + ox;
    		(*it).y1 = (float)(*op++) + oy;
    		(*it).x2 = (float)(*op++) + ox;
    		(*it).y2 = (float)(*op++) + oy;
    		(*it).width = (float)(*op++);
    		(*it).p = (float)(*op++);
    		(*it).NFA = (float)(*op++);
    	}
    }
    else //Do the whole image
    {
    	for(; it != eit; ++it)
    	{
    		(*it).x1 = (float)(*op++);
    		(*it).y1 = (float)(*op++);
    		(*it).x2 = (float)(*op++);
    		(*it).y2 = (float)(*op++);
    		(*it).width = (float)(*op++);
    		(*it).p = (float)(*op++);
    		(*it).NFA = (float)(*op++);
    	}
    }
    free( (void *) out ); //Clean up internal allocations
    return n;
}

/**
 * Subdivide image and call by parts
 * Basically, this just creates Rois to break up the image into div_factor x div_factor pieces, then calls lsd(I,segmetns,ROI) above
 *
 * @param I				Input image, will be converted to gray, CV_64C1
 * @param segments		Return: Vector of found line segments "seg"
 * @param div_factor	How many parts should image be divided into in each dimension? 2 => 4 parts, 3=>9 etc. [1,I.rows()/2]
 * @return				Return number of segments found
 */
int  LsdWrap::lsd_subdivided(Mat &I, std::vector<seg> &segments, int div_factor)
{
	int width = I.cols;
	int height = I.rows;
	segments.clear();
	if((div_factor < 1)||(div_factor > height/2))
		div_factor = 1;
	int dw = width/div_factor;
	int dh = height/div_factor;
	int dwo = dw/20; //overlap
	int dho = dh/20; //overlap
	int N = 0;
	vector<seg> segs;
	for(int x = 0; x < width; x += dw)
	{
		int w = dw + dwo;
		if(x+w >= width)
		{
			int dx = x+w-width;
			x -= dx/2;
			w = width - 1 - x;
		}
		for(int y = 0; y < height; y += dh)
		{
			int h = dh + dho;
			if(h+y >= height)
			{
				int dy = h+y-height;
				y -= dy/2;
				h = height - 1 - y;
			}
			Rect R(x,y,w,h);
			N += lsdw(I,segs,&R);
			segments.insert(segments.end(),segs.begin(),segs.end());
		}
	}
	return N;
}

/**
 * Visualize segments
 *
 * @param name		Name of window to display in when you declared it outside with void namedWindow(const string& winname)
 * @param gray		Image, CV_8UC1.  Will convert it to color image so that it can paint segments in red
 * @param segments  Segments found from a call to lsd(...)
 */
void LsdWrap::imshow_segs(const std::string &name, Mat &gray, std::vector<seg> &segments)
{
	Mat Ig;
	if(gray.empty())
		return;
	if(gray.channels() == 3)
	{
		cv::cvtColor(gray, Ig, CV_BGR2GRAY);
	}
	else if (gray.channels() == 1)
	{
		Ig = gray;  //Just point to it
	}
	else
		return;
	//Create 3 channel image so we can draw colored line segments on the gray scale image
	vector<Mat> planes;
	planes.push_back(Ig);
	planes.push_back(Ig);
	planes.push_back(Ig);
	//Merge them into a color image
	Mat combine;
	merge(planes,combine);
	//draw segments ...
	vector<seg>::iterator it = segments.begin(), eit = segments.end();
	for(; it != eit; ++it)
	{
		Point p1((int)((*it).x1 + 0.5), (int)((*it).y1 + 0.5));
		Point p2((int)((*it).x2 + 0.5), (int)((*it).y2 + 0.5));
		line(combine,p1,p2,Scalar(0,0,255),2);
	}
	//and display ...
	imshow(name.c_str(),combine);
}


/**
 * To help in unit tests, compare 2 different segment finds
 * @param seg1		Vector1 of segments (in Blue)
 * @param seg2		Vector2 of segments (in Red)
 * @param size		Size of images that were used to find the segment
 * @param name		Optional name of window to display results in
 * @param I			Optional pointer. If not null, then draw image in namedWindow(name).
 * @return			How many points do not overlap between segments in seg1 and seg2
 */
int LsdWrap::CompareSegs(std::vector<seg> &seg1,std::vector<seg> &seg2,
		cv::Size size, const std::string &name, cv::Mat *I)
{
	//SET UP AND SIZE
	Mat I1, I2;
	if(I && size != I->size()) size = I->size();
	I1 = Mat(size,CV_8UC1,Scalar::all(0));
	I2 = Mat(size,CV_8UC1,Scalar::all(0));

	//DRAW SEGMENTS
	vector<seg>::iterator it = seg1.begin(), eit = seg1.end();
	for(; it != eit; ++it)
	{
		Point p1((int)((*it).x1 + 0.5), (int)((*it).y1 + 0.5));
		Point p2((int)((*it).x2 + 0.5), (int)((*it).y2 + 0.5));
		line(I1,p1,p2,Scalar::all(255),1);
	}
	it = seg2.begin();
	eit = seg2.end();
	for(; it != eit; ++it)
	{
		Point p1((int)((*it).x1 + 0.5), (int)((*it).y1 + 0.5));
		Point p2((int)((*it).x2 + 0.5), (int)((*it).y2 + 0.5));
		line(I2,p1,p2,Scalar::all(255),1);
	}

	//MARKPLACES WHERE THEY DO NOT AGREE
	Mat Ixor;
	bitwise_xor(I1, I2, Ixor);
	int N = countNonZero(Ixor);  //This is the missmatch number

	//DRAW IT IF ASKED
	if(I)
	{
		Mat Ig;
		if(size != I->size())
		{
			*I = Mat(size,CV_8UC1,Scalar::all(0));
		}
		if(I->channels() == 3)
		{
			cv::cvtColor(*I, Ig, CV_BGR2GRAY);
		}
		else if (I->channels() == 1)
		{
			Ig = *I;  //Just point to it
		}
		else
			return -2;
		//Create 3 channel image so we can draw colored line segments on the gray scale image
		vector<Mat> planes;
		planes.push_back(I1);
		planes.push_back(Ig);
		planes.push_back(I2);
		//Merge them into a color image
		Mat combine;
		merge(planes,combine);
		imshow(name.c_str(),combine); //Show it
	}

	//DONE
	return(N);
}

/**
 * To help in unit tests, compare 2 different segments, one from a gray image, 1 from a
 * list & give number of differing pixels
 * @param Ig		CV_8UC1 image of drawn segments (Blue)
 * @param seg2		Vector2 of segments (Red)
 * @param size		Size of images that were used to find the segment
 * @param name		Optional name of window to display results in
 * @param I			Optional pointer. If not null, then draw image in namedWindow(name).
 * @return			How many points do not overlap between segments in seg1 and seg2
 */
int LsdWrap::CompareSegs(cv::Mat &Ig, std::vector<seg> &seg2, const std::string &name, cv::Mat *I)
{
	//SET UP AND SIZE
	Mat I1,I2;

	//GET SEGMETNS 1
	if(Ig.empty())
		return -1;
	if(Ig.channels() == 3)
	{
		cv::cvtColor(Ig, I1, CV_BGR2GRAY);
	}
	else if (Ig.channels() == 1)
	{
		I1 = Ig;  //Just point to it
	}
	else
		return -2;
	I2 = Mat(Ig.size(),CV_8UC1,Scalar::all(0));

	//DRAW SEGMENTS 2
	vector<seg>::iterator it = seg2.begin(), eit = seg2.end();
	for(; it != eit; ++it)
	{
		Point p1((int)((*it).x1 + 0.5), (int)((*it).y1 + 0.5));
		Point p2((int)((*it).x2 + 0.5), (int)((*it).y2 + 0.5));
		line(I2,p1,p2,Scalar::all(255),1);
	}

	//MARKPLACES WHERE THEY DO NOT AGREE
	Mat Ixor;
	bitwise_xor(I1, I2, Ixor);
	int N = countNonZero(Ixor);  //This is the missmatch number

	//DRAW IT IF ASKED
	if(I)
	{
		if(Ig.size() != I->size())
		{
			*I = Ig.clone();
			I->setTo(Scalar::all(0));
		}
		Mat Igreen;
		if(I->channels() == 3)
		{
			cv::cvtColor(*I, Igreen, CV_BGR2GRAY);
		}
		else if (I->channels() == 1)
		{
			Igreen = *I;  //Just point to it
		}
		else
			return -2;
		//Create 3 channel image so we can draw colored line segments on the gray scale image
		vector<Mat> planes;
		planes.push_back(I1);
		planes.push_back(Igreen);
		planes.push_back(I2);
		//Merge them into a color image
		Mat combine;
		merge(planes,combine);
		imshow(name.c_str(),combine); //Show it
	}

	//DONE
	return(N);
}


/**
 * Pass in segments and image size, return segments drawn on 8UC1 image
 * @param seg1		Vector of segments to be drawn
 * @param size		width & height of image
 * @return			Drawn image segments on CV_8UC1 image
 */
cv::Mat LsdWrap::segments_to_image8UC1(std::vector<seg> &seg1, cv::Size size)
{
	Mat I1(size,CV_8UC1,Scalar::all(0)); //allocate an image, set it to zero

	//DRAW SEGMENTS
	vector<seg>::iterator it = seg1.begin(), eit = seg1.end();
	for(; it != eit; ++it)
	{
		Point p1((int)((*it).x1 + 0.5), (int)((*it).y1 + 0.5));
		Point p2((int)((*it).x2 + 0.5), (int)((*it).y2 + 0.5));
		line(I1,p1,p2,Scalar::all(255),1);
	}
	return I1;
}

