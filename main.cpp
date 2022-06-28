#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <cinttypes>
using namespace std;

// Baumer SDK : camera SDK
#include "bgapi.hpp"

// OPENCV : display preview and save video 
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"


// Boost : parse config file / create thread for preview
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "boost/thread.hpp"
#include "boost/chrono.hpp"
namespace po = boost::program_options;
namespace fs = boost::filesystem;
#include <armadillo>


// PARAMETERS-------------------------------------------------------------------
int preview;				
int subsample;
string result_dir;			

// Global variables-------------------------------------------------------------
int sys = 0;
int cam = 0;
BGAPI::System * pSystem = NULL;
BGAPI::Camera * pCamera = NULL;
BGAPI::Image ** pImage = NULL;
cv::VideoWriter writer;
ofstream file;
ofstream skelfile;
BGAPIX_TypeINT iTimeHigh, iTimeLow, iFreqHigh, iFreqLow;
BGAPIX_CameraImageFormat cformat; 
cv::Mat img_display;
cv::Mat img_skel;
cv::Mat background;
std::vector<cv::Point> points;
uint64_t first_ts = 0;
uint64_t current_ts = 0;
uint64_t previous_ts = 0;
int roi_left = 0;	            
int roi_top = 0;
int roi_right = 0;
int roi_bottom = 0;
int height = 0;		            
int width = 0;			
int gainvalue = 0;
int gainmax = 0;
int exposurevalue = 0;
int exposuremax = 0;
int triggers = 0;
int fps = 0;
int packetsizevalue = 0;
int fpsmax = 0;
int formatindex = 0;
int formatindexmax = 0;
uint32_t numbuffer = 0;
uint32_t exposuremax_slider = 0;
int fpsmax_slider = 0;

int n_skel = 5;
int length_tail = 180;

// memory buffer
list<cv::Mat> ImageList; // list of incoming images from camera
list<cv::Mat> SkeletonList; // list of images to compute skeleton
list<double> timeStampsList; // list of image timings
list<int> counterList; // list of image timings
list<double> fpsList; // list of image timings

boost::mutex mtx_skel;
boost::mutex mtx_buffer;
boost::mutex mtx;

int read_config(int ac, char* av[]) {
	try {
		string config_file;

		// only on command line
		po::options_description generic("Generic options");
		generic.add_options()
			("help", "produce help message")
			("config,c", po::value<string>(&config_file)->default_value("behavior.cfg"),
				"configuration file")
			;

		// both on command line and config file
		po::options_description config("Configuration");
		config.add_options()
			("preview,p", po::value<int>(&preview)->default_value(1), "preview")
			("subsample,b", po::value<int>(&subsample)->default_value(1), "subsampling factor")
			("left", po::value<int>(&roi_left)->default_value(1), "ROI left")
			("top", po::value<int>(&roi_top)->default_value(1), "ROI top")
			("right", po::value<int>(&roi_right)->default_value(1), "ROI right")
			("bottom", po::value<int>(&roi_bottom)->default_value(1), "ROI bottom")
			("formatindex", po::value<int>(&formatindex)->default_value(0), "image format")
			("gain", po::value<int>(&gainvalue)->default_value(0), "gain")
			("exposure", po::value<int>(&exposurevalue)->default_value(3000), "exposure")
			("triggers", po::value<int>(&triggers)->default_value(0), "triggers")
			("fps", po::value<int>(&fps)->default_value(300), "fps")
			("result_dir,d", po::value<string>(&result_dir)->default_value(""), "result directory")
			("numbuffer,n", po::value<uint32_t>(&numbuffer)->default_value(100), "buffer size")
			("packetsize", po::value<int>(&packetsizevalue)->default_value(576), "buffer size")
			("exposuremax,e", po::value<uint32_t>(&exposuremax_slider)->default_value(3000), "max exposure slider")
			("fpsmax,f", po::value<int>(&fpsmax_slider)->default_value(300), "max fps slider")
			("n_skel", po::value<int>(&n_skel)->default_value(5), "number of skeleton points")
			("length_tail", po::value<int>(&length_tail)->default_value(180), "length of the tail")
			;

		po::options_description cmdline_options;
		cmdline_options.add(generic).add(config);

		po::options_description config_file_options;
		config_file_options.add(config);

		po::options_description visible("Allowed options");
		visible.add(generic).add(config);

		po::variables_map vm;
		store(po::command_line_parser(ac, av).options(cmdline_options).run(), vm);
		notify(vm);

		ifstream ifs(config_file.c_str());
		if (!ifs) {
			cout << "can not open config file: " << config_file << "\n";
			return 1;
		}
		else {
			store(parse_config_file(ifs, config_file_options), vm);
			notify(vm);
		}

		if (vm.count("help")) {
			cout << visible << "\n";
			return 2;
		}

		width = roi_right - roi_left;
		height = roi_bottom - roi_top;

		cout << "Running with following options " << endl
			<< "  Preview: " << preview << endl
			<< "  Subsample: " << subsample << endl
			<< "  Left: " << roi_left << endl
			<< "  Top: " << roi_top << endl
			<< "  Right: " << roi_right << endl
			<< "  Bottom: " << roi_bottom << endl
			<< "  Width: " << width << endl
			<< "  Height: " << height << endl
			<< "  Format Index: " << formatindex << endl
			<< "  Gain: " << gainvalue << endl
			<< "  Exposure: " << exposurevalue << endl
			<< "  Triggers: " << triggers << endl
			<< "  FPS: " << fps << endl
			<< "  Result directory: " << result_dir << endl
			<< "  Buffer size: " << numbuffer << endl
			<< "  Packet size: " << packetsizevalue << endl
			<< "  Max exposure slider: " << exposuremax_slider << endl
			<< "  Max fps slider: " << fpsmax_slider << endl
			<< "  Number of skeleton points: " << n_skel << endl
			<< "  Length of the tail: " << length_tail << endl;
			
	}
	catch (exception& e)
	{
		cout << e.what() << "\n";
		return 1;
	}
	return 0;
}
    
BGAPI_RESULT BGAPI_CALLBACK imageCallback(void * callBackOwner, BGAPI::Image* pCurrImage)
{
	cv::Mat img;
	cv::Mat img_resized;
	int swc;
	int hwc;
	int timestamplow = 0;
	int timestamphigh = 0;
	uint32_t timestamplow_u = 0;
	uint32_t timestamphigh_u = 0;
	BGAPI_RESULT res = BGAPI_RESULT_OK;

	unsigned char* imagebuffer = NULL;
	res = pCurrImage->get(&imagebuffer);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Image::get Errorcode: %d\n", res);
		return 0;
	}

	//TODO: print image counters somewhere
	res = pCurrImage->getNumber(&swc, &hwc);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Image::getNumber Errorcode: %d\n", res);
		return 0;
	}

	res = pCurrImage->getTimeStamp(&timestamphigh, &timestamplow);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Image::getTimeStamp Errorcode: %d\n", res);
		return 0;
	}
	timestamplow_u = timestamplow;
	timestamphigh_u = timestamphigh;
	if (swc == 0) {
		first_ts = (uint64_t) timestamphigh_u << 32 | timestamplow_u;
	}
	current_ts = (uint64_t) timestamphigh_u << 32 | timestamplow_u;
	double current_time = (double)(current_ts - first_ts) / (double)iFreqLow.current;
	double fps_hat = (double)iFreqLow.current / (double)(current_ts - previous_ts);
	previous_ts = current_ts;

	img = cv::Mat(cv::Size(width, height), CV_8U, imagebuffer);

	mtx_buffer.lock();
	// add current image and timestamp to buffers
	ImageList.push_back(img.clone()); // image memory buffer
	timeStampsList.push_back(current_time); // timing buffer
	counterList.push_back(swc);
	fpsList.push_back(fps_hat);
	mtx_buffer.unlock();

	res = ((BGAPI::Camera*)callBackOwner)->setImage(pCurrImage);
	if (res != BGAPI_RESULT_OK) {
		printf("setImage failed with %d\n", res);
	}
	return res;
}

static void trackbar_callback(int,void*) {

	BGAPI_RESULT res = BGAPI_RESULT_FAIL;
	BGAPI_FeatureState state; 
	BGAPIX_TypeROI roi;
	BGAPIX_TypeRangeFLOAT gain;
	BGAPIX_TypeRangeINT exposure;
	BGAPIX_TypeRangeFLOAT framerate;
	BGAPIX_TypeListINT imageformat;
	BGAPI_Resend resendvalues;
	BGAPIX_TypeRangeFLOAT sensorfreq;
	BGAPIX_TypeINT readouttime;
	BGAPIX_TypeRangeINT packetsize;


	state.cbSize = sizeof(BGAPI_FeatureState);
	roi.cbSize = sizeof(BGAPIX_TypeROI);
	gain.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	exposure.cbSize = sizeof(BGAPIX_TypeRangeINT);
	framerate.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	imageformat.cbSize = sizeof(BGAPIX_TypeListINT);
	resendvalues.cbSize = sizeof(BGAPI_Resend);
	sensorfreq.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	readouttime.cbSize = sizeof(BGAPIX_TypeINT);
	packetsize.cbSize = sizeof(BGAPIX_TypeRangeINT);

	// FORMAT INDEX : this goes first ?
	res = pCamera->setImageFormat(formatindex);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setImageFormat Errorcode: %d\n", res);
	}

	res = pCamera->getImageFormat(&state, &imageformat);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setImageFormat Errorcode: %d\n", res);
	}
	formatindex = imageformat.current;

	res = pCamera->getImageFormatDescription(formatindex, &cformat);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getImageFormatDescription Errorcode: %d\n", res);
	}

	// ROI
	// check dimensions 
	if ((roi_left + width > cformat.iSizeX) || (roi_top + height > cformat.iSizeY)) {
		printf("Image size is not compatible with selected format\n");
	}

	res = pCamera->setPartialScan(1, roi_left, roi_top, roi_left + width, roi_top + height);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setPartialScan Errorcode: %d\n", res);
	}

	res = pCamera->getPartialScan(&state, &roi);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getImageFormat Errorcode: %d\n", res);
	}

	roi_left = roi.curleft;
	roi_top = roi.curtop;
	roi_right = roi.curright;
	roi_bottom = roi.curbottom;
	width = roi.curright - roi.curleft;
	height = roi.curbottom - roi.curtop;

	// change size of display accordingly
	mtx.lock();
	img_display = cv::Mat(height / subsample, width / subsample, CV_8UC1);
	mtx.unlock();
	// change image size -> detach and reallocate images (only if using external buffer)  

	// GAIN
	res = pCamera->setGain(gainvalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setGain Errorcode: %d\n", res);
	}

	res = pCamera->getGain(&state, &gain);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setGain Errorcode: %d\n", res);
	}
	gainvalue = gain.current;

	// EXPOSURE 
	res = pCamera->setExposure(exposurevalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setExposure Errorcode: %d\n", res);
	}

	res = pCamera->getExposure(&state, &exposure);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setExposure Errorcode: %d\n", res);
	}
	exposurevalue = exposure.current;

	// TRIGGERS
	if (triggers) {
		res = pCamera->setTriggerSource(BGAPI_TRIGGERSOURCE_HARDWARE1);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerSource Errorcode: %d\n", res);
		}

		res = pCamera->setTrigger(true);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTrigger Errorcode: %d\n", res);
		}

		res = pCamera->setTriggerActivation(BGAPI_ACTIVATION_RISINGEDGE);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerActivation Errorcode: %d\n", res);
		}

		res = pCamera->setTriggerDelay(0);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerDelay Errorcode: %d\n", res);
		}
	}
	else {
		res = pCamera->setTriggerSource(BGAPI_TRIGGERSOURCE_SOFTWARE);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerSource Errorcode: %d\n", res);
		}

		res = pCamera->setTrigger(false);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTrigger Errorcode: %d\n", res);
		}

		// FPS: maybe do that only in preview mode without triggers ?
		res = pCamera->setFramesPerSecondsContinuous(fps);
		if (res != BGAPI_RESULT_OK) {
			printf("BGAPI::Camera::setFramesPerSecondsContinuous Errorcode: %d\n", res);
		}

		res = pCamera->getFramesPerSecondsContinuous(&state, &framerate);
		if (res != BGAPI_RESULT_OK) {
			printf("BGAPI::Camera::getFramesPerSecondsContinuous Errorcode: %d\n", res);
		}
		fps = framerate.current;
	}
	res = pCamera->getTrigger(&state);
	if (res != BGAPI_RESULT_OK)
	{
		printf("BGAPI::Camera::getTrigger Errorcode: %d\n", res);
	}
	triggers = state.bIsEnabled;

	res = pCamera->getReadoutTime(&state, &readouttime);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getReadoutTime Errorcode: %d\n", res);
	}
	cout << "Readout time: " << readouttime.current << endl;
	
	cv::setTrackbarMax("ROI height", "Controls", cformat.iSizeY);
	cv::setTrackbarMax("ROI width", "Controls", cformat.iSizeX);
	cv::setTrackbarMax("ROI left", "Controls", cformat.iSizeX);
	cv::setTrackbarMax("ROI top", "Controls", cformat.iSizeY);
}

void display_preview() {

	cv::namedWindow("Preview", cv::WINDOW_AUTOSIZE);
	if (preview) {
		cv::namedWindow("Controls", cv::WINDOW_NORMAL);
		cv::createTrackbar("ROI left", "Controls", &roi_left, cformat.iSizeX, trackbar_callback);
		cv::createTrackbar("ROI top", "Controls", &roi_top, cformat.iSizeY, trackbar_callback);
		cv::createTrackbar("ROI width", "Controls", &width, cformat.iSizeX, trackbar_callback);
		cv::createTrackbar("ROI height", "Controls", &height, cformat.iSizeY, trackbar_callback);
		cv::createTrackbar("Exposure", "Controls", &exposurevalue, exposuremax_slider, trackbar_callback);
		cv::createTrackbar("Gain", "Controls", &gainvalue, gainmax, trackbar_callback);
		cv::createTrackbar("FPS", "Controls", &fps, fpsmax_slider, trackbar_callback);
		cv::createTrackbar("Triggers", "Controls", &triggers, 1, trackbar_callback);
		cv::createTrackbar("Format Index", "Controls", &formatindex, formatindexmax, trackbar_callback);
	}
	while (true) {
		mtx.lock();
		cv::imshow("Preview", img_display);
		cv::imshow("Skeleton", img_skel);
		mtx.unlock();
		cv::waitKey(32);
		boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
	}
}

int setup_camera() {
	
	BGAPI_RESULT res = BGAPI_RESULT_FAIL;
	BGAPI_FeatureState state; 
	BGAPIX_TypeROI roi; 
	BGAPIX_TypeRangeFLOAT gain;
	BGAPIX_TypeRangeFLOAT framerate;
	BGAPIX_TypeRangeINT exposure;
	BGAPIX_TypeListINT imageformat;
	BGAPI_Resend resendvalues;
	BGAPIX_TypeRangeFLOAT sensorfreq;
	BGAPIX_TypeINT readouttime;
	BGAPIX_TypeRangeINT packetsize;
	BGAPIX_TypeListINT driverlist;

	cformat.cbSize = sizeof(BGAPIX_CameraImageFormat);
	state.cbSize = sizeof(BGAPI_FeatureState);
	iTimeHigh.cbSize = sizeof(BGAPIX_TypeINT);
	iTimeLow.cbSize = sizeof(BGAPIX_TypeINT);
	iFreqHigh.cbSize = sizeof(BGAPIX_TypeINT);
	iFreqLow.cbSize = sizeof(BGAPIX_TypeINT);
	roi.cbSize = sizeof(BGAPIX_TypeROI);
	gain.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	exposure.cbSize = sizeof(BGAPIX_TypeRangeINT);
	framerate.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	imageformat.cbSize = sizeof(BGAPIX_TypeListINT);
	resendvalues.cbSize = sizeof(BGAPI_Resend);
	sensorfreq.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	readouttime.cbSize = sizeof(BGAPIX_TypeINT);
	packetsize.cbSize = sizeof(BGAPIX_TypeRangeINT);
	driverlist.cbSize = sizeof(BGAPIX_TypeListINT);

	// Initializing the system--------------------------------------------------
	res = BGAPI::createSystem(sys, &pSystem);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::createSystem Errorcode: %d System index: %d\n", res, sys);
		return EXIT_FAILURE;
	}
	printf("Created system: System index %d\n", sys);

	res = pSystem->open();
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::System::open Errorcode: %d System index: %d\n", res, sys);
		return EXIT_FAILURE;
	}
	printf("System opened: System index %d\n", sys);

	res = pSystem->getGVSDriverModel(&state, &driverlist);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::System::getGVSDriverModel Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Available driver models: " << endl;
	for (int i = 0; i < driverlist.length; i++) {
		cout << driverlist.array[i] << endl;
	}
	cout << "Current driver models: " << driverlist.current << endl;

	res = pSystem->createCamera(cam, &pCamera);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::System::createCamera Errorcode: %d Camera index: %d\n", res, cam);
		return EXIT_FAILURE;
	}
	printf("Created camera: Camera index %d\n", cam);

	res = pCamera->open();
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::open Errorcode: %d Camera index: %d\n", res, cam);
		return EXIT_FAILURE;
	}
	printf("Camera opened: Camera index %d\n", cam);

	// CAMERA FEATURES ------------------------------------------------------

	// FORMAT INDEX
	res = pCamera->getImageFormat(&state, &imageformat);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setImageFormat Errorcode: %d\n", res);
	}
	formatindexmax = imageformat.length;

	if ((formatindex < 0) || (formatindex > formatindexmax)) {
		printf("Image size is not compatible with selected format\n");
		return EXIT_FAILURE;
	}

	res = pCamera->setImageFormat(formatindex);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setImageFormat Errorcode: %d\n", res);
	}

	res = pCamera->getImageFormat(&state, &imageformat);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setImageFormat Errorcode: %d\n", res);
	}
	formatindex = imageformat.current;

	// ROI
	// check dimensions 
	res = pCamera->getImageFormatDescription(formatindex, &cformat);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getImageFormatDescription Errorcode: %d\n", res);
	}

	if ((roi_left + width > cformat.iSizeX) || (roi_top + height > cformat.iSizeY)) {
		printf("Image size is not compatible with selected format\n");
		return EXIT_FAILURE;
	}

	res = pCamera->setPartialScan(1, roi_left, roi_top, roi_left + width, roi_top + height);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setPartialScan Errorcode: %d\n", res);
	}

	res = pCamera->getPartialScan(&state, &roi);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getImageFormat Errorcode: %d\n", res);
	}

	roi_left = roi.curleft;
	roi_top = roi.curtop;
	roi_right = roi.curright;
	roi_bottom = roi.curbottom;
	width = roi.curright - roi.curleft;
	height = roi.curbottom - roi.curtop;

	// change size of display accordingly
	mtx.lock();
	img_display = cv::Mat(height/subsample, width / subsample, CV_8UC1);
	mtx.unlock();

	// GAIN
	res = pCamera->getGain(&state, &gain);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setGain Errorcode: %d\n", res);
	}
	gainmax = gain.maximum;

	if ((gainvalue < 0) || (gainvalue > gainmax)) {
		printf("Gain value is incorrect\n");
		return EXIT_FAILURE;
	}

	res = pCamera->setGain(gainvalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setGain Errorcode: %d\n", res);
	}

	res = pCamera->getGain(&state, &gain);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setGain Errorcode: %d\n", res);
	}
	gainvalue = gain.current;

	// EXPOSURE
	res = pCamera->getExposure(&state, &exposure);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setExposure Errorcode: %d\n", res);
	}
	exposuremax = exposure.maximum;

	if ((exposurevalue <= 0) || (exposurevalue > exposuremax)) {
		printf("Exposure value is incorrect\n");
		return EXIT_FAILURE;
	}

	res = pCamera->setExposure(exposurevalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setExposure Errorcode: %d\n", res);
	}

	res = pCamera->getExposure(&state, &exposure);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setExposure Errorcode: %d\n", res);
	}
	exposurevalue = exposure.current;

	// TRIGGERS
	if (triggers) {
		res = pCamera->setTriggerSource(BGAPI_TRIGGERSOURCE_HARDWARE1);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerSource Errorcode: %d\n", res);
		}

		res = pCamera->setTrigger(true);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTrigger Errorcode: %d\n", res);
		}

		res = pCamera->setTriggerActivation(BGAPI_ACTIVATION_RISINGEDGE);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerActivation Errorcode: %d\n", res);
		}

		res = pCamera->setTriggerDelay(0);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerDelay Errorcode: %d\n", res);
		}
	}
	else {
		res = pCamera->setTriggerSource(BGAPI_TRIGGERSOURCE_SOFTWARE);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerSource Errorcode: %d\n", res);
		}

		res = pCamera->setTrigger(false);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTrigger Errorcode: %d\n", res);
		}

		// FPS
		res = pCamera->getFramesPerSecondsContinuous(&state, &framerate);
		if (res != BGAPI_RESULT_OK) {
			printf("BGAPI::Camera::getFramesPerSecondsContinuous Errorcode: %d\n", res);
		}
		fpsmax = framerate.maximum;

		if ((fps <= 0) || (fps > fpsmax)) {
			printf("FPS continuous value is incorrect\n");
			return EXIT_FAILURE;
		}

		res = pCamera->setFramesPerSecondsContinuous(fps);
		if (res != BGAPI_RESULT_OK) {
			printf("BGAPI::Camera::setFramesPerSecondsContinuous Errorcode: %d\n", res);
		}

		res = pCamera->getFramesPerSecondsContinuous(&state, &framerate);
		if (res != BGAPI_RESULT_OK) {
			printf("BGAPI::Camera::getFramesPerSecondsContinuous Errorcode: %d\n", res);
		}
		fps = framerate.current;
	}
	res = pCamera->getTrigger(&state);
	if (res != BGAPI_RESULT_OK)
	{
		printf("BGAPI::Camera::getTrigger Errorcode: %d\n", res);
	}
	triggers = state.bIsEnabled;

	// READOUT
	res = pCamera->setReadoutMode(BGAPI_READOUTMODE_OVERLAPPED);
	if (res != BGAPI_RESULT_OK)
	{
		if (res == BGAPI_RESULT_FEATURE_NOTIMPLEMENTED) {
			printf("BGAPI::Camera::setReadoutMode not implemented, ignoring\n");
		}
		else {
			printf("BGAPI::Camera::setReadoutMode Errorcode: %d\n", res);
			return EXIT_FAILURE;
		}
	}

	// DIGITIZATION TAP
	res = pCamera->setSensorDigitizationTaps(BGAPI_SENSORDIGITIZATIONTAPS_SIXTEEN);
	if (res != BGAPI_RESULT_OK)
	{
		if (res == BGAPI_RESULT_FEATURE_NOTIMPLEMENTED) {
			printf("BGAPI::Camera::setSensorDigitizationTaps not implemented, ignoring\n");
		}
		else {
			printf("BGAPI::Camera::setSensorDigitizationTaps Errorcode: %d\n", res);
			return EXIT_FAILURE;
		}
	}

	// EXPOSURE MODE 
	// maybe change this to trigger width ?
	res = pCamera->setExposureMode(BGAPI_EXPOSUREMODE_TIMED);
	if (res != BGAPI_RESULT_OK)
	{
		if (res == BGAPI_RESULT_FEATURE_NOTIMPLEMENTED) {
			printf("BGAPI::Camera::setExposureMode not implemented, ignoring\n");
		}
		else {
			printf("BGAPI::Camera::setExposureMode Errorcode: %d\n", res);
			return EXIT_FAILURE;
		}
	}

	// TIME STAMPS
	res = pCamera->getTimeStamp(&state, &iTimeHigh, &iTimeLow, &iFreqHigh, &iFreqLow);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getTimeStamp Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	printf("Timestamps frequency [%d,%d]\n", iFreqHigh.current, iFreqLow.current);

	// For some reason this seems to freeze the hxg20nir
	/*
	res = pCamera->resetTimeStamp();
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::resetTimeStamp Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}*/

	res = pCamera->setFrameCounter(0, 0);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setFrameCounter Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}

	// Setting the right packet size is crucial for reliable performance
	// large packet size (7200 bytes) should be used for high-speed recording.
	// To allow the use of large packets, the network card must support 
	// "Jumbo frames" (this can be set in windows device manager)
	res = pCamera->setPacketSize(packetsizevalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setPacketSize Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}

	// WARNING the minimum and maximum packet size seem to not always
	// reflect  the actual max size for the network card/camera.
	// Do not trust those values.
	res = pCamera->getPacketSize(&state, &packetsize);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getPacketSize Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Packet size: "  << packetsize.current << ", Max: " << packetsize.maximum << ", Min: " << packetsize.minimum << endl;

	// Resend algorithm: default values are probably fine
	res = pCamera->getGVSResendValues(&state, &resendvalues);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getGVSResendValues Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Resend values: " << endl
		<< "\t MaxResendsPerImage: " << resendvalues.gigeresend.MaxResendsPerImage << endl
		<< "\t MaxResendsPerPacket: " << resendvalues.gigeresend.MaxResendsPerPacket << endl
		<< "\t FirstResendWaitPackets: " << resendvalues.gigeresend.FirstResendWaitPackets << endl
		<< "\t FirstResendWaitTime: " << resendvalues.gigeresend.FirstResendWaitTime << endl
		<< "\t NextResendWaitPackets: " << resendvalues.gigeresend.NextResendWaitPackets << endl
		<< "\t NextResendWaitTime: " << resendvalues.gigeresend.NextResendWaitTime << endl
		<< "\t FirstResendWaitPacketsDualLink: " << resendvalues.gigeresend.FirstResendWaitPacketsDualLink << endl
		<< "\t NextResendWaitPacketsDualLink: " << resendvalues.gigeresend.NextResendWaitPacketsDualLink << endl;

	res = pCamera->getDeviceClockFrequency(BGAPI_DEVICECLOCK_SENSOR, &state, &sensorfreq);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getDeviceClockFrequency Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Sensor freq: " << sensorfreq.current << ", Max: " << sensorfreq.maximum << ", Min: " << sensorfreq.minimum << endl;

	res = pCamera->getReadoutTime(&state, &readouttime);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getReadoutTime Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Readout time: " << readouttime.current << endl;

	return EXIT_SUCCESS;
}

int run_camera()
{
	BGAPI_RESULT res = BGAPI_RESULT_FAIL;
	BGAPI_FeatureState state; state.cbSize = sizeof(BGAPI_FeatureState);
	BGAPIX_CameraStatistic statistics; statistics.cbSize = sizeof(BGAPIX_CameraStatistic);

	res = pCamera->setFrameCounter(0, 0);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setFrameCounter Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}

	// ALLOCATE BUFFERS 
	res = pCamera->setDataAccessMode(BGAPI_DATAACCESSMODE_QUEUEDINTERN, numbuffer);
	if (res != BGAPI_RESULT_OK)
	{
		printf("BGAPI::Camera::setDataAccessMode Errorcode %d\n", res);
		return EXIT_FAILURE;
	}

	// dynamic allocation
	pImage = new BGAPI::Image * [numbuffer];

	int i = 0;
	for (i = 0; i < numbuffer; i++)
	{
		res = BGAPI::createImage(&pImage[i]);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::createImage for Image %d Errorcode %d\n", i, res);
			break;
		}
	}
	printf("Images created successful!\n");

	for (i = 0; i < numbuffer; i++)
	{
		res = pCamera->setImage(pImage[i]);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::System::setImage for Image %d Errorcode %d\n", i, res);
			break;
		}
	}
	printf("Images allocated successful!\n");

	res = pCamera->registerNotifyCallback(pCamera, (BGAPI::BGAPI_NOTIFY_CALLBACK)&imageCallback);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::registerNotifyCallback Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}

    res = pCamera->setStart(true);
    if(res != BGAPI_RESULT_OK) {
        printf("BGAPI::Camera::setStart Errorcode: %d\n",res);
		return EXIT_FAILURE;
    }
	printf("Acquisition started\n");
    
    printf("\n\n=== ENTER TO STOP ===\n\n");
	int d;
	scanf("&d",&d);
    while ((d = getchar()) != '\n' && d != EOF)

    res = pCamera->setStart(false);
    if(res != BGAPI_RESULT_OK) {
        printf("BGAPI::Camera::setStart Errorcode: %d\n",res);
		return EXIT_FAILURE;
    }

	res = pCamera->getStatistic(&state, &statistics);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getStatistic Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << endl << "Camera statistics:" << endl
		<< "  Received Frames Good: " << statistics.statistic[0] << endl
		<< "  Received Frames Corrupted: " << statistics.statistic[1] << endl
		<< "  Lost Frames: " << statistics.statistic[2] << endl
		<< "  Resend Requests: " << statistics.statistic[3] << endl
		<< "  Resend Packets: " << statistics.statistic[4] << endl
		<< "  Lost Packets: " << statistics.statistic[5] << endl
		<< "  Bandwidth: " << statistics.statistic[6] << endl 
		<< endl;		

	// release all resources ?

    res = pSystem->release();
    if(res != BGAPI_RESULT_OK) {
        printf( "BGAPI::System::release Errorcode: %d System index: %d\n", res,sys);
        return EXIT_FAILURE;
    }
    printf("System released: System index %d\n", sys);

	return EXIT_SUCCESS;
}

int exit_gracefully(int exitcode) {

	printf("\n\n=== ENTER TO CLOSE ===\n\n");
	scanf("&d");

	delete[] pImage;

	// Stop the program and release resources 
	if (!preview) {
		file.close();
		skelfile.close();
	}

	cv::destroyAllWindows();
	return exitcode;
}

void process() {

	cv::Mat current_image;
	cv::Mat img_resized;
	double current_timing = 0;
	int swc = 0;
	double fps_hat = 0;

	while (true)
	{
		if (!ImageList.empty())
		{
			mtx_buffer.lock();
			current_image = ImageList.front();
			current_timing = timeStampsList.front();
			swc = counterList.front();
			fps_hat = fpsList.front();
			mtx_buffer.unlock();

			size_t buflen = ImageList.size();
			size_t skelbuflen = SkeletonList.size();

			cv::resize(current_image, img_resized, cv::Size(), 1.0 / subsample, 1.0 / subsample);

			// compress image
			if (!preview) {
				file << swc << "\t" << setprecision(3) << std::fixed << 1000 * current_timing << std::endl;
				writer << img_resized;
			}

			// if you want to do online processing of the images, it should go here

			mtx.lock();
			img_resized.copyTo(img_display);
			mtx.unlock();

			if (((int)(current_timing) * 1000) % 100 == 0)
			{
				printf("FPS %.2f, elapsed : %d sec, bufsize %zd, skel bufsize %zd \r", fps_hat, (int)(current_timing), buflen, skelbuflen);
				fflush(stdout);
			}

			mtx_buffer.lock();
			ImageList.pop_front();
			timeStampsList.pop_front();
			counterList.pop_front();
			fpsList.pop_front();
			mtx_buffer.unlock();

			// make the image available for skeletonization
			mtx_skel.lock();
			SkeletonList.push_back(img_resized.clone());
			mtx_skel.unlock();
		}
		else {
			boost::this_thread::sleep_for(boost::chrono::milliseconds(1)); 
		}
	}
}

void skeletonize() {

	cv::Mat frame;
	cv::Mat bckg_sub;
	cv::Mat blur;
	cv::Mat fish_pad;
	cv::Mat skeleton_rgb;

	double radius_max = (double)length_tail / (double)(n_skel - 1);
	arma::vec theta = arma::linspace(-2 * M_PI / 3, 2 * M_PI / 3, 90);
	arma::vec radius = arma::linspace(1, radius_max, round(radius_max));

	while (true)
	{
		if (!SkeletonList.empty())
		{
			mtx_skel.lock();
			frame = SkeletonList.front();
			mtx_skel.unlock();

			frame.copyTo(skeleton_rgb);
			cv::cvtColor(skeleton_rgb, skeleton_rgb, cv::COLOR_GRAY2RGB);
			frame.convertTo(frame, CV_32F, 1.0 / 255.0);

			cv::absdiff(frame, background, bckg_sub);

			double min_fish, max_fish;
			cv::minMaxLoc(bckg_sub, &min_fish, &max_fish);
			bckg_sub = (bckg_sub - min_fish) / (max_fish - min_fish);

			cv::GaussianBlur(bckg_sub, blur, cv::Size(21, 21), 6, 6);
			cv::copyMakeBorder(blur, fish_pad, round(length_tail), round(length_tail), round(length_tail), round(length_tail), cv::BORDER_CONSTANT, 0);

			//cv::imshow("Bckg sub", fish_pad);

			int x_0 = points[0].x + round(length_tail);
			int y_0 = points[0].y + round(length_tail);
			double best_theta = 0;
			arma::vec theta_frame(n_skel - 1);
			arma::vec skel_x(n_skel);
			arma::vec skel_y(n_skel);
			skel_x(0) = points[0].x;
			skel_y(0) = points[0].y;

			for (int s = 1; s < n_skel; s++) {
				arma::mat Xgrid = arma::round(radius * arma::cos(theta + best_theta).t() + x_0);
				arma::mat Ygrid = arma::round(radius * arma::sin(theta + best_theta).t() + y_0);
				arma::Mat<double> pixels(radius.n_elem, theta.n_elem);
				for (int i = 0; i < radius.n_elem; i++) {
					for (int j = 0; j < theta.n_elem; j++) {
						pixels(i, j) = fish_pad.at<float>(Ygrid(i, j), Xgrid(i, j));
					}
				}

				arma::Row<double> profile = arma::sum(pixels, 0);
				arma::Row<double> gaussian = arma::exp(-arma::pow(arma::linspace(-(int)profile.n_rows / 2, profile.n_rows / 2, profile.n_rows), 2) / 8);
				gaussian = gaussian / arma::sum(gaussian);
				profile = arma::conv(profile, gaussian, "same");
				arma::uword pos_max = arma::index_max(profile);
				double best_theta = theta(pos_max);
				x_0 = x_0 + round(radius_max * cos(best_theta));
				y_0 = y_0 + round(radius_max * sin(best_theta));
				skel_x(s) = x_0 - round(length_tail);
				skel_y(s) = y_0 - round(length_tail);
				theta_frame(s - 1) = best_theta;
			}

			for (int s = 0; s < n_skel; s++) {
				cv::circle(skeleton_rgb, cv::Point(skel_x(s), skel_y(s)), 4, cv::Scalar(0, 0, 255), 1);
			}
			for (int s = 0; s < n_skel - 1; s++) {
				cv::line(skeleton_rgb, cv::Point(skel_x(s), skel_y(s)), cv::Point(skel_x(s + 1), skel_y(s + 1)), cv::Scalar(0, 0, 255));
			}

			// write to file
			for (int s = 0; s < n_skel; s++) {
				skelfile << skel_x(s) << ",";
				skelfile << skel_y(s) << ",";
			}
			for (int s = 0; s < n_skel - 1; s++) {
				if (s == n_skel - 2) {
					skelfile << setprecision(5) << theta_frame(s);
				}
				else {
					skelfile << setprecision(5) << theta_frame(s) << ",";
				}
			}
			skelfile << endl;

			mtx.lock();
			skeleton_rgb.copyTo(img_skel);
			mtx.unlock();

			mtx_skel.lock();
			SkeletonList.pop_front();
			mtx_skel.unlock();
		}
		else {
			boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
		}
	}
}

void onMouse(int evt, int x, int y, int flags, void* param) {
	if (evt == CV_EVENT_LBUTTONDOWN) {
		std::vector<cv::Point>* ptPtr = (std::vector<cv::Point>*)param;
		ptPtr->push_back(cv::Point(x, y));
	}
}

int compute_background() {

	cout << "Computing background" << endl;

	BGAPI::Image * Im = NULL;
	BGAPI_RESULT res = BGAPI_RESULT_FAIL;
	cv::Mat img;

	res = pCamera->setImagePolling(true);
	if (res != BGAPI_RESULT_OK)
	{
		printf("Error %d while set Image polling.\n", res);
	}

	//create an image 
	res = BGAPI::createImage(&Im);
	if (res != BGAPI_RESULT_OK)
	{
		printf("Error %d while creating an image.\n", res);
	}

	//set the image to the camera 
	res = pCamera->setImage(Im);
	if (res != BGAPI_RESULT_OK)
	{
		printf("Error %d while setting an image to the camera.\n", res);
	}

	cout << "Image allocated, starting camera" << endl;

	res = pCamera->setStart(true);
	if (res != BGAPI_RESULT_OK)
	{
		printf("BGAPI::Camera::setStart returned with errorcode %d\n", res);
	}

	unsigned char* imagebuffer = NULL;
	int receiveTimeout = 100;
	background = cv::Mat(cv::Size(width, height), CV_32F, cv::Scalar(0));

	for (int i = 0; i < 1000; i++)
	{
		res = pCamera->getImage(&Im, receiveTimeout);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI_Camera_getImage returned with errorcode %d\n", res);
		}
		else
		{
			Im->get(&imagebuffer);

			img = cv::Mat(cv::Size(width,height), CV_8U, imagebuffer);
			img.convertTo(img, CV_32F, 1.0 / 255.0);
			background = 1.0 / 1000 * img + background;

			//after you are ready with this image, return it to the camera for the next image
			res = pCamera->setImage(Im);
			if (res != BGAPI_RESULT_OK)
			{
				printf("setImage failed with %d\n", res);
			}
		}
	}

	//stop the camera when you are done with the capture
	res = pCamera->setStart(false);
	if (res != BGAPI_RESULT_OK)
	{
		printf("BGAPI::Camera::setStart Errorcode: %d\n", res);
	}

	cout << "Background acquisition done" << endl;

	res = pCamera->setImagePolling(false);
	if (res != BGAPI_RESULT_OK)
	{
		printf("Error %d while set Image polling.\n", res);
	}

	res = BGAPI::releaseImage(Im);
	if (res != BGAPI_RESULT_OK)
	{
		printf("releaseImage Errorcode: %d\n", res);
	}
	
	cout << "Select origin of the tail" << endl;

	cv::namedWindow("Select start");
	cv::setMouseCallback("Select start", onMouse, (void*)&points);
	while (true) {
		cv::imshow("Select start", background);
		if (points.size() > 0) {
			break;
		}
		cv::waitKey(16);
	}

	cout << "Select outline of the fish" << endl;

	cv::Rect2d fish_rect = cv::selectROI("Select fish", background, false);
	cv::Mat mask = cv::Mat::zeros(background.rows, background.cols, CV_8U);
	mask(fish_rect) = 1;
	cv::inpaint(background, mask, background, 10, CV_INPAINT_NS);

	cv::namedWindow("Background");
	char exit_key_press = 0;
	while (exit_key_press != 'q') {
		cv::imshow("Background", background);
		exit_key_press = cvWaitKey(16);
	}
	cv::destroyAllWindows();

	return 0;
}

int main(int ac, char* av[])
{
	int retcode = 0;

	// read configuration files
	int read = 1;
	read = read_config(ac, av);
	if (read == 1) {
		printf("Problem parsing options, aborting");
		return exit_gracefully(1);
	}
	else if (read == 2) {
		return exit_gracefully(0);
	}

	retcode = setup_camera();
	if (retcode == EXIT_FAILURE) {
		return exit_gracefully(EXIT_FAILURE);
	}
	printf("Camera setup complete\n");

	if (!preview) {
		// Check if result directory exists
		fs::path dir(result_dir);
		if (!exists(dir)) {
			if (!fs::create_directory(dir)) {
				cout << "unable to create result directory, aborting" << endl;
				return exit_gracefully(1);
			}
		}

		// Get formated time string 
		time_t now;
		struct tm* timeinfo;
		char buffer[100];
		time(&now);
		timeinfo = localtime(&now);
		strftime(buffer, sizeof(buffer), "%Y_%m_%d_", timeinfo);
		string timestr(buffer);

		// Check if video file exists
		fs::path video;
		stringstream ss;
		int i = 0;
		do {
			ss << setfill('0') << setw(2) << i;
			video = dir / (timestr + ss.str() + ".avi");
			i++;
			ss.str("");
		} while (exists(video));
		char videoname[100];
		wcstombs(videoname, video.c_str(), 100);

		// Check if timestamps file exists
		fs::path ts = video.replace_extension("txt");
		if (exists(ts)) {
			printf("timestamp file exists already, aborting\n");
			return exit_gracefully(1);
		}

		// Check if skeleton file exists
		ss << setfill('0') << setw(2) << i-1;
		fs::path skel = dir / (timestr + ss.str() + "_skeleton.txt");
		if (exists(skel)) {
			printf("skeleton file exists already, aborting\n");
			return exit_gracefully(1);
		}

		writer.open(videoname, cv::VideoWriter::fourcc('X', '2', '6', '4'), 24, cv::Size(width, height), 0);
		if (!writer.isOpened()) {
			printf("Problem opening Video writer, aborting\n");
			return exit_gracefully(1);
		}

		// Create timestamps file
		file.open(ts.string());
		skelfile.open(skel.string());
	}
    
	img_display = cv::Mat(height / subsample, width / subsample, CV_8UC1);
	img_skel = cv::Mat(height / subsample, width / subsample, CV_8UC3);

	// launch acquisition 
	retcode = compute_background();
	if (retcode == EXIT_FAILURE) {
		return exit_gracefully(EXIT_FAILURE);
	}

	boost::thread bt(display_preview);
	boost::thread bt1(process);
	boost::thread bt2(skeletonize);

	retcode = run_camera();
	if (retcode == EXIT_FAILURE) {
		return exit_gracefully(EXIT_FAILURE);
	}

	bt.interrupt();
	bt1.interrupt();
	bt2.interrupt();

	// Stop the program 
	return exit_gracefully(EXIT_SUCCESS);
}
