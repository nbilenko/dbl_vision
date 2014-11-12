#include <future>
#include <sstream>
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <unistd.h>
#include <libgen.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

/* print debug messages to screen when updating brain */
#define DEBUG_MESGS

/* math const */
#define PI 3.141592653589793

/* time interval between captures (usec) */
#define CAPTURE_INTERVAL 10000

/* retinotopy binning  */
#define NUM_BINS_ANG 8
#define NUM_BINS_ECC 4
#define ANG_OK_RIGHT(i) (i <= 1 || i >= 6)
#define ANG_OK_LEFT(i)  (i >= 2 && i <= 5)

/* level for a node with no data */
#define NODE_NO_DATA -1.0


/******************************************************************************
 ** Support code **************************************************************
 ******************************************************************************/

/**
 * Class performing polar binning on dual VideoCapture (left and right "visual
 * field") objects, and correspondingly updating DrB lighting over UDP
 */
class Retinotopy
{
  private:

  bool have_node_labels;
  vector<int> node_angle, node_eccentricity;

  bool initialized_binning;
  int nx, ny;
  double d_ang, d_ecc;
  vector<int> pixel_angle_left, pixel_eccentricity_left;
  vector<int> pixel_angle_right, pixel_eccentricity_right;

  bool initialized_udp;
  int sockfd;
  struct addrinfo hints, *servinfo, *servinfo_ptr;

  const char *mux_host, *mux_port;

  int
  bl_update_udp(vector<int> &mean)
  {
    // ensure we have the node labels
    assert(have_node_labels);

    // build OSC message
    ostringstream msg_stream;
    msg_stream << "/drbvision";
    for (int n = 0; n < node_angle.size(); n++) {
      msg_stream << " ";
      if (node_angle[n] < 0)
        msg_stream << NODE_NO_DATA;
      else {
        int b = node_eccentricity[n] * NUM_BINS_ANG + node_angle[n];
        msg_stream << (float) mean[b] / 255.0;
      }
    }

    // cast to c string
    string msg_string = msg_stream.str();
    const char *msg_ptr = msg_string.c_str();

#ifdef DEBUG_MESGS
    // print dummy report
    printf("\e[1;1H\e[2J");
    printf("(EC, AN)\n");
    for (int i = 0; i < NUM_BINS_ECC; i++)
      for (int j = 0; j < NUM_BINS_ANG; j++) {
        int bar_width = 40;
        int bar_len = (int)((double) mean[i * NUM_BINS_ANG + j] / (255.0 / bar_width));
        printf("(%2.2i, %2.2i): 0 |", i, j);
        int k;
        for (k = 0; k < bar_len; k++)
          printf("#");
        for (; k < bar_width; k++)
          printf(" ");
        printf("| 255\n");
      }
    printf("MESSAGE: \"%s\"\n", msg_ptr);
#endif

    // initialize network socket if not already done
    if (! initialized_udp) {
      memset(&hints, 0, sizeof(hints));
      hints.ai_family   = AF_UNSPEC;
      hints.ai_socktype = SOCK_DGRAM;

      int ret;
      if ((ret = getaddrinfo(mux_host, mux_port, &hints, &servinfo)) != 0) {
        printf("getaddrinfo: %s\n", gai_strerror(ret));
        return 1;
      }

      for (servinfo_ptr = servinfo;
           servinfo_ptr != NULL;
           servinfo_ptr = servinfo_ptr->ai_next) {
        if ((sockfd = socket(servinfo_ptr->ai_family,
                             servinfo_ptr->ai_socktype,
                             servinfo_ptr->ai_protocol)) == -1) {
          perror("cannot open socket");
          continue;
        }
        break;
      }

      if (servinfo_ptr == NULL) {
        printf("failed to bind socket\n");
        return 1;
      }

      // done
      initialized_udp = true;
    }

    // send update to node lighting multiplexer
    int numbytes;
    if ((numbytes = sendto(sockfd, msg_ptr, strlen(msg_ptr), 0,
                           servinfo_ptr->ai_addr,
                           servinfo_ptr->ai_addrlen)) == -1) {
      perror("cannot complete sendto");
      return 1;
    }

    // looks like we succeeded
    return 0;
  }

  void
  load_node_labels(const char *fname)
  {
    // already initialized?
    assert(! have_node_labels);

    FILE *f = fopen(fname, "r");
    assert(f != NULL);

    // read labels (angle, eccentricity)
    int angle, eccentricity;
    while (fscanf(f, "%i %i", &angle, &eccentricity) == 2) {
      node_angle.push_back(angle);
      node_eccentricity.push_back(eccentricity);
      assert(angle < NUM_BINS_ANG);
      assert(eccentricity < NUM_BINS_ECC);
    }
    fclose(f);

    // done
    have_node_labels = true;
  }

  int
  run_binning(Mat &image_left, Mat &image_right)
  {
    // setup
    if (! initialized_binning) {
      // ensure images have identical shape
      assert(image_left.size().height == image_right.size().height);
      assert(image_left.size().width  == image_right.size().width);

      // store image and derived bin dimensions
      ny = image_left.size().height;
      nx = image_left.size().width;
      d_ang = 2 * PI / NUM_BINS_ANG;
      d_ecc = ny / 2 / NUM_BINS_ECC;

      // precompute pixel bins ...

      // right
      for (int i = 0; i < ny; i++)
        for (int j = 0; j < ny / 2; j++) {
          // infer bin for this pixel
          double x = j, y = i - ny / 2;
          double r = sqrt(x * x + y * y);
          double theta = atan2(y, x);
          int ib = r / d_ecc;
          while (theta < 0)
            theta += 2 * PI;
          int jb = theta / d_ang;
          pixel_angle_right.push_back(jb);
          pixel_eccentricity_right.push_back(ib);
        }

      // left
      for (int i = 0; i < ny; i++)
        for (int j = nx - ny / 2; j < nx; j++) {
          // infer bin for this pixel
          double x = j - nx + 1, y = i - ny / 2;
          double r = sqrt(x * x + y * y);
          double theta = atan2(y, x);
          int ib = r / d_ecc;
          while (theta < 0)
            theta += 2 * PI;
          int jb = theta / d_ang;
          pixel_angle_left.push_back(jb);
          pixel_eccentricity_left.push_back(ib);
        }

      // done
      initialized_binning = true;
    }


    // process pixels ...

    int ipx;
    vector<int>  mean(NUM_BINS_ECC * NUM_BINS_ANG, 0);
    vector<int> count(NUM_BINS_ECC * NUM_BINS_ANG, 0);

    // right
    ipx = 0;
    for (int i = 0; i < ny; i++)
      for (int j = 0; j < ny / 2; j++) {
        int ib = pixel_eccentricity_right[ipx];
        int jb = pixel_angle_right[ipx];
        // make sure we're within a reasonable eccentricity
        if (ib < NUM_BINS_ECC && ANG_OK_RIGHT(jb)) {
          mean[ib * NUM_BINS_ANG + jb]  += image_right.at<uchar>(i,j);
          count[ib * NUM_BINS_ANG + jb] += 1;
        }
        ipx += 1;
      }

    // left
    ipx = 0;
    for (int i = 0; i < ny; i++)
      for (int j = nx - ny / 2; j < nx; j++) {
        int ib = pixel_eccentricity_left[ipx];
        int jb = pixel_angle_left[ipx];
        // make sure we're within a reasonable eccentricity
        if (ib < NUM_BINS_ECC && ANG_OK_LEFT(jb)) {
          mean[ib * NUM_BINS_ANG + jb]  += image_left.at<uchar>(i,j);
          count[ib * NUM_BINS_ANG + jb] += 1;
        }
        ipx += 1;
      }

    // normalize
    for (int i = 0; i < NUM_BINS_ECC; i++)
      for (int j = 0; j < NUM_BINS_ANG; j++)
        if (count[i * NUM_BINS_ANG + j] > 0)
          mean[i * NUM_BINS_ANG + j] /= count[i * NUM_BINS_ANG + j];

    // update the good doctor
    return bl_update_udp(mean);
  }

  public:

  Retinotopy(const char *fname_labels,
             const char *mux_host, const char *mux_port) :
    have_node_labels(false), initialized_binning(false), initialized_udp(false),
    mux_host(mux_host), mux_port(mux_port)
  {
    load_node_labels(fname_labels);
  }

  ~Retinotopy()
  {
    if (initialized_udp) {
      freeaddrinfo(servinfo);
      close(sockfd);
    }
  }

  int
  update(VideoCapture &cam_left, VideoCapture &cam_right)
  {
    Mat image_left, image_right;
    if (! cam_left.read(image_left) || ! cam_right.read(image_right))
      return 1;
    Mat image_left_gray,
        image_right_gray;
    //cvtColor(image_left,  image_left_gray,  CV_BGR2GRAY);
    //cvtColor(image_right, image_right_gray, CV_BGR2GRAY);
    //return run_binning(image_left_gray, image_right_gray);
    return run_binning(image_left, image_right);
  }
};

/******************************************************************************
 ** Main routine **************************************************************
 ******************************************************************************/

void
open_vcap(VideoCapture *vcap, const char *addr)
{
  vcap->open(addr);
}

int
main(int argc, char** argv)
{
  if (argc != 6) {
    printf("Usage: %s <left-camera> <right-camera> <node-file> <mux-host> <mux-port>\n"
           "   where:\n"
           "     * <left-camera> and <right-camera> are the URLs associated with the\n"
           "       left and right Dr. Brainlove cameras\n"
           "     * <node-file> is a text file mapping the brainlove nodes to angle and\n"
           "       eccentricity bins (two-columns of ints, one row per node)\n"
           "     * <mux-host> and <mux-port> are the hostname or IP and port to target\n"
           "       when sending updates to Sean's lighting multiplexer\n\n",
           basename(argv[0]));
    exit(1);
  }
  const char *cam_url_l = argv[1],
             *cam_url_r = argv[2],
             *node_file = argv[3],
             *mux_host  = argv[4],
             *mux_port  = argv[5];

  // if ever we lose our connection to either camera, simply abort and retart
  // same goes for errors returned by the Retinotopy obj (e.g. socket issues)
  while (1) {
    VideoCapture vcap_l, vcap_r;
    future<bool> fut_l = async([&](const char *addr){ return vcap_l.open(addr); }, cam_url_l);
    future<bool> fut_r = async([&](const char *addr){ return vcap_r.open(addr); }, cam_url_r);
    // .release() method called by destrc
    if (fut_l.get() && fut_r.get()) {
      Retinotopy ret(node_file, mux_host, mux_port);
      while (ret.update(vcap_l, vcap_r) == 0)
        usleep(CAPTURE_INTERVAL);
    }
  }

  return 0;
}
